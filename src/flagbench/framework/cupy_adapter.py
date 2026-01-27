"""
CupyAdapter - Cupy 框架适配器

用于从 cuBLAS baseline 函数生成 Triton kernel
"""

import inspect
from typing import Any, Dict
from .adapter import FrameworkAdapter
from .generate_args import CupyGenerateArgs


class CupyAdapter(FrameworkAdapter):
    """Cupy 框架适配器 - 用于 cuBLAS baseline 函数"""

    @property
    def framework_name(self) -> str:
        """返回框架名称"""
        return "cupy"

    def get_operator_function(self, op_name: str) -> Any:
        """
        从 CUPY_OPERATORS 获取 baseline 函数

        Args:
            op_name: 算子名称，格式如 "cupy::saxpy"

        Returns:
            baseline 函数对象
        """
        # 解析格式
        if "::" not in op_name:
            raise ValueError(f"Invalid op_name format: {op_name}. Expected format: 'cupy::function_name'")

        # 导入 CUPY_OPERATORS
        from flagbench.dataset import CUPY_OPERATORS

        # 检查算子是否存在
        if op_name not in CUPY_OPERATORS:
            raise KeyError(f"Operator {op_name} not found in CUPY_OPERATORS")

        return CUPY_OPERATORS[op_name]

    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        """
        获取 cupy baseline 的签名信息

        Args:
            func: baseline 函数对象
            op_name: 算子名称

        Returns:
            签名信息字典，包含 parameters, return_annotation, func_desc
        """
        # 使用 inspect 获取函数签名
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            sig = None

        # 获取函数描述
        func_desc = f"cuBLAS baseline: {op_name}"
        if func.__doc__:
            func_desc = func.__doc__.strip()

        return {
            "signature": sig,
            "parameters": sig.parameters if sig else {},
            "return_annotation": sig.return_annotation if sig else None,
            "func_desc": func_desc,
        }

    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> CupyGenerateArgs:
        """
        创建 CupyGenerateArgs

        Args:
            op_name: 算子名称，格式如 "cupy::saxpy"
            func: baseline 函数对象
            impl_info: 实现信息

        Returns:
            CupyGenerateArgs 实例
        """
        # 获取签名信息
        sig_info = self.get_signature_info(func, op_name)

        # 获取 baseline 源代码
        baseline_code = self.get_reference_code(func, op_name)

        # 确定 BLAS 操作类型
        blas_type = self._determine_blas_type(op_name)

        # 创建 CupyGenerateArgs
        return CupyGenerateArgs(
            cupy_kernel_name=op_name,
            baseline_func=func,
            baseline_code=baseline_code,
            func_desc=sig_info["func_desc"],
            blas_operation_type=blas_type,
            impl_info=impl_info,
            from_mcp=False,
        )

    def get_reference_code(self, func: Any, op_name: str) -> str:
        """
        获取 baseline 函数源代码

        Args:
            func: baseline 函数对象
            op_name: 算子名称

        Returns:
            函数源代码字符串
        """
        try:
            source_code = inspect.getsource(func)
            return source_code.strip()
        except (OSError, TypeError) as e:
            # 如果无法获取源代码，返回函数签名
            return f"# Unable to get source code for {op_name}: {e}\n# Function: {func}"

    def _determine_blas_type(self, op_name: str) -> str:
        """
        根据函数名确定 BLAS 操作类型

        Args:
            op_name: 算子名称，格式如 "cupy::saxpy"

        Returns:
            BLAS 操作类型："Level 1" / "Level 2" / "Level 3" / "Extension"
        """
        # 提取 kernel 名称（去掉 namespace）
        kernel_name = op_name.split("::")[-1].lower()

        # 去掉精度前缀（s, d, c, z）
        # 例如: saxpy -> axpy, dgemm -> gemm
        op_base = kernel_name[1:] if len(kernel_name) > 1 and kernel_name[0] in 'sdcz' else kernel_name

        # Level 3 BLAS: 矩阵-矩阵操作（先检查，避免被 Level 2 的子串匹配）
        # gemm, symm, hemm, syrk, herk, syr2k, her2k, trmm, trsm
        level3_ops = ['gemm', 'symm', 'hemm', 'syrk', 'herk', 'syr2k', 'her2k', 'trmm', 'trsm']
        if op_base in level3_ops:
            return "Level 3"

        # Level 2 BLAS: 矩阵-向量操作
        # gemv, gbmv, hemv, hbmv, hpmv, symv, sbmv, spmv, trmv, tbmv, tpmv,
        # trsv, tbsv, tpsv, ger, geru, gerc, her, hpr, her2, hpr2, syr, spr, syr2, spr2
        level2_ops = ['gemv', 'gbmv', 'hemv', 'hbmv', 'hpmv', 'symv', 'sbmv', 'spmv',
                      'trmv', 'tbmv', 'tpmv', 'trsv', 'tbsv', 'tpsv',
                      'ger', 'geru', 'gerc', 'her', 'hpr', 'her2', 'hpr2',
                      'syr', 'spr', 'syr2', 'spr2']
        if op_base in level2_ops:
            return "Level 2"

        # Level 1 BLAS: 向量-向量操作
        # asum, axpy, copy, dot, dotc, dotu, nrm2, rot, rotg, rotm, rotmg, scal, swap
        level1_ops = ['asum', 'axpy', 'copy', 'dot', 'dotc', 'dotu', 'nrm2',
                      'rot', 'rotg', 'rotm', 'rotmg', 'scal', 'swap']
        if op_base in level1_ops:
            return "Level 1"

        # Extensions: geam, dgmm, etc.
        extension_ops = ['geam', 'dgmm']
        if op_base in extension_ops:
            return "Extension"

        # 默认返回 Extension
        return "Extension"

    def get_impl_info(self, kernel_name: str) -> Any:
        """
        获取 cupy 算子的实现信息

        对于 cupy，不需要 torch 的 impl_info，返回 None

        Args:
            kernel_name: 算子名称（不含namespace，如 "caxpy"）

        Returns:
            None
        """
        return None
