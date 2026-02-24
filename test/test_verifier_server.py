"""
Verifier Server 集成测试

测试 Server/Client 架构是否正常工作。

运行方式:
    pytest test/test_verifier_server.py -v

注意：需要在有 GPU 的机器上运行
"""

import subprocess
import time
import sys
import os
import pytest
import requests

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sandbox.server import VerifierClient

# 测试端口（避免与其他服务冲突）
TEST_PORT = 18888
TEST_SERVER = f"http://localhost:{TEST_PORT}"

# 测试用 kernel 代码（简单的 square 算子）
TEST_KERNEL_CODE = '''
import torch
import triton
import triton.language as tl


@triton.jit
def square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * x
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_square_kernel(x: torch.Tensor, out: torch.Tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    square_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


def square(x: torch.Tensor):
    """Wrapper for ATen operator: square"""
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    _launch_square_kernel(x_contig, out)
    return out
'''


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """等待 Server 启动"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


class TestVerifierServer:
    """Verifier Server 集成测试"""

    server_process = None

    @classmethod
    def setup_class(cls):
        """启动 Server（测试类开始前）"""
        print(f"\n启动 Verifier Server (port={TEST_PORT})...")

        # 启动 Server 进程
        cls.server_process = subprocess.Popen(
            [
                sys.executable, "-m", "sandbox.server.verifier_server",
                "--port", str(TEST_PORT)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.join(os.path.dirname(__file__), '..')
        )

        # 等待 Server 启动
        if not wait_for_server(TEST_SERVER, timeout=30):
            cls.teardown_class()
            pytest.fail("Server 启动超时")

        print("Server 启动成功")

    @classmethod
    def teardown_class(cls):
        """关闭 Server（测试类结束后）"""
        if cls.server_process:
            print("\n关闭 Verifier Server...")
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
            print("Server 已关闭")

    def test_01_server_health(self):
        """测试健康检查"""
        client = VerifierClient(TEST_SERVER)
        health = client.health_check()

        assert health["status"] == "healthy"
        assert "device_type" in health
        assert "timestamp" in health
        print(f"健康检查通过: device_type={health['device_type']}")

    def test_02_server_status(self):
        """测试状态查询"""
        client = VerifierClient(TEST_SERVER)
        status = client.get_status()

        # status 结构: {"device": {"device_type": ..., "total": ..., ...}, "output_dir": ...}
        assert "device" in status
        device_info = status["device"]
        assert "device_type" in device_info
        assert "total" in device_info
        assert "idle" in device_info
        print(f"状态查询通过: device_type={device_info['device_type']}, "
              f"total={device_info['total']}, idle={device_info['idle']}")

    def test_03_submit_kernel(self):
        """测试提交 kernel（核心测试）

        注意：测试 kernel 代码不完整（缺少 square_out 等函数），
        验证会失败，但这里主要测试 Server 能正确接收和处理请求。
        """
        client = VerifierClient(TEST_SERVER)

        print("提交测试 kernel...")
        result = client.submit_test(
            operator_name="square",
            kernel_code=TEST_KERNEL_CODE,
            test_set="v2_1",
            timeout=300
        )

        print(f"测试结果: {result}")
        # 验证响应结构正确
        assert "success" in result
        assert "operator" in result
        assert result["operator"] == "square"
        # 注意：success 可能为 False（kernel 不完整），但 Server 应该正常处理
        # 这里主要验证 Server 端到端通信正常
        if not result["success"]:
            print(f"Kernel 验证失败（预期行为）: {result.get('error') or result.get('traceback', '')[:200]}")


def check_server_available():
    """检查 Server 是否可用（辅助函数）"""
    try:
        client = VerifierClient(TEST_SERVER)
        health = client.health_check()
        print(f"Server 状态: {health}")
        return True
    except Exception as e:
        print(f"Server 不可用: {e}")
        return False


@pytest.mark.skip(reason="独立测试，需要手动启动 Server")
def test_server_standalone():
    """独立测试（不使用 pytest fixture）

    用于快速验证 Server 是否已在运行
    需要先手动启动 Server 再运行此测试
    """
    assert check_server_available(), "Server 不可用"


if __name__ == "__main__":
    # 直接运行时，执行独立测试
    print("=" * 60)
    print("Verifier Server 测试")
    print("=" * 60)

    if test_server_standalone():
        print("\n运行完整测试...")
        pytest.main([__file__, "-v", "-s"])
    else:
        print("\n请先启动 Server:")
        print(f"  python -m sandbox.server.verifier_server --port {TEST_PORT}")
