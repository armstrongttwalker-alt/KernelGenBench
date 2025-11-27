import argparse
import os
import re
from pathlib import Path

def pytest_to_custom_test_framework(content: str) -> str:
    # change pytest.mark to label decorator
    # 只替换单独的 @pytest.mark.xxx（后面跟换行符），且 xxx 必须是纯字母
    content = re.sub(r'@pytest\.mark\.([a-zA-Z_]+)\n', r'@label("\1")\n', content)
    # change pytest.mark.parametrize to @parametrize
    content = content.replace('@pytest.mark.parametrize', '@parametrize')
    return content

def ignore_pytest_skip(content: str) -> str:
    pattern = r'^(\s*)(@pytest\.mark\.skipif.*?)(?=\n\s*(?:@|def|class))'
    
    def add_comment(match):
        indent = match.group(1)
        skipif_block = match.group(2)
        # 给每一行添加 # 注释
        lines = skipif_block.split('\n')
        commented_lines = [indent + '# ' + line.lstrip() if line.strip() else line 
                          for line in lines]
        return '\n'.join(commented_lines)
    
    result = re.sub(pattern, add_comment, content, flags=re.MULTILINE | re.DOTALL)
    return result

def process_file(file_path: Path) -> None:
    """
    处理单个Python文件
    
    Args:
        file_path: Python文件路径
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 根据规则进行内容替换
    modified_content = content
    modified_content = modified_content.replace('import flag_gems', 'import flagbench')
    modified_content = modified_content.replace('from .conftest import QUICK_MODE', 'from sandbox.config import QUICK_MODE')
    modified_content = modified_content.replace('with flag_gems.use_gems():', 'with flagbench.use_gems(REGISTERED_OPS):')
    modified_content = modified_content.replace('from flag_gems.runtime import device, torch_device_fn', 'from sandbox import device, torch_device_fn')
    modified_content = modified_content.replace('flag_gems.device', 'sandbox.device')
    modified_content = modified_content.replace('from .conftest import TO_CPU', 'from sandbox.config import TO_CPU')
    modified_content = modified_content.replace('flag_gems.vendor_name', 'sandbox.vendor_name')
    # 如果需要,在文件开头添加 import sandbox
    if 'sandbox.device' in modified_content and 'import sandbox' not in modified_content:
        # 找到第一个import语句的位置,在其后添加
        lines = modified_content.split('\n')
        import_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_index = i
                break
        
        if import_index != -1:
            # 在第一个import之后插入
            lines.insert(import_index + 1, 'import sandbox')
            modified_content = '\n'.join(lines)
        else:
            # 如果没有找到import语句,在文件开头添加
            modified_content = 'import sandbox\n' + modified_content
    
    modified_content = pytest_to_custom_test_framework(modified_content)
    modified_content = ignore_pytest_skip(modified_content)
    modified_content = modified_content.replace('from .accuracy_utils import', 'from sandbox.utils.accuracy_utils import')
    modified_content = "from sandbox.verifier.test_parametrize import parametrize, label\n" + modified_content
    modified_content = "from sandbox.register import REGISTERED_OPS\n" + modified_content

    # 如果内容有变化,写回文件
    if modified_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"  已更新文件: {file_path}")
    else:
        print(f"  文件无需更新: {file_path}")
    pass


def find_and_process_python_files(directory: str) -> None:
    """
    查找目录中所有Python文件并处理
    
    Args:
        directory: 目标目录路径
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"错误: 目录 '{directory}' 不存在")
        return
    
    if not dir_path.is_dir():
        print(f"错误: '{directory}' 不是一个目录")
        return
    
    # 递归查找所有.py文件
    python_files = list(dir_path.rglob("*.py"))
    
    if not python_files:
        print(f"在目录 '{directory}' 中没有找到Python文件")
        return
    
    print(f"找到 {len(python_files)} 个Python文件")
    
    for py_file in python_files:
        print(f"处理文件: {py_file}")
        try:
            process_file(py_file)
        except Exception as e:
            print(f"处理文件 {py_file} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="处理指定目录中的所有Python文件")
    parser.add_argument("--path", type=str, help="要处理的目录路径")
    
    args = parser.parse_args()
    
    find_and_process_python_files(args.path)


if __name__ == "__main__":
    main()