from rich.logging import RichHandler
from rich._log_render import LogRender, FormatTimeCallable
from rich.text import Text, TextType
from rich.console import Console, ConsoleRenderable, RenderableType
from typing import Iterable, List, Optional, TYPE_CHECKING, Union, Callable
from datetime import datetime
import re
import importlib
import itertools
from rich.containers import Renderables
from rich.table import Table


def add_register_decorator(code: str, operator: str, namespace: str = None, api: str = None) -> str:
    op_func_name = operator.replace(".", "_")
    pattern = f"def {op_func_name}("
    api = api if api else operator
    parts = code.rsplit(pattern, 1)
    code = f'{pattern}'.join([
        parts[0] + f'@register("{api}", "{operator}", False)\n', parts[-1]
    ]) if namespace is None else f'{pattern}'.join([
        parts[0] + f'@register("{api}", "{operator}", False, namespace="{namespace}")\n', parts[-1]
    ])
    return code

def expand_params(slots):
    # 每个 slot 的值是 [(v1,...), (v2,...)] 这样的绑定值
    # 单参数 slot 会存成 [v1, v2]，所以我们转换成元组保证一致
    slot_values = []
    for slot in slots:
        names = slot["names"]
        vals = slot["values"]
        if len(names) == 1:
            vals = [(v,) for v in vals]  # 单值转成元组
        slot_values.append((names, vals))

    # slot 间笛卡尔积
    results = []
    for combo in itertools.product(*[vals for _, vals in slot_values]):
        merged = {}
        for (names, _), vals in zip(slot_values, combo):
            merged.update(dict(zip(names, vals)))
        results.append(merged)

    return results

def generate_speedup_html(speedup_data, title="性能对比结果", language="zh_CN"):
    console = Console(record=True)
    print(speedup_data)
    speedup_str = "Speedup"
    match language:
        case "zh_CN":
            speedup_str = "加速比"
        case "en_US":
            speedup_str = "Speedup"
        case _:
            speedup_str = "Speedup"

    # 创建主表格
    table = Table(
        title="",
        show_header=True,
        header_style="bold bright_magenta",
        border_style="bright_blue",
        title_style="bold bright_cyan"
    )

    # 动态分析参数结构，获取所有参数键
    all_param_keys = set()
    for item in speedup_data:
        if item.get("params") != "avg" and isinstance(item.get("params"), dict):
            all_param_keys.update(item["params"].keys())

    # 排序参数键，确保显示顺序一致
    param_keys = sorted(list(all_param_keys))

    # 添加列：加速比 + 参数列 + 时间列
    table.add_column(speedup_str, style="bold bright_white", justify="right", width=10)
    
    # 为每个参数键添加一列
    for key in param_keys:
        table.add_column(key, style="bold bright_yellow", width=10)
    
    table.add_column("PyTorch (ms)", style="bold bright_green", justify="right", width=14)
    table.add_column("Triton (ms)", style="bold bright_green", justify="right", width=14)

    # 添加数据行
    for item in speedup_data:
        if item.get("params") == "avg":
            continue  # 跳过平均值，稍后单独处理

        params = item.get("params", {})
        
        # 转换时间为毫秒
        ref_time_ms = item["ref_time"] * 1000
        res_time_ms = item["res_time"] * 1000
        speedup = item["speedup"]

        # 根据加速比设置颜色
        if speedup > 1:
            speedup_color = "bold bright_green"
        else:
            speedup_color = "bold bright_red"

        # 构建行数据
        row_data = [Text(f"{speedup:.3f}", style=speedup_color)]  # 加速比

        # 添加参数值
        for key in param_keys:
            if isinstance(params, dict):
                value = params.get(key, "—")
                # 处理torch类型显示
                if "torch." in str(value):
                    value = str(value).replace("torch.", "")
                row_data.append(str(value))
            else:
                row_data.append("—")

        # 添加时间数据
        row_data.extend([
            f"{ref_time_ms:.3f}",
            f"{res_time_ms:.3f}",
        ])

        table.add_row(*row_data)

    # 添加平均值行
    avg_data = next((item for item in speedup_data if item.get("params") == "avg"), None)
    if avg_data:
        table.add_section()  # 添加分隔线
        avg_speedup = avg_data["speedup"]
        avg_ref_time_ms = avg_data["ref_time"] * 1000
        avg_res_time_ms = avg_data["res_time"] * 1000

        if avg_speedup > 1:
            avg_color = "bold bright_green"
        else:
            avg_color = "bold bright_red"

        # 构建平均值行
        avg_row = [Text(f"{avg_speedup:.3f}", style=avg_color)]  # 加速比
        avg_row.extend([Text("—", style="dim")] * len(param_keys))  # 参数列用破折号填充
        avg_row.extend([
            f"{avg_ref_time_ms:.3f}",
            f"{avg_res_time_ms:.3f}",
        ])

        table.add_row(*avg_row)

    # 用 console 记录表格
    console.print(table)

    # 导出 HTML
    html_content = console.export_html(inline_styles=True, clear=False)


    # 包装成完整 HTML
    html_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title></title>
        <style>
            body {{
                background-color: #0a0a0a;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', 'Lucida Console', monospace;
                padding: 20px;
                line-height: 1.6;
            }}
            pre {{
                background-color: #1a1a1a;
                padding: 20px;
                border-radius: 12px;
                overflow-x: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                border: 1px solid #333;
            }}
            /* 增强颜色对比度 */
            .ansi-bright-green-fg {{ color: #00ff00 !important; }}
            .ansi-bright-red-fg {{ color: #ff4444 !important; }}
            .ansi-bright-cyan-fg {{ color: #44ffff !important; }}
            .ansi-bright-yellow-fg {{ color: #ffff44 !important; }}
            .ansi-bright-blue-fg {{ color: #4444ff !important; }}
            .ansi-bright-magenta-fg {{ color: #ff44ff !important; }}
            .ansi-bright-white-fg {{ color: #ffffff !important; }}
        </style>
    </head>
    <body>
        <h1></h1>
        {html_content}
    </body>
    </html>
    """
    return html_page

class CustomLogRender(LogRender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(
        self,
        console: "Console",
        renderables: Iterable["ConsoleRenderable"],
        log_time: Optional[datetime] = None,
        time_format: Optional[Union[str, FormatTimeCallable]] = None,
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
    ) -> "Table":
        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_path and path:
            output.add_column(style="log.path")
        row: List["RenderableType"] = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        if self.show_level:
            row.append(level)

        row.append(Renderables(renderables))
        if self.show_path and path:
            path_text = Text()
            path_text.append(
                path, style=f"link {link_path}" if link_path else ""
            )
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link {link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)

        output.add_row(*row)
        return output


class CustomRichHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_render = CustomLogRender(
            show_time=True,
            show_level=True,
            show_path=True,
            time_format="[%x %X]",
            omit_repeated_times=True,
            level_width=None,
        )


def save_benchmark_result(bench_result: "BenchmarkResult", save_path: Optional[str] = None):
    """
    save benchmark result to txt file
    """
    with open(save_path, "w") as f:
        f.write(str(bench_result))
    if save_path:
        print(f"Benchmark result saved to {save_path}")