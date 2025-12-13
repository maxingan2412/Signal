#!/usr/bin/env python
"""
执行流程追踪脚本
用于追踪 Signal 项目的代码执行流程，帮助理解代码调用关系

使用方法:
    # 追踪函数调用（推荐，输出简洁）
    python scripts/trace_execution.py --mode calls --config configs/RGBNT201/Signal_test.yml

    # 追踪所有执行的行（输出详细）
    python scripts/trace_execution.py --mode lines --config configs/RGBNT201/Signal_test.yml

    # 只追踪特定模块
    python scripts/trace_execution.py --mode calls --filter modeling --config configs/RGBNT201/Signal_test.yml

    # 保存追踪结果到文件
    python scripts/trace_execution.py --mode calls --output trace_log.txt --config configs/RGBNT201/Signal_test.yml
"""

import sys
import os
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

class ExecutionTracer:
    """代码执行追踪器"""

    def __init__(self, filter_pattern=None, output_file=None, max_depth=50):
        self.filter_pattern = filter_pattern or 'Signal'
        self.output_file = output_file
        self.max_depth = max_depth
        self.depth = 0
        self.call_stack = []
        self.seen_calls = set()  # 避免重复输出

        if output_file:
            self.file_handle = open(output_file, 'w', encoding='utf-8')
        else:
            self.file_handle = None

    def log(self, message):
        """输出日志"""
        if self.file_handle:
            self.file_handle.write(message + '\n')
        else:
            print(message)

    def trace_calls(self, frame, event, arg):
        """追踪函数调用"""
        filename = frame.f_code.co_filename

        # 只追踪项目内的文件
        if self.filter_pattern not in filename or 'site-packages' in filename:
            return self.trace_calls

        if event == 'call':
            self.depth += 1
            if self.depth <= self.max_depth:
                func_name = frame.f_code.co_name
                lineno = frame.f_lineno
                # 简化路径显示
                short_path = filename.replace(PROJECT_ROOT, '').lstrip('/')

                call_sig = f"{short_path}:{lineno}:{func_name}"
                if call_sig not in self.seen_calls:
                    self.seen_calls.add(call_sig)
                    indent = "  " * (self.depth - 1)
                    self.log(f"{indent}-> {short_path}:{lineno} {func_name}()")

        elif event == 'return':
            self.depth = max(0, self.depth - 1)

        return self.trace_calls

    def trace_lines(self, frame, event, arg):
        """追踪每一行执行"""
        filename = frame.f_code.co_filename

        if self.filter_pattern not in filename or 'site-packages' in filename:
            return self.trace_lines

        if event == 'line':
            lineno = frame.f_lineno
            short_path = filename.replace(PROJECT_ROOT, '').lstrip('/')
            func_name = frame.f_code.co_name
            self.log(f"{short_path}:{lineno} in {func_name}()")

        return self.trace_lines

    def close(self):
        if self.file_handle:
            self.file_handle.close()


def run_with_trace(config_file, mode='calls', filter_pattern=None, output_file=None):
    """运行训练脚本并追踪执行流程"""

    tracer = ExecutionTracer(
        filter_pattern=filter_pattern,
        output_file=output_file
    )

    print(f"=" * 60)
    print(f"开始追踪执行流程")
    print(f"模式: {mode}")
    print(f"过滤: {filter_pattern or 'Signal'}")
    print(f"配置: {config_file}")
    if output_file:
        print(f"输出: {output_file}")
    print(f"=" * 60)

    # 设置命令行参数
    sys.argv = ['train.py', '--config_file', config_file]

    # 选择追踪模式
    if mode == 'calls':
        sys.settrace(tracer.trace_calls)
    elif mode == 'lines':
        sys.settrace(tracer.trace_lines)

    try:
        # 执行训练脚本
        exec(open(os.path.join(PROJECT_ROOT, 'train.py')).read(), {'__name__': '__main__'})
    except Exception as e:
        print(f"\n追踪过程中出现异常: {e}")
    finally:
        sys.settrace(None)
        tracer.close()

    print(f"\n{'=' * 60}")
    print("追踪完成")
    if output_file:
        print(f"结果已保存到: {output_file}")


def run_with_hunter(config_file, filter_module=None):
    """使用 hunter 库进行高级追踪"""
    try:
        import hunter
    except ImportError:
        print("请先安装 hunter: pip install hunter")
        return

    print(f"使用 hunter 追踪，过滤模块: {filter_module or 'Signal'}")

    sys.argv = ['train.py', '--config_file', config_file]

    # 配置 hunter 追踪
    hunter.trace(
        module_contains=filter_module or 'Signal',
        action=hunter.CallPrinter(repr_limit=80)
    )

    try:
        exec(open(os.path.join(PROJECT_ROOT, 'train.py')).read(), {'__name__': '__main__'})
    finally:
        hunter.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Signal 项目执行流程追踪工具')
    parser.add_argument('--mode', choices=['calls', 'lines', 'hunter'], default='calls',
                        help='追踪模式: calls=函数调用, lines=每行执行, hunter=使用hunter库')
    parser.add_argument('--config', default='configs/RGBNT201/Signal_test.yml',
                        help='配置文件路径')
    parser.add_argument('--filter', default=None,
                        help='过滤模式，只追踪包含此字符串的文件路径')
    parser.add_argument('--output', '-o', default=None,
                        help='输出文件路径，不指定则输出到终端')

    args = parser.parse_args()

    if args.mode == 'hunter':
        run_with_hunter(args.config, args.filter)
    else:
        run_with_trace(args.config, args.mode, args.filter, args.output)
