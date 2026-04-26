"""
Agent-Zero 评测 CLI

提供命令行工具运行回归测试、查看报告、检查阈值。
"""

import asyncio
import json
import sys
from pathlib import Path

import click


@click.group()
def cli():
    """Agent-Zero 评测系统 CLI"""
    pass


@cli.command()
@click.option("--test-cases", "-t", required=False, default="usr/evaluation/test_cases", help="测试用例目录")
@click.option("--results", "-r", required=False, default="usr/evaluation/results", help="结果输出目录")
@click.option("--categories", "-c", multiple=True, help="只运行指定分类")
@click.option("--ids", "-i", multiple=True, help="只运行指定 ID 的用例")
@click.option("--tags", multiple=True, help="按标签筛选")
@click.option("--parallel", "-p", default=1, help="并行执行数")
@click.option("--json-output", "-j", is_flag=True, help="输出 JSON 格式")
def run(test_cases, results, categories, ids, tags, parallel, json_output):
    """运行回归测试"""
    from evaluation.regression.runner import RegressionRunner

    runner = RegressionRunner(test_cases, results)

    # 运行测试
    report = asyncio.run(runner.run_all(
        categories=list(categories) if categories else None,
        ids=list(ids) if ids else None,
        tags=list(tags) if tags else None,
        parallel=parallel,
    ))

    if json_output:
        click.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        # 使用 UTF-8 编码输出
        import sys
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        click.echo("\n" + report.to_markdown())


@cli.command()
@click.option("--results", "-r", required=True, help="结果文件路径 (summary.json)")
@click.option("--min-pass-rate", default=0.8, type=float, help="最低通过率阈值")
@click.option("--max-hallucination-rate", default=0.05, type=float, help="最高幻觉率阈值")
@click.option("--min-intention-accuracy", default=0.85, type=float, help="最低意图识别准确率阈值")
@click.option("--max-avg-iterations", default=10.0, type=float, help="最高平均循环次数阈值")
def check_threshold(results, min_pass_rate, max_hallucination_rate, min_intention_accuracy, max_avg_iterations):
    """检查评估结果是否达到阈值"""
    results_path = Path(results)
    if not results_path.exists():
        click.echo(f"[ERROR] 结果文件不存在: {results}", err=True)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    issues = []

    # 检查通过率
    pass_rate = summary.get("pass_rate", 0)
    if pass_rate < min_pass_rate:
        issues.append(f"通过率 {pass_rate:.1%} 低于阈值 {min_pass_rate:.1%}")

    # 检查幻觉率
    hallucination_rate = summary.get("hallucination_rate", 0)
    if hallucination_rate > max_hallucination_rate:
        issues.append(f"幻觉率 {hallucination_rate:.1%} 高于阈值 {max_hallucination_rate:.1%}")

    # 检查意图识别准确率
    intention_accuracy = summary.get("intention_accuracy", 1.0)
    if intention_accuracy < min_intention_accuracy:
        issues.append(f"意图识别准确率 {intention_accuracy:.1%} 低于阈值 {min_intention_accuracy:.1%}")

    # 检查平均循环次数
    avg_iterations = summary.get("avg_iterations", 0)
    if avg_iterations > max_avg_iterations:
        issues.append(f"平均循环次数 {avg_iterations:.1f} 高于阈值 {max_avg_iterations}")

    if issues:
        click.echo("[FAIL] 阈值检查失败:")
        for issue in issues:
            click.echo(f"  - {issue}")
        sys.exit(1)
    else:
        click.echo("[PASS] 所有阈值检查通过")
        click.echo(f"  通过率: {pass_rate:.1%}")
        click.echo(f"  幻觉率: {hallucination_rate:.1%}")
        click.echo(f"  意图识别准确率: {intention_accuracy:.1%}")
        click.echo(f"  平均循环次数: {avg_iterations:.1f}")


@cli.command()
@click.option("--test-cases", "-t", required=False, default="usr/evaluation/test_cases", help="测试用例目录")
def list_cases(test_cases):
    """列出所有测试用例"""
    from evaluation.regression.test_case import TestCaseLoader

    loader = TestCaseLoader(test_cases)
    cases = loader.load_all()

    if not cases:
        click.echo("未找到测试用例")
        return

    click.echo(f"共 {len(cases)} 个测试用例:\n")
    for tc in cases:
        click.echo(f"  [{tc.id}] {tc.name}")
        click.echo(f"    分类: {tc.category} | 难度: {tc.difficulty}")
        click.echo(f"    评估: 意图={tc.eval_intention} 幻觉={tc.eval_hallucination} 完成={tc.eval_completion}")
        click.echo()


@cli.command()
@click.option("--results", "-r", required=False, default="usr/evaluation/results/latest", help="结果目录")
def report(results):
    """查看最新报告"""
    results_path = Path(results)

    # 查找 summary.json
    summary_path = results_path / "summary.json"
    if not summary_path.exists():
        # 尝试在子目录中查找
        for subdir in sorted(results_path.iterdir(), reverse=True):
            if subdir.is_dir() and (subdir / "summary.json").exists():
                summary_path = subdir / "summary.json"
                break

    if not summary_path.exists():
        click.echo(f"[ERROR] 未找到报告文件", err=True)
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 生成报告
    from evaluation.metrics.models import EvaluationReport

    # 转换为报告对象
    report_obj = EvaluationReport(
        run_id=summary.get("run_id", ""),
        timestamp=summary.get("timestamp", ""),
        git_commit=summary.get("git_commit", ""),
        git_branch=summary.get("git_branch", ""),
        total_cases=summary.get("total_cases", 0),
        passed=summary.get("passed", 0),
        failed=summary.get("failed", 0),
        skipped=summary.get("skipped", 0),
        needs_review_count=summary.get("needs_review_count", 0),
        intention_accuracy=summary.get("intention_accuracy", 0),
        tool_call_success_rate=summary.get("tool_call_success_rate", 0),
        hallucination_rate=summary.get("hallucination_rate", 0),
        avg_iterations=summary.get("avg_iterations", 0),
    )

    click.echo(report_obj.to_markdown())


@cli.command()
@click.option("--chats-dir", "-c", default="usr/chats", help="对话记录目录")
@click.option("--output", "-o", default="usr/evaluation/badcase_report.json", help="输出文件")
def analyze_badcases(chats_dir, output):
    """分析对话记录中的 badcase"""
    from helpers.badcase_analysis import scan_all_chats, generate_report

    results = scan_all_chats(chats_dir)
    report = generate_report(results, verbose=True)

    # 保存报告
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": report["generated_at"],
            "total_chats": report["total_chats"],
            "total_warnings": report["total_warnings"],
            "total_misformats": report["total_misformats"],
            "total_repeats": report["total_repeats"],
            "loop_efficiency": report["loop_efficiency"],
            "tool_call_counts": dict(report["tool_call_counts"]),
            "tool_error_counts": dict(report["tool_error_counts"]),
            "chats_with_warnings": report["chats_with_warnings"],
            "high_loop_chats": report["high_loop_chats"],
        }, f, ensure_ascii=False, indent=2)

    click.echo(f"Badcase 分析完成，报告已保存到: {output}")
    click.echo(f"总对话数: {report['total_chats']}")
    click.echo(f"总警告数: {report['total_warnings']}")
    click.echo(f"格式错误: {report['total_misformats']}")
    click.echo(f"重复响应: {report['total_repeats']}")


if __name__ == "__main__":
    cli()
