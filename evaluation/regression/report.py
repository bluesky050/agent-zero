"""
报告生成器

生成评估报告，支持 Markdown 和 HTML 格式。
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from evaluation.metrics.models import EvaluationReport, EvaluationResult


class ReportGenerator:
    """报告生成器"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown(self, report: EvaluationReport) -> str:
        """生成 Markdown 报告"""
        return report.to_markdown()

    def generate_html(self, report: EvaluationReport) -> str:
        """生成 HTML 报告"""
        md_content = self.generate_markdown(report)

        # 简单的 HTML 转换
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Agent-Zero 评测报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
        }}
        h2 {{
            color: #555;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .pass {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .fail {{
            color: #f44336;
            font-weight: bold;
        }}
        .warning {{
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {self._markdown_to_html(md_content)}
    </div>
</body>
</html>"""
        return html

    def _markdown_to_html(self, md: str) -> str:
        """简单的 Markdown 到 HTML 转换"""
        html = md

        # 标题
        html = html.replace("# Agent-Zero 评测报告", "<h1>Agent-Zero 评测报告</h1>")
        html = html.replace("## ", "<h2>")
        html = html.replace("\n\n", "</h2>\n")

        # 表格
        lines = html.split("\n")
        in_table = False
        new_lines = []
        for line in lines:
            if "|" in line and not in_table:
                in_table = True
                new_lines.append("<table>")
                # 添加表头
                new_lines.append("<thead><tr>")
                cells = [c.strip() for c in line.split("|") if c.strip()]
                for cell in cells:
                    new_lines.append(f"<th>{cell}</th>")
                new_lines.append("</tr></thead><tbody>")
            elif "|" in line and in_table:
                if "---" in line:
                    continue  # 跳过分隔线
                new_lines.append("<tr>")
                cells = [c.strip() for c in line.split("|") if c.strip()]
                for cell in cells:
                    # 处理状态标记
                    if "✅" in cell:
                        cell = f'<span class="pass">{cell}</span>'
                    elif "❌" in cell:
                        cell = f'<span class="fail">{cell}</span>'
                    new_lines.append(f"<td>{cell}</td>")
                new_lines.append("</tr>")
            elif in_table and "|" not in line:
                in_table = False
                new_lines.append("</tbody></table>")
                new_lines.append(line)
            else:
                new_lines.append(line)

        html = "\n".join(new_lines)

        # 列表
        lines = html.split("\n")
        in_list = False
        new_lines = []
        for line in lines:
            if line.startswith("- ") and not in_list:
                in_list = True
                new_lines.append("<ul>")
                new_lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith("- ") and in_list:
                new_lines.append(f"<li>{line[2:]}</li>")
            elif in_list and not line.startswith("- "):
                in_list = False
                new_lines.append("</ul>")
                new_lines.append(line)
            else:
                new_lines.append(line)

        return "\n".join(new_lines)

    def save_report(self, report: EvaluationReport, format: str = "both") -> dict:
        """保存报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_dir = self.results_dir / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # JSON
        json_path = report_dir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        saved_files["json"] = str(json_path)

        # Markdown
        if format in ["markdown", "both"]:
            md_path = report_dir / "report.md"
            md_content = self.generate_markdown(report)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            saved_files["markdown"] = str(md_path)

        # HTML
        if format in ["html", "both"]:
            html_path = report_dir / "report.html"
            html_content = self.generate_html(report)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            saved_files["html"] = str(html_path)

        return saved_files