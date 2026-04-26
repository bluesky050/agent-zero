"""
Badcase analysis for Agent-Zero.

Scans all persisted chat.json files and produces a structured report
of error patterns, warning events, tool call statistics, and loop
efficiency metrics.

Usage:
    python -m helpers.badcase_analysis [--chats-dir DIR] [--json] [--verbose]
"""

import json
import os
import sys
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _load_chat(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_ai_tool_call(content: str) -> dict | None:
    """Try to parse an AI message as a tool-call JSON."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "tool_name" in obj:
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def extract_chat_badcases(chat: dict) -> dict:
    """Extract badcase-relevant data from a single chat.json."""
    result = {
        "chat_id": chat.get("id", ""),
        "chat_name": chat.get("name", ""),
        "created_at": chat.get("created_at", ""),
        "agents": [],
        "warnings": [],
        "tool_calls": Counter(),
        "tool_errors": Counter(),
        "misformats": 0,
        "repeats": 0,
        "total_ai_messages": 0,
        "total_loops": 0,
    }

    for agent in chat.get("agents", []):
        agent_no = agent.get("number", -1)
        agent_data = agent.get("data", {})

        agent_info = {
            "agent_number": agent_no,
            "iteration_no": agent_data.get("iteration_no", 0),
            "consecutive_error_count": agent_data.get("consecutive_error_count", 0),
            "last_error_message": agent_data.get("last_error_message", ""),
        }
        result["agents"].append(agent_info)
        result["total_loops"] += agent_info["iteration_no"]

        # Parse history for tool calls and warnings
        hist_str = agent.get("history", "")
        if not hist_str:
            continue

        try:
            hist = json.loads(hist_str)
        except json.JSONDecodeError:
            continue

        for topic in hist.get("topics", []):
            for msg in topic.get("messages", []):
                if msg.get("ai"):
                    result["total_ai_messages"] += 1
                    content = msg.get("content", "")
                    tool_call = _parse_ai_tool_call(content)
                    if tool_call:
                        tn = tool_call.get("tool_name", "")
                        result["tool_calls"][tn] += 1
                        # Detect tool_args format issues
                        ta = tool_call.get("tool_args")
                        if ta is None:
                            result["tool_errors"]["missing_tool_args"] += 1
                        elif isinstance(ta, str):
                            result["tool_errors"]["tool_args_is_string"] += 1
                else:
                    content = msg.get("content", {})
                    if isinstance(content, dict):
                        # System warnings (misformat, repeat, etc.)
                        if "system_warning" in content:
                            warn_text = str(content["system_warning"])
                            result["warnings"].append({
                                "agent": agent_no,
                                "text": warn_text,
                            })
                            warn_lower = warn_text.lower()
                            if "misformat" in warn_lower:
                                result["misformats"] += 1
                            elif "repeat" in warn_lower or "same" in warn_lower:
                                result["repeats"] += 1

                        # Tool result errors
                        if "tool_result" in content:
                            tr = str(content.get("tool_result", ""))
                            if "error" in tr.lower() or "exception" in tr.lower():
                                tn = content.get("tool_name", "unknown")
                                result["tool_errors"][f"tool_error:{tn}"] += 1

    return result


def scan_all_chats(chats_dir: str) -> list[dict]:
    """Scan all chat.json files and return list of extracted badcase data."""
    results = []
    if not os.path.isdir(chats_dir):
        return results

    for entry in sorted(os.listdir(chats_dir)):
        chat_path = os.path.join(chats_dir, entry, "chat.json")
        if os.path.exists(chat_path):
            chat = _load_chat(chat_path)
            if chat:
                results.append(extract_chat_badcases(chat))
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], verbose: bool = False) -> dict:
    """Aggregate scan results into a structured report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_chats": len(results),
        "total_ai_messages": sum(r["total_ai_messages"] for r in results),
        "total_loops": sum(r["total_loops"] for r in results),
        "total_warnings": sum(len(r["warnings"]) for r in results),
        "total_misformats": sum(r["misformats"] for r in results),
        "total_repeats": sum(r["repeats"] for r in results),
        "tool_call_counts": Counter(),
        "tool_error_counts": Counter(),
        "chats_with_warnings": [],
        "high_loop_chats": [],
        "loop_efficiency": {},
    }

    # Aggregate tool stats
    for r in results:
        report["tool_call_counts"].update(r["tool_calls"])
        report["tool_error_counts"].update(r["tool_errors"])

    # Chats with warnings
    for r in results:
        if r["warnings"]:
            report["chats_with_warnings"].append({
                "chat_id": r["chat_id"],
                "chat_name": r["chat_name"],
                "warning_count": len(r["warnings"]),
                "warnings": r["warnings"] if verbose else [
                    w["text"][:100] for w in r["warnings"]
                ],
            })

    # High loop chats (>10 loops = likely stuck)
    for r in results:
        if r["total_loops"] > 10:
            report["high_loop_chats"].append({
                "chat_id": r["chat_id"],
                "chat_name": r["chat_name"],
                "loops": r["total_loops"],
                "ai_messages": r["total_ai_messages"],
                "agents": r["agents"],
            })

    # Loop efficiency
    total_msgs = report["total_ai_messages"]
    total_loops = report["total_loops"]
    if total_loops > 0:
        report["loop_efficiency"] = {
            "avg_ai_messages_per_chat": round(total_msgs / max(len(results), 1), 1),
            "avg_loops_per_chat": round(total_loops / max(len(results), 1), 1),
            "misformat_rate": round(
                report["total_misformats"] / max(total_msgs, 1) * 100, 2
            ),
            "repeat_rate": round(
                report["total_repeats"] / max(total_msgs, 1) * 100, 2
            ),
        }

    return report


def format_report_text(report: dict) -> str:
    """Format the report as a human-readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("  Agent-Zero Badcase Analysis Report")
    lines.append(f"  Generated: {report['generated_at']}")
    lines.append("=" * 60)

    # Overview
    lines.append("")
    lines.append("## Overview")
    lines.append(f"  Total chats:          {report['total_chats']}")
    lines.append(f"  Total AI messages:    {report['total_ai_messages']}")
    lines.append(f"  Total loop iterations:{report['total_loops']}")
    lines.append(f"  Total warnings:       {report['total_warnings']}")
    lines.append(f"  Misformats:           {report['total_misformats']}")
    lines.append(f"  Repeats:              {report['total_repeats']}")

    eff = report.get("loop_efficiency", {})
    if eff:
        lines.append("")
        lines.append("## Loop Efficiency")
        lines.append(f"  Avg AI msgs/chat:     {eff.get('avg_ai_messages_per_chat', 'N/A')}")
        lines.append(f"  Avg loops/chat:       {eff.get('avg_loops_per_chat', 'N/A')}")
        lines.append(f"  Misformat rate:       {eff.get('misformat_rate', 'N/A')}%")
        lines.append(f"  Repeat rate:          {eff.get('repeat_rate', 'N/A')}%")

    # Tool stats
    if report["tool_call_counts"]:
        lines.append("")
        lines.append("## Tool Call Frequency")
        for tool, count in report["tool_call_counts"].most_common():
            lines.append(f"  {tool:30s} {count}")

    if report["tool_error_counts"]:
        lines.append("")
        lines.append("## Tool Errors")
        for err, count in report["tool_error_counts"].most_common():
            lines.append(f"  {err:30s} {count}")

    # Chats with warnings
    if report["chats_with_warnings"]:
        lines.append("")
        lines.append("## Chats with Warnings")
        for cw in report["chats_with_warnings"]:
            lines.append(f"  [{cw['chat_id']}] {cw['chat_name']} ({cw['warning_count']} warnings)")
            for w in cw["warnings"]:
                text = w if isinstance(w, str) else w.get("text", str(w))
                lines.append(f"    - {text[:120]}")

    # High loop chats
    if report["high_loop_chats"]:
        lines.append("")
        lines.append("## High-Loop Chats (>10 iterations, likely stuck)")
        for hc in report["high_loop_chats"]:
            lines.append(
                f"  [{hc['chat_id']}] {hc['chat_name']} "
                f"- loops={hc['loops']} msgs={hc['ai_messages']}"
            )
            for ag in hc["agents"]:
                lines.append(
                    f"    Agent {ag['agent_number']}: iter={ag['iteration_no']} "
                    f"consec_err={ag['consecutive_error_count']}"
                )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent-Zero Badcase Analysis")
    parser.add_argument(
        "--chats-dir",
        default=os.path.join(os.getcwd(), "usr", "chats"),
        help="Path to usr/chats directory (default: ./usr/chats)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full warning text instead of truncated",
    )
    args = parser.parse_args()

    results = scan_all_chats(args.chats_dir)
    report = generate_report(results, verbose=args.verbose)

    if args.json:
        # Convert Counters to regular dicts for JSON serialization
        out = dict(report)
        out["tool_call_counts"] = dict(out["tool_call_counts"])
        out["tool_error_counts"] = dict(out["tool_error_counts"])
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(format_report_text(report))


if __name__ == "__main__":
    main()
