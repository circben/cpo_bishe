"""
ToT 评估脚本 - 交互接口

功能：
1. 加载评估数据
2. 运行 ToT 搜索
3. 保存结果到 JSON/CSV
4. 输出性能统计
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cpo.eval.metrics_gsm8k import exact_match
from cpo.llm.generation import GenerationRuntime
from cpo.tot.sc_tot_search import (
    ToTSearchConfig,
    ToTNodeResult,
    run_tot_search,
    backtrack_path,
    extract_final_answer,
)


# ============================================================================
# 默认配置
# ============================================================================

MINIMAL_CONFIG = {
    "task": "gsm8k",
    "split": "test",
}

ADVANCED_CONFIG = {
    # 模型配置
    "base_model": "/root/autodl-tmp/llm/Qwen3-8B",

    # 评测范围
    "max_samples": 20,
    "start_index": 0,

    # ToT 搜索配置（论文标准）
    "tot_max_depth": 5,
    "tot_candidates_per_step": 10,
    "tot_beam_width": 5,

    # SC 投票配置
    "tot_sc_votes": 5,
    "tot_sc_temperature": 0.7,

    # ToT 子集样本数，0 表示与 max_samples 相同
    "tot_max_samples": 0,

    # 输出配置
    "log_interval": 1,
    "output_prefix": "stage4_tot_only",
}


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class TotEvalConfig:
    """评估配置"""

    task: str
    split: str
    base_model: str
    max_samples: int
    start_index: int
    tot_max_depth: int
    tot_candidates_per_step: int
    tot_beam_width: int
    tot_sc_votes: int
    tot_sc_temperature: float
    tot_max_samples: int
    log_interval: int
    output_prefix: str

    def to_search_config(self) -> ToTSearchConfig:
        """转换为 ToT 搜索配置"""
        return ToTSearchConfig(
            max_depth=self.tot_max_depth,
            candidates_per_step=self.tot_candidates_per_step,
            beam_width=self.tot_beam_width,
            model_name=self.base_model,
            strict_llm=True,
            sc_votes=self.tot_sc_votes,
            sc_temperature=self.tot_sc_temperature,
        )


# ============================================================================
# 工具函数
# ============================================================================


def _extract_gsm8k_gold(answer: str) -> str:
    """从 GSM8K 答案中提取标准答案"""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    if match:
        return match.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    return nums[-1] if nums else answer.strip().lower()


def _normalize_binary(text: str) -> str:
    """标准化二值答案"""
    s = text.strip().lower()
    if s in {"1", "true", "yes", "true"}:
        return "yes"
    if s in {"0", "false", "no", "false"}:
        return "no"
    if "yes" in s:
        return "yes"
    if "no" in s:
        return "no"
    return s


def _is_correct(task: str, pred: str, gold: str) -> bool:
    """判断预测是否正确"""
    if task == "gsm8k":
        return bool(exact_match(pred, gold))
    return _normalize_binary(pred) == _normalize_binary(gold)


def _preview_text(text: str, max_len: int = 200) -> str:
    """预览文本（用于日志）"""
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "..."


def _should_log_progress(idx: int, total: int, log_interval: int) -> bool:
    """判断是否应该打印进度"""
    if total <= 0:
        return False
    if idx == 0 or idx + 1 == total:
        return True
    return log_interval > 0 and (idx + 1) % log_interval == 0


# ============================================================================
# 数据加载
# ============================================================================


def _load_eval_examples(
    task: str,
    split: str,
    max_samples: int,
    start_index: int,
) -> list[dict[str, Any]]:
    """加载评估数据"""
    from datasets import load_from_disk

    if task == "gsm8k":
        split_file = Path("data/raw/gsm8k") / split / f"{split}.jsonl"
        if split_file.exists():
            rows = [
                json.loads(line)
                for line in split_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            ds = load_from_disk("data/raw/gsm8k")
            chosen_split = split if split in ds else "test"
            rows = [dict(ds[chosen_split][i]) for i in range(len(ds[chosen_split]))]
    else:  # strategyqa
        split_file = Path("data/raw/strategyqa") / split / f"{split}.jsonl"
        if split_file.exists():
            rows = [
                json.loads(line)
                for line in split_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            legacy_split_file = Path("data/raw/strategyqa/splits") / f"{split}.jsonl"
            if legacy_split_file.exists():
                rows = [
                    json.loads(line)
                    for line in legacy_split_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                ds = load_from_disk("data/raw/strategyqa")
                chosen_split = split if split in ds else "train"
                rows = [dict(ds[chosen_split][i]) for i in range(len(ds[chosen_split]))]

    end = min(start_index + max_samples, len(rows)) if max_samples > 0 else len(rows)
    return rows[start_index:end]


def _row_to_qg(task: str, row: dict[str, Any]) -> tuple[str, str]:
    """从数据行提取 question 和 gold"""
    question = str(row.get("question", "")).strip()
    if task == "gsm8k":
        gold = _extract_gsm8k_gold(str(row.get("answer", "")))
    else:
        gold = _normalize_binary(str(row.get("answer", "")))
    return question, gold


# ============================================================================
# 结果序列化
# ============================================================================


def _node_to_dict(node: ToTNodeResult) -> dict[str, Any]:
    """将节点转换为字典"""
    return {
        "node_id": node.node_id,
        "depth": node.depth,
        "thought_preview": _preview_text(node.thought, 120),
        "state_preview": _preview_text(node.state, 300),
        "score": node.score,
        "is_terminal": node.is_terminal,
        "sc_score": node.sc_score,
        "sc_answer": node.sc_answer,
        "sc_vote_map": node.sc_vote_map,
        "sc_valid_votes": node.sc_valid_votes,
    }


# ============================================================================
# 核心评估逻辑
# ============================================================================


def _run_tot_eval(cfg: TotEvalConfig, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    运行 ToT 评估

    返回：
    - summary: 汇总统计
    - details: 每条样本的详细结果
    """
    # 确定 ToT 子集大小
    tot_rows = rows if cfg.tot_max_samples <= 0 else rows[: min(cfg.tot_max_samples, len(rows))]
    print(f"ToT subset size: {len(tot_rows)} / {len(rows)}")

    # 搜索配置
    search_config = cfg.to_search_config()

    # 统计变量
    details: list[dict[str, Any]] = []
    correct_count = 0
    total_latency = 0.0
    abstain_count = 0

    t0 = time.perf_counter()

    for idx, row in enumerate(tot_rows):
        q, gold = _row_to_qg(cfg.task, row)
        sample_start = time.perf_counter()

        # 运行 ToT 搜索
        result = run_tot_search(
            question=q,
            task=cfg.task,
            config=search_config,
        )

        # 提取最终答案
        pred = extract_final_answer(cfg.task, result.best_node)
        latency = time.perf_counter() - sample_start

        # 判断正确性
        correct = _is_correct(cfg.task, pred, gold)
        correct_count += int(correct)

        # 记录弃权
        abstain = not bool(pred)
        abstain_count += int(abstain)

        # 回溯路径
        path = backtrack_path(result.nodes, result.best_node.node_id)

        # 构建详细结果
        detail = {
            "index": idx,
            "question_preview": _preview_text(q, 150),
            "gold": gold,
            "prediction": pred,
            "correct": correct,
            "abstain": abstain,
            "latency_sec": round(latency, 3),

            # 树统计
            "total_nodes": result.total_nodes,
            "terminal_nodes": len(result.terminal_nodes),
            "max_depth_reached": result.max_depth_reached,

            # 最佳节点信息
            "best_node_id": result.best_node.node_id,
            "best_node_depth": result.best_node.depth,
            "best_node_score": result.best_node.score,
            "best_is_terminal": result.best_node.is_terminal,
            "best_thought_preview": _preview_text(result.best_node.thought, 120),
            "best_state_preview": _preview_text(result.best_node.state, 300),

            # SC 投票信息
            "sc_score": result.best_node.sc_score,
            "sc_answer": result.best_node.sc_answer,
            "sc_valid_votes": result.best_node.sc_valid_votes,
            "sc_vote_map": result.best_node.sc_vote_map,

            # 路径信息
            "path_length": len(path),
            "path_preview": [_preview_text(n.thought, 80) for n in path if n.node_id != "root"],

            # 所有节点预览（用于可视化）
            "nodes_preview": [_node_to_dict(n) for n in result.nodes if n.node_id != "root"],
        }

        details.append(detail)
        total_latency += latency

        # 打印进度
        if _should_log_progress(idx, len(tot_rows), cfg.log_interval):
            avg_acc = correct_count / (idx + 1)
            avg_latency = total_latency / (idx + 1)
            print(
                f"[ToT] idx={idx + 1}/{len(tot_rows)} "
                f"acc={avg_acc:.4f} "
                f"avg_latency={avg_latency:.2f}s "
                f"this_latency={latency:.2f}s",
                flush=True,
            )

    total_time = time.perf_counter() - t0

    # 构建汇总
    summary = {
        "method": "tot",
        "model": cfg.base_model,
        "count": len(tot_rows),
        "accuracy": round(correct_count / len(tot_rows), 4) if tot_rows else 0.0,
        "avg_latency_sec": round(total_latency / len(tot_rows), 3) if tot_rows else 0.0,
        "abstain_count": abstain_count,
        "max_depth": cfg.tot_max_depth,
        "candidates_per_step": cfg.tot_candidates_per_step,
        "beam_width": cfg.tot_beam_width,
        "sc_votes": cfg.tot_sc_votes,
        "subset_of_count": len(rows),
    }

    return summary, details


# ============================================================================
# 输出保存
# ============================================================================


def _save_outputs(
    cfg: TotEvalConfig,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """保存评估结果"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/eval/runs") / f"{cfg.output_prefix}_{stamp}" / cfg.task
    run_dir.mkdir(parents=True, exist_ok=True)

    # JSON 输出（包含详细信息）
    json_path = run_dir / f"{cfg.output_prefix}_{cfg.task}_{cfg.split}.json"
    payload = {
        "task": cfg.task,
        "split": cfg.split,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "summary": summary,
        "details": details,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV 输出（只包含汇总）
    csv_path = run_dir / f"{cfg.output_prefix}_{cfg.task}_{cfg.split}.csv"
    fields = [
        "method",
        "model",
        "count",
        "accuracy",
        "avg_latency_sec",
        "max_depth",
        "candidates_per_step",
        "beam_width",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: summary.get(k, "") for k in fields})

    return json_path, csv_path


# ============================================================================
# 配置构建
# ============================================================================


def _build_config(args: argparse.Namespace) -> TotEvalConfig:
    """从默认配置和命令行参数构建评估配置"""

    # 从 MINIMAL_CONFIG 开始
    task = str(MINIMAL_CONFIG.get("task", "gsm8k")).strip().lower()
    split = str(MINIMAL_CONFIG.get("split", "test")).strip().lower()

    # 合并 ADVANCED_CONFIG
    merged = dict(ADVANCED_CONFIG)
    merged["task"] = task
    merged["split"] = split

    # 应用命令行参数
    if args.task:
        merged["task"] = args.task
    if args.split:
        merged["split"] = args.split
    if args.base_model:
        merged["base_model"] = args.base_model
    if args.max_samples >= 0:
        merged["max_samples"] = args.max_samples
    if args.start_index >= 0:
        merged["start_index"] = args.start_index
    if args.tot_max_depth >= 0:
        merged["tot_max_depth"] = args.tot_max_depth
    if args.tot_candidates_per_step >= 0:
        merged["tot_candidates_per_step"] = args.tot_candidates_per_step
    if args.tot_beam_width >= 0:
        merged["tot_beam_width"] = args.tot_beam_width
    if args.tot_sc_votes >= 0:
        merged["tot_sc_votes"] = args.tot_sc_votes
    if args.tot_sc_temperature >= 0:
        merged["tot_sc_temperature"] = args.tot_sc_temperature
    if args.tot_max_samples >= 0:
        merged["tot_max_samples"] = args.tot_max_samples
    if args.log_interval >= 0:
        merged["log_interval"] = args.log_interval
    if args.output_prefix:
        merged["output_prefix"] = args.output_prefix

    # 验证
    if merged["task"] not in {"gsm8k", "strategyqa"}:
        raise SystemExit(f"Invalid task: {merged['task']}. Must be gsm8k or strategyqa.")
    if not merged["base_model"].strip():
        raise SystemExit("base_model is required.")

    return TotEvalConfig(**merged)


def _print_config(cfg: TotEvalConfig) -> None:
    """打印配置信息"""
    print("[minimal]")
    for k, v in MINIMAL_CONFIG.items():
        print(f"  {k} = {v}")
    print("[advanced]")
    for k, v in ADVANCED_CONFIG.items():
        print(f"  {k} = {v}")
    print("[effective]")
    for k, v in asdict(cfg).items():
        print(f"  {k} = {v}")


# ============================================================================
# 主入口
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ToT 评估脚本 - 基于论文标准的 Tree of Thoughts 实现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run_tot_eval_only.py --task gsm8k --base-model /path/to/model
  python run_tot_eval_only.py --task strategyqa --tot-max-depth 4 --tot-sc-votes 3
  python run_tot_eval_only.py --print-config
        """,
    )

    # 基础参数
    parser.add_argument("--print-config", action="store_true", help="打印配置并退出")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], help="评测任务")
    parser.add_argument("--split", default="", help="数据集划分 (test/train)")

    # 模型参数
    parser.add_argument("--base-model", default="", help="基座模型路径或名称")

    # 数据范围参数
    parser.add_argument("--max-samples", type=int, default=-1, help="最大样本数")
    parser.add_argument("--start-index", type=int, default=-1, help="起始索引")

    # ToT 搜索参数
    parser.add_argument("--tot-max-depth", type=int, default=-1, help="最大推理深度 (论文标准: 5)")
    parser.add_argument("--tot-candidates-per-step", type=int, default=-1, help="每步候选数 (论文标准: 10)")
    parser.add_argument("--tot-beam-width", type=int, default=-1, help="Beam 宽度 (论文标准: 5)")

    # SC 投票参数
    parser.add_argument("--tot-sc-votes", type=int, default=-1, help="SC 投票次数 (论文标准: 5)")
    parser.add_argument("--tot-sc-temperature", type=float, default=-1.0, help="SC 采样温度 (论文标准: 0.7)")

    # 子集参数
    parser.add_argument("--tot-max-samples", type=int, default=-1, help="ToT 子集大小")

    # 输出参数
    parser.add_argument("--log-interval", type=int, default=-1, help="日志打印间隔")
    parser.add_argument("--output-prefix", default="", help="输出文件前缀")

    args = parser.parse_args()

    # 构建配置
    cfg = _build_config(args)

    # 打印配置并退出
    if args.print_config:
        _print_config(cfg)
        return

    # 加载数据
    print(f"Loading data: task={cfg.task}, split={cfg.split}, "
          f"start={cfg.start_index}, count={cfg.max_samples}")
    rows = _load_eval_examples(cfg.task, cfg.split, cfg.max_samples, cfg.start_index)
    if not rows:
        raise SystemExit("No evaluation rows loaded.")

    print(f"Loaded {len(rows)} rows")

    # 运行评估
    print(f"Running ToT eval with config:")
    print(f"  max_depth={cfg.tot_max_depth}, "
          f"candidates={cfg.tot_candidates_per_step}, "
          f"beam_width={cfg.tot_beam_width}")
    print(f"  sc_votes={cfg.tot_sc_votes}, sc_temperature={cfg.tot_sc_temperature}")

    summary, details = _run_tot_eval(cfg, rows)

    # 保存结果
    json_path, csv_path = _save_outputs(cfg, summary, details)

    # 打印汇总
    print("\n" + "=" * 60)
    print("ToT Evaluation Summary")
    print("=" * 60)
    print(f"Task: {cfg.task}")
    print(f"Model: {summary['model']}")
    correct_num = sum(1 for d in details if d.get("correct"))
    print(f"Accuracy: {summary['accuracy']:.4f} ({correct_num}/{summary['count']})")
    print(f"Avg Latency: {summary['avg_latency_sec']:.3f}s")
    print(f"Abstain Count: {summary['abstain_count']}")
    avg_nodes = (sum(d.get("total_nodes", 0) for d in details) / len(details)) if details else 0.0
    print(f"Avg Nodes: {avg_nodes:.1f}")
    print("=" * 60)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
