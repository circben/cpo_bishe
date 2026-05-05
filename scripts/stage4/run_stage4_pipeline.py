from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Stage4Config:
    task: str
    split: str
    stage3_manifest: str
    base_model: str
    ts_sft_model_entry: str
    cpo_model_entry: str
    strategyqa_ts_sft_model_entry: str
    strategyqa_cpo_model_entry: str
    max_samples: int
    start_index: int
    max_new_tokens: int
    tot_max_depth: int
    tot_candidates_per_step: int
    tot_beam_width: int
    tot_sc_votes: int
    tot_sc_temperature: float
    tot_max_samples: int
    tot_score_samples: int
    tot_workers: int
    log_interval: int
    run_cot: int
    run_tot: int
    run_ts_sft: int
    run_cpo: int
    save_nodes: int
    nodes_output_root: str
    output_prefix: str


MINIMAL_CONFIG = {
    # 任务选择: gsm8k | strategyqa | both
    "task": "strategyqa",
    "split": "test",
}


ADVANCED_CONFIG = {
    # 置空时自动选择最新的 stage3 manifest（优先新结构 outputs/checkpoints/<task>/stage3_*/stage3_manifest.json）。
    # 若 ts_sft_model_entry 与 cpo_model_entry 同时给出，则优先走这两个入口并自动生成临时 manifest。
    "stage3_manifest": "",
    # 基座模型（用于 direct-entry 模式生成评测 manifest）。
    "base_model": "/root/autodl-tmp/llm/Qwen3-8B",
    # TS-SFT 入口（可填 run 根目录 / 模型目录 / final_checkpoint 目录）。
    "ts_sft_model_entry": "outputs/checkpoints/gsm8k/stage3_tssft_20260405_165631",
    # CPO 入口（可填 run 根目录 / 模型目录 / final_checkpoint 目录）。
    "cpo_model_entry": "outputs/checkpoints/gsm8k/stage3_cpo_20260405_183114",
    # StrategyQA 的 TS-SFT 入口（可填 run 根目录 / 模型目录 / final_checkpoint 目录）。
    "strategyqa_ts_sft_model_entry": "outputs/checkpoints/strategyqa/stage3_tssft_20260421_111833",
    # StrategyQA 的 CPO 入口（可填 run 根目录 / 模型目录 / final_checkpoint 目录）。
    "strategyqa_cpo_model_entry": "outputs/checkpoints/strategyqa/stage3_cpo_20260421_102753",
    
    # CoT/TS-SFT/CPO 共用评测切片的样本数。
    "max_samples": 5,
    # ToT 仅在同一评测切片的子集上运行；0 表示与主评测同样本数。
    "tot_max_samples": 5,
    # 在指定 split 上的起始偏移。
    "start_index": 0,

    # 每条回答的最大生成长度。
    "max_new_tokens": 256,
    # ToT 搜索预算。
    "tot_max_depth": 4,
    "tot_candidates_per_step": 10,
    "tot_beam_width": 5,
    "tot_sc_votes": 3,
    "tot_sc_temperature": 0.7,
    # ToT 每次扩展时的打分调用次数。
    "tot_score_samples": 1,
    # ToT 并行 worker 数。
    "tot_workers": 1,
    
    # 进度打印间隔；1 表示每条样本都打印。
    "log_interval": 1,
    # 四种方法运行开关（0=不运行，1=运行）。
    "run_cot":      1,
    "run_tot":      1,
    "run_ts_sft":   1,
    "run_cpo":      1,
    # 是否保存四种基线推理节点（0=否，1=是）。
    "save_nodes": 1,
    # 节点输出根目录（会在其下按任务/基线分目录）。
    "nodes_output_root": "outputs/eval/nodes",
    # 输出到 outputs/eval/runs 时的文件名前缀。
    "output_prefix": "stage4_pipeline",
}


def _latest_stage3_manifest() -> Path:
    root = Path("outputs/checkpoints")
    candidates = list(root.glob("*/stage3_*/stage3_manifest.json"))
    if not candidates:
        raise SystemExit(
            "No stage3 manifest found under outputs/checkpoints/<task>/stage3_*/stage3_manifest.json"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_config() -> Stage4Config:
    task = str(MINIMAL_CONFIG.get("task", "")).strip().lower()
    if task not in {"gsm8k", "strategyqa", "both"}:
        raise SystemExit("MINIMAL_CONFIG.task must be one of: gsm8k, strategyqa, both")

    split = str(MINIMAL_CONFIG.get("split", "test")).strip().lower()
    merged = dict(ADVANCED_CONFIG)
    merged["task"] = task
    merged["split"] = split

    for key, value in MINIMAL_CONFIG.items():
        if key in {"task", "split"}:
            continue
        merged[key] = value

    merged["stage3_manifest"] = str(merged.get("stage3_manifest", "")).strip()
    merged["base_model"] = str(merged.get("base_model", "")).strip()
    merged["ts_sft_model_entry"] = str(merged.get("ts_sft_model_entry", "")).strip()
    merged["cpo_model_entry"] = str(merged.get("cpo_model_entry", "")).strip()
    merged["strategyqa_ts_sft_model_entry"] = str(merged.get("strategyqa_ts_sft_model_entry", "")).strip()
    merged["strategyqa_cpo_model_entry"] = str(merged.get("strategyqa_cpo_model_entry", "")).strip()
    merged["nodes_output_root"] = str(merged.get("nodes_output_root", "outputs/eval/nodes")).strip()
    return Stage4Config(**merged)


def _resolve_checkpoint_entry(entry_path: str, tag: str) -> Path:
    path = Path(entry_path)
    if not path.exists():
        raise SystemExit(f"{tag} entry path not found: {path}")

    if path.is_dir() and path.name == "final_checkpoint":
        return path

    direct = path / "final_checkpoint"
    if direct.exists() and direct.is_dir():
        return direct

    candidates = sorted(path.rglob("final_checkpoint"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No final_checkpoint found under {tag} entry path: {path}")
    return candidates[0]


def _build_direct_entry_manifest(cfg: Stage4Config, run_root: Path) -> Path:
    if not cfg.base_model.strip():
        raise SystemExit("base_model is required when using direct TS-SFT/CPO entries.")

    def _task_entries(task_name: str) -> tuple[str, str]:
        if task_name == "strategyqa":
            ts_entry = cfg.strategyqa_ts_sft_model_entry.strip() or cfg.ts_sft_model_entry.strip()
            cpo_entry = cfg.strategyqa_cpo_model_entry.strip() or cfg.cpo_model_entry.strip()
            return ts_entry, cpo_entry
        return cfg.ts_sft_model_entry.strip(), cfg.cpo_model_entry.strip()

    task_runs = []
    tasks = [cfg.task] if cfg.task in {"gsm8k", "strategyqa"} else ["gsm8k", "strategyqa"]
    for task in tasks:
        ts_entry, cpo_entry = _task_entries(task)
        if not ts_entry or not cpo_entry:
            raise SystemExit(f"Missing direct-entry checkpoint path for task={task}")
        ts_ckpt = _resolve_checkpoint_entry(ts_entry, f"TS-SFT ({task})")
        cpo_ckpt = _resolve_checkpoint_entry(cpo_entry, f"CPO ({task})")
        task_runs.append(
            {
                "task": task,
                "base_model": cfg.base_model,
                "ts_sft_final_checkpoint": ts_ckpt.as_posix(),
                "cpo_final_checkpoint": cpo_ckpt.as_posix(),
            }
        )

    payload = {
        "run_id": f"stage3_direct_entries_{_slug(datetime.now().isoformat())}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "stage4_direct_entries",
        "task_runs": task_runs,
    }
    out = run_root / "stage3_manifest_from_direct_entries.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _task_direct_entries(cfg: Stage4Config, task_name: str) -> tuple[str, str]:
    if task_name == "strategyqa":
        ts_entry = cfg.strategyqa_ts_sft_model_entry.strip() or cfg.ts_sft_model_entry.strip()
        cpo_entry = cfg.strategyqa_cpo_model_entry.strip() or cfg.cpo_model_entry.strip()
        return ts_entry, cpo_entry
    return cfg.ts_sft_model_entry.strip(), cfg.cpo_model_entry.strip()


def _has_direct_entries(cfg: Stage4Config) -> bool:
    tasks = [cfg.task] if cfg.task in {"gsm8k", "strategyqa"} else ["gsm8k", "strategyqa"]
    for task_name in tasks:
        ts_entry, cpo_entry = _task_direct_entries(cfg, task_name)
        if not ts_entry or not cpo_entry:
            return False
    return True


def _slug(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-")


def _default_run_id(cfg: Stage4Config) -> str:
    samples = str(cfg.max_samples) if cfg.max_samples > 0 else "all"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"stage4_{cfg.task}_{cfg.split}_s{samples}_"
        f"totd{cfg.tot_max_depth}c{cfg.tot_candidates_per_step}b{cfg.tot_beam_width}_{ts}"
    )


def _run(cmd: list[str]) -> None:
    print("=" * 90)
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段四一键流水线（当前仅含基线评测）")
    parser.add_argument("--print-config", action="store_true", help="打印最终生效配置并退出")
    parser.add_argument("--run-id", default="", help="可选运行 ID，用于 outputs/eval/pipelines/<run-id>")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa", "both"], default="", help="任务选择")
    parser.add_argument("--split", default="", help="数据集切分，例如 test")
    parser.add_argument("--manifest", default="", help="可选 stage3 manifest 路径；默认自动选最新")
    parser.add_argument("--base-model", default="", help="direct-entry 模式下的基座模型路径")
    parser.add_argument("--ts-sft-model-entry", default="", help="TS-SFT 入口路径（支持上层目录自动解析）")
    parser.add_argument("--cpo-model-entry", default="", help="CPO 入口路径（支持上层目录自动解析）")
    parser.add_argument(
        "--strategyqa-ts-sft-model-entry",
        default="",
        help="StrategyQA 的 TS-SFT 入口路径（支持上层目录自动解析）",
    )
    parser.add_argument(
        "--strategyqa-cpo-model-entry",
        default="",
        help="StrategyQA 的 CPO 入口路径（支持上层目录自动解析）",
    )

    parser.add_argument("--max-samples", type=int, default=-1, help="CoT/TS-SFT/CPO 共用样本数")
    parser.add_argument("--start-index", type=int, default=-1, help="评测切片起始偏移")
    parser.add_argument("--max-new-tokens", type=int, default=-1, help="每条回答最大生成 token 数")

    parser.add_argument("--tot-max-depth", type=int, default=-1, help="ToT 最大搜索深度")
    parser.add_argument("--tot-candidates-per-step", type=int, default=-1, help="SC-ToT 每步候选数")
    parser.add_argument("--tot-beam-width", type=int, default=-1, help="SC-ToT beam 宽度")
    parser.add_argument("--tot-sc-votes", type=int, default=-1, help="SC-ToT 投票次数")
    parser.add_argument("--tot-sc-temperature", type=float, default=-1.0, help="SC-ToT 投票温度")
    parser.add_argument("--tot-max-samples", type=int, default=-1, help="ToT 子集样本数（来自同一评测切片）")
    parser.add_argument("--tot-score-samples", type=int, default=-1, help="ToT 每步打分调用次数")
    parser.add_argument("--tot-workers", type=int, default=-1, help="ToT 并行 worker 数")
    parser.add_argument("--log-interval", type=int, default=-1, help="进度打印间隔")
    parser.add_argument("--run-cot", type=int, choices=[0, 1], default=-1, help="是否运行 CoT")
    parser.add_argument("--run-tot", type=int, choices=[0, 1], default=-1, help="是否运行 ToT")
    parser.add_argument("--run-ts-sft", type=int, choices=[0, 1], default=-1, help="是否运行 TS-SFT")
    parser.add_argument("--run-cpo", type=int, choices=[0, 1], default=-1, help="是否运行 CPO")
    parser.add_argument("--save-nodes", type=int, choices=[0, 1], default=-1, help="是否保存四种基线的推理节点")
    parser.add_argument("--nodes-output-root", default="", help="节点输出根目录")
    parser.add_argument("--nodes-run-id", default="", help="节点输出的运行ID；默认使用本次run-id")
    parser.add_argument("--output-prefix", default="", help="输出文件名前缀")
    args = parser.parse_args()

    cfg = _build_config()

    # 命令行参数覆盖配置默认值
    if args.task:
        cfg.task = args.task
    if args.split:
        cfg.split = args.split
    if args.manifest:
        cfg.stage3_manifest = args.manifest
    if args.base_model:
        cfg.base_model = args.base_model
    if args.ts_sft_model_entry:
        cfg.ts_sft_model_entry = args.ts_sft_model_entry
    if args.cpo_model_entry:
        cfg.cpo_model_entry = args.cpo_model_entry
    if args.strategyqa_ts_sft_model_entry:
        cfg.strategyqa_ts_sft_model_entry = args.strategyqa_ts_sft_model_entry
    if args.strategyqa_cpo_model_entry:
        cfg.strategyqa_cpo_model_entry = args.strategyqa_cpo_model_entry
    if args.max_samples >= 0:
        cfg.max_samples = args.max_samples
    if args.start_index >= 0:
        cfg.start_index = args.start_index
    if args.max_new_tokens >= 0:
        cfg.max_new_tokens = args.max_new_tokens
    if args.tot_max_depth >= 0:
        cfg.tot_max_depth = args.tot_max_depth
    if args.tot_candidates_per_step >= 0:
        cfg.tot_candidates_per_step = args.tot_candidates_per_step
    if args.tot_beam_width >= 0:
        cfg.tot_beam_width = args.tot_beam_width
    if args.tot_sc_votes >= 0:
        cfg.tot_sc_votes = args.tot_sc_votes
    if args.tot_sc_temperature >= 0:
        cfg.tot_sc_temperature = args.tot_sc_temperature
    if args.tot_max_samples >= 0:
        cfg.tot_max_samples = args.tot_max_samples
    if args.tot_score_samples >= 0:
        cfg.tot_score_samples = args.tot_score_samples
    if args.tot_workers >= 0:
        cfg.tot_workers = args.tot_workers
    if args.log_interval >= 0:
        cfg.log_interval = args.log_interval
    if args.run_cot >= 0:
        cfg.run_cot = args.run_cot
    if args.run_tot >= 0:
        cfg.run_tot = args.run_tot
    if args.run_ts_sft >= 0:
        cfg.run_ts_sft = args.run_ts_sft
    if args.run_cpo >= 0:
        cfg.run_cpo = args.run_cpo
    if args.save_nodes >= 0:
        cfg.save_nodes = args.save_nodes
    if args.nodes_output_root:
        cfg.nodes_output_root = args.nodes_output_root
    if args.output_prefix:
        cfg.output_prefix = args.output_prefix

    if cfg.run_cot + cfg.run_tot + cfg.run_ts_sft + cfg.run_cpo <= 0:
        raise SystemExit("At least one method must be enabled among run_cot/run_tot/run_ts_sft/run_cpo")

    if args.print_config:
        print("[minimal]")
        for k, v in MINIMAL_CONFIG.items():
            print(f"{k}={v}")
        print("[advanced]")
        for k, v in ADVANCED_CONFIG.items():
            print(f"{k}={v}")
        print("[effective]")
        for k, v in asdict(cfg).items():
            print(f"{k}={v}")
        return

    run_id = args.run_id.strip() or _default_run_id(cfg)
    nodes_run_id = args.nodes_run_id.strip() or _slug(run_id)
    run_root = Path("outputs/eval/pipelines") / _slug(run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    direct_entry_mode = _has_direct_entries(cfg)
    if direct_entry_mode:
        manifest_path = _build_direct_entry_manifest(cfg, run_root)
        tasks = [cfg.task] if cfg.task in {"gsm8k", "strategyqa"} else ["gsm8k", "strategyqa"]
        for task_name in tasks:
            ts_entry, cpo_entry = _task_direct_entries(cfg, task_name)
            print(f"[stage4] using direct entries for {task_name}: ts_sft={ts_entry}, cpo={cpo_entry}")
        print(f"[stage4] generated manifest={manifest_path.as_posix()}")
    else:
        stage3_manifest = cfg.stage3_manifest.strip()
        if stage3_manifest:
            manifest_path = Path(stage3_manifest)
        else:
            manifest_path = _latest_stage3_manifest()
        if not manifest_path.exists():
            raise SystemExit(f"Stage3 manifest not found: {manifest_path}")

    py = sys.executable
    cmd = [
        py,
        "scripts/stage4/run_stage4_eval.py",
        "--manifest",
        manifest_path.as_posix(),
        "--split",
        cfg.split,
        "--max-samples",
        str(cfg.max_samples),
        "--start-index",
        str(cfg.start_index),
        "--max-new-tokens",
        str(cfg.max_new_tokens),
        "--tot-max-depth",
        str(cfg.tot_max_depth),
        "--tot-candidates-per-step",
        str(cfg.tot_candidates_per_step),
        "--tot-beam-width",
        str(cfg.tot_beam_width),
        "--tot-sc-votes",
        str(cfg.tot_sc_votes),
        "--tot-sc-temperature",
        str(cfg.tot_sc_temperature),
        "--tot-max-samples",
        str(cfg.tot_max_samples),
        "--tot-score-samples",
        str(cfg.tot_score_samples),
        "--tot-workers",
        str(cfg.tot_workers),
        "--log-interval",
        str(cfg.log_interval),
        "--run-cot",
        str(cfg.run_cot),
        "--run-tot",
        str(cfg.run_tot),
        "--run-ts-sft",
        str(cfg.run_ts_sft),
        "--run-cpo",
        str(cfg.run_cpo),
        "--save-nodes",
        str(cfg.save_nodes),
        "--nodes-output-root",
        cfg.nodes_output_root,
        "--nodes-run-id",
        nodes_run_id,
        "--output-prefix",
        f"{cfg.output_prefix}_{_slug(run_id)}",
    ]
    if cfg.task in {"gsm8k", "strategyqa"}:
        cmd.extend(["--task", cfg.task])

    _run(cmd)

    stage4_manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": "stage4",
        "modules": ["baselines_4_plus_cpo"],
        "config": asdict(cfg),
        "stage3_manifest": manifest_path.as_posix(),
        "run_root": run_root.as_posix(),
        "notes": {
            "result_files": (
                "Per-task JSON/CSV are saved under outputs/eval/runs/ with prefix "
                f"{cfg.output_prefix}_{_slug(run_id)}_*"
            )
        },
    }
    out = run_root / "stage4_manifest.json"
    out.write_text(json.dumps(stage4_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage4] run_id={run_id}")
    print(f"[stage4] pipeline_manifest={out.as_posix()}")


if __name__ == "__main__":
    main()
