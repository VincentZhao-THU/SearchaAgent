import argparse
import json
import math
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_INPUT_PATH = "log/batch_infer_entropy_testsets_grpo_140_max500_s3/batch_infer_entropy_testsets.jsonl"
DEFAULT_OUTPUT_DIR = "log/batch_infer_entropy_testsets_grpo_140_max500_s3"
DEFAULT_MAX_TURNS = 4
DEFAULT_WRAP_WIDTH = 70
DEFAULT_ROLLING_WINDOW = 5
DEFAULT_EMA_ALPHA = 0.1
DEFAULT_TRIGGER_SHORT_WINDOW = 5
DEFAULT_TRIGGER_LONG_WINDOW = 20
DEFAULT_TRIGGER_TAIL_TOKENS = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze batch token-entropy results and generate plots.")
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--token-wrap-width", type=int, default=DEFAULT_WRAP_WIDTH)
    parser.add_argument("--rolling-window-size", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA)
    parser.add_argument("--trigger-short-window", type=int, default=DEFAULT_TRIGGER_SHORT_WINDOW)
    parser.add_argument("--trigger-long-window", type=int, default=DEFAULT_TRIGGER_LONG_WINDOW)
    parser.add_argument("--trigger-tail-tokens", type=int, default=DEFAULT_TRIGGER_TAIL_TOKENS)
    return parser.parse_args()


def load_records(input_path: Path):
    records = []
    with input_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def sanitize_source_name(source: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source)


def join_turn_tokens(turn):
    return "".join(turn["generated_tokens"]).replace("\n", "\\n")


def _join_tokens_with_offsets(tokens):
    pieces = []
    offsets = []
    current = 0
    for token in tokens:
        offsets.append(current)
        pieces.append(token)
        current += len(token)
    return "".join(pieces), offsets


def extract_think_token_indices(turn):
    tokens = turn.get("generated_tokens", [])
    if not tokens:
        return []

    text, offsets = _join_tokens_with_offsets(tokens)
    think_spans = []
    for match in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        think_spans.append((match.start(1), match.end(1)))

    if not think_spans:
        return []

    think_indices = []
    for idx, token in enumerate(tokens):
        if "<" in token or ">" in token:
            continue
        token_start = offsets[idx]
        token_end = token_start + len(token)
        if token_end <= token_start:
            continue
        token_center = token_start + (token_end - token_start) / 2.0
        for span_start, span_end in think_spans:
            if span_start <= token_center < span_end:
                think_indices.append(idx)
                break
    return think_indices


def compute_turn_rolling_mean(entropies, window_size: int):
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    rolling = []
    for idx in range(len(entropies)):
        if idx + 1 < window_size:
            rolling.append(None)
            continue
        rolling.append(float(np.mean(entropies[idx - window_size + 1:idx + 1])))
    return rolling


def compute_ema_series(entropies, alpha: float):
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not entropies:
        return []
    ema = [float(entropies[0])]
    for value in entropies[1:]:
        ema.append(float(alpha * value + (1 - alpha) * ema[-1]))
    return ema


def build_turn_trigger_series(turn, rolling_short_window: int, rolling_long_window: int, ema_alpha: float):
    think_indices = extract_think_token_indices(turn)
    think_tokens = [turn["generated_tokens"][idx] for idx in think_indices]
    raw_entropies = [float(turn["token_entropies"][idx]) for idx in think_indices]
    return {
        "think_token_indices": think_indices,
        "think_tokens": think_tokens,
        "raw_entropies": raw_entropies,
        "rolling_short": compute_turn_rolling_mean(raw_entropies, rolling_short_window) if raw_entropies else [],
        "rolling_long": compute_turn_rolling_mean(raw_entropies, rolling_long_window) if raw_entropies else [],
        "ema": compute_ema_series(raw_entropies, ema_alpha) if raw_entropies else [],
    }


def build_trigger_tail_rows(records, short_window: int, long_window: int, ema_alpha: float, tail_tokens: int):
    rows = []
    for record in records:
        for turn in record["turns"]:
            trigger_series = build_turn_trigger_series(turn, short_window, long_window, ema_alpha)
            think_tokens = trigger_series["think_tokens"]
            if not think_tokens:
                continue

            start_index = max(0, len(think_tokens) - tail_tokens)
            for local_idx in range(start_index, len(think_tokens)):
                rows.append({
                    "data_source": record["data_source"],
                    "id": record["id"],
                    "dataset_index": record.get("dataset_index"),
                    "is_correct": record["is_correct"],
                    "turn_index": turn["turn_index"],
                    "think_token_index": local_idx,
                    "token": think_tokens[local_idx].replace("\n", "\\n"),
                    "raw_entropy": trigger_series["raw_entropies"][local_idx],
                    "scheme_a_rolling_short": trigger_series["rolling_short"][local_idx],
                    "scheme_b_rolling_long": trigger_series["rolling_long"][local_idx],
                    "scheme_c_ema": trigger_series["ema"][local_idx],
                })
    return rows


def compute_source_statistics(records, max_turns: int):
    summary_rows = []
    per_turn_entropy_rows = []
    per_turn_length_rows = []

    grouped = defaultdict(list)
    for record in records:
        grouped[record["data_source"]].append(record)

    for source in sorted(grouped):
        source_records = grouped[source]
        all_entropies = []
        for record in source_records:
            for turn in record["turns"]:
                all_entropies.extend(turn["token_entropies"])

        source_mean_entropy = float(np.mean(all_entropies)) if all_entropies else math.nan
        summary_rows.append({
            "data_source": source,
            "num_trajectories": len(source_records),
            "num_tokens": len(all_entropies),
            "mean_entropy_all_turns": source_mean_entropy,
        })

        for turn_index in range(max_turns):
            turn_mean_entropies = []
            turn_lengths = []
            num_trajectories_with_turn = 0
            for record in source_records:
                if turn_index >= len(record["turns"]):
                    continue
                turn = record["turns"][turn_index]
                if turn["token_entropies"]:
                    turn_mean_entropies.append(float(np.mean(turn["token_entropies"])))
                turn_lengths.append(len(turn["token_entropies"]))
                num_trajectories_with_turn += 1

            per_turn_entropy_rows.append({
                "data_source": source,
                "turn_index": turn_index,
                "num_trajectories_with_turn": num_trajectories_with_turn,
                "mean_entropy": float(np.mean(turn_mean_entropies)) if turn_mean_entropies else math.nan,
            })
            per_turn_length_rows.append({
                "data_source": source,
                "turn_index": turn_index,
                "num_trajectories_with_turn": num_trajectories_with_turn,
                "mean_length": float(np.mean(turn_lengths)) if turn_lengths else math.nan,
            })

    return summary_rows, per_turn_entropy_rows, per_turn_length_rows


def build_text_summary(summary_rows, per_turn_entropy_rows, per_turn_length_rows):
    lines = []
    lines.append("# Batch Entropy Analysis Summary")
    lines.append("")
    lines.append("## Dataset Mean Entropy (all turns merged)")
    lines.append("")
    for row in summary_rows:
        lines.append(
            f"- {row['data_source']}: mean_entropy_all_turns={row['mean_entropy_all_turns']:.6f}, "
            f"num_trajectories={row['num_trajectories']}, num_tokens={row['num_tokens']}"
        )
    lines.append("")
    lines.append("## Dataset Mean Entropy by Turn")
    lines.append("")
    for row in per_turn_entropy_rows:
        mean_entropy = "nan" if math.isnan(row["mean_entropy"]) else f"{row['mean_entropy']:.6f}"
        lines.append(
            f"- {row['data_source']} turn={row['turn_index']}: "
            f"mean_entropy={mean_entropy}, trajectories={row['num_trajectories_with_turn']}"
        )
    lines.append("")
    lines.append("## Dataset Mean Length by Turn")
    lines.append("")
    for row in per_turn_length_rows:
        mean_length = "nan" if math.isnan(row["mean_length"]) else f"{row['mean_length']:.6f}"
        lines.append(
            f"- {row['data_source']} turn={row['turn_index']}: "
            f"mean_length={mean_length}, trajectories={row['num_trajectories_with_turn']}"
        )
    lines.append("")
    return "\n".join(lines)


def plot_source_trajectories(source: str, records, output_path: Path, wrap_width: int):
    num_records = len(records)
    cols = 2 if num_records > 1 else 1
    rows = math.ceil(num_records / cols)
    fig_width = 8 * cols
    fig_height = max(5.5 * rows, 7)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for ax_idx, record in enumerate(records):
        ax = axes[ax_idx]
        x_offset = 0
        boundary_positions = []

        for turn in record["turns"]:
            entropies = turn["token_entropies"]
            if not entropies:
                continue
            xs = list(range(x_offset, x_offset + len(entropies)))
            ax.plot(xs, entropies, linewidth=1.5, marker="o", markersize=2)
            boundary_positions.append((x_offset, turn["turn_index"]))
            x_offset += len(entropies)

        for boundary_x, turn_index in boundary_positions[1:]:
            ax.axvline(boundary_x - 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(boundary_x, ax.get_ylim()[1] if ax.lines else 0.0, f"turn {turn_index}",
                    fontsize=8, color="gray", va="top", ha="left")

        ax.set_title(
            f"id={record['id']} | rounds={record['num_rounds']} | correct={record['is_correct']}",
            fontsize=10,
        )
        ax.set_xlabel("token index")
        ax.set_ylabel("entropy")
        ax.grid(True, alpha=0.3)

        turn_text_lines = []
        for turn in record["turns"]:
            token_text = join_turn_tokens(turn)
            wrapped = textwrap.fill(token_text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
            turn_text_lines.append(f"turn {turn['turn_index']}: {wrapped}")
        footnote = "\n".join(turn_text_lines)
        ax.text(
            0.0,
            -0.33,
            footnote,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
        )

    for ax in axes[num_records:]:
        ax.axis("off")

    fig.suptitle(f"{source}: token entropy by trajectory", fontsize=14)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=4.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_source_trajectories_with_smoothing(
    source: str,
    records,
    output_path: Path,
    wrap_width: int,
    rolling_window_size: int,
    ema_alpha: float,
):
    num_records = len(records)
    cols = 2 if num_records > 1 else 1
    rows = math.ceil(num_records / cols)
    fig_width = 8 * cols
    fig_height = max(6.2 * rows, 8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for ax_idx, record in enumerate(records):
        ax = axes[ax_idx]
        x_offset = 0
        boundary_positions = []
        legend_added = False

        for turn in record["turns"]:
            entropies = turn["token_entropies"]
            if not entropies:
                continue

            xs = np.arange(x_offset, x_offset + len(entropies))
            rolling = compute_turn_rolling_mean(entropies, rolling_window_size)
            ema = compute_ema_series(entropies, ema_alpha)

            raw_label = "raw entropy" if not legend_added else None
            rolling_label = f"rolling-{rolling_window_size}" if not legend_added else None
            ema_label = f"ema alpha={ema_alpha}" if not legend_added else None

            ax.plot(xs, entropies, linewidth=1.0, color="tab:blue", alpha=0.45, label=raw_label)

            rolling_xs = [x for x, value in zip(xs, rolling) if value is not None]
            rolling_values = [value for value in rolling if value is not None]
            if rolling_xs:
                ax.plot(rolling_xs, rolling_values, linewidth=1.8, color="tab:orange", label=rolling_label)

            ax.plot(xs, ema, linewidth=1.6, color="tab:green", label=ema_label)

            boundary_positions.append((x_offset, turn["turn_index"]))
            x_offset += len(entropies)
            legend_added = True

        ylim = ax.get_ylim()
        for boundary_x, turn_index in boundary_positions[1:]:
            ax.axvline(boundary_x - 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(boundary_x, ylim[1], f"turn {turn_index}", fontsize=8, color="gray", va="top", ha="left")

        ax.set_title(
            f"id={record['id']} | rounds={record['num_rounds']} | correct={record['is_correct']}",
            fontsize=10,
        )
        ax.set_xlabel("token index")
        ax.set_ylabel("entropy")
        ax.grid(True, alpha=0.3)
        if legend_added:
            ax.legend(loc="upper right", fontsize=8)

        turn_text_lines = []
        for turn in record["turns"]:
            token_text = join_turn_tokens(turn)
            wrapped = textwrap.fill(token_text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
            turn_text_lines.append(f"turn {turn['turn_index']}: {wrapped}")
        footnote = "\n".join(turn_text_lines)
        ax.text(
            0.0,
            -0.4,
            footnote,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
        )

    for ax in axes[num_records:]:
        ax.axis("off")

    fig.suptitle(
        f"{source}: token entropy with rolling-{rolling_window_size} and EMA overlays",
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=4.8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_source_trigger_schemes(
    source: str,
    records,
    output_path: Path,
    wrap_width: int,
    short_window: int,
    long_window: int,
    ema_alpha: float,
):
    num_records = len(records)
    cols = 2 if num_records > 1 else 1
    rows = math.ceil(num_records / cols)
    fig_width = 8.5 * cols
    fig_height = max(6.5 * rows, 8.5)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for ax_idx, record in enumerate(records):
        ax = axes[ax_idx]
        x_offset = 0
        boundary_positions = []
        legend_added = False
        turn_text_lines = []

        for turn in record["turns"]:
            trigger_series = build_turn_trigger_series(turn, short_window, long_window, ema_alpha)
            think_tokens = trigger_series["think_tokens"]
            think_entropies = trigger_series["raw_entropies"]
            token_text = join_turn_tokens(turn)
            wrapped = textwrap.fill(token_text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
            turn_text_lines.append(f"turn {turn['turn_index']}: {wrapped}")

            if not think_tokens:
                continue

            xs = np.arange(x_offset, x_offset + len(think_tokens))
            raw_label = "A: raw entropy" if not legend_added else None
            short_label = f"B: rolling-{short_window}" if not legend_added else None
            long_label = f"C: rolling-{long_window}" if not legend_added else None
            ema_label = f"EMA alpha={ema_alpha}" if not legend_added else None

            ax.plot(xs, think_entropies, linewidth=1.0, color="tab:blue", alpha=0.45, label=raw_label)

            short_xs = [x for x, value in zip(xs, trigger_series["rolling_short"]) if value is not None]
            short_values = [value for value in trigger_series["rolling_short"] if value is not None]
            if short_xs:
                ax.plot(short_xs, short_values, linewidth=1.6, color="tab:orange", label=short_label)

            long_xs = [x for x, value in zip(xs, trigger_series["rolling_long"]) if value is not None]
            long_values = [value for value in trigger_series["rolling_long"] if value is not None]
            if long_xs:
                ax.plot(long_xs, long_values, linewidth=1.9, color="tab:red", label=long_label)

            ax.plot(xs, trigger_series["ema"], linewidth=1.6, color="tab:green", label=ema_label)

            boundary_positions.append((x_offset, turn["turn_index"]))
            x_offset += len(think_tokens)
            legend_added = True

        ylim = ax.get_ylim()
        for boundary_x, turn_index in boundary_positions[1:]:
            ax.axvline(boundary_x - 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(boundary_x, ylim[1], f"turn {turn_index}", fontsize=8, color="gray", va="top", ha="left")

        ax.set_title(
            f"id={record['id']} | rounds={record['num_rounds']} | correct={record['is_correct']}",
            fontsize=10,
        )
        ax.set_xlabel("think token index")
        ax.set_ylabel("entropy / trigger metric")
        ax.grid(True, alpha=0.3)
        if legend_added:
            ax.legend(loc="upper right", fontsize=8)

        footnote = "\n".join(turn_text_lines)
        ax.text(
            0.0,
            -0.42,
            footnote,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
        )

    for ax in axes[num_records:]:
        ax.axis("off")

    fig.suptitle(
        f"{source}: think-only trigger curves (raw / rolling-{short_window} / rolling-{long_window} / EMA)",
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=5.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    summary_rows, per_turn_entropy_rows, per_turn_length_rows = compute_source_statistics(records, args.max_turns)

    pd.DataFrame(summary_rows).to_csv(output_dir / "entropy_dataset_summary.csv", index=False)
    pd.DataFrame(per_turn_entropy_rows).to_csv(output_dir / "entropy_dataset_turn_summary.csv", index=False)
    pd.DataFrame(per_turn_length_rows).to_csv(output_dir / "length_dataset_turn_summary.csv", index=False)

    summary_payload = {
        "dataset_summary": summary_rows,
        "dataset_turn_entropy_summary": per_turn_entropy_rows,
        "dataset_turn_length_summary": per_turn_length_rows,
    }
    (output_dir / "entropy_analysis_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "entropy_analysis_summary.md").write_text(
        build_text_summary(summary_rows, per_turn_entropy_rows, per_turn_length_rows),
        encoding="utf-8",
    )

    grouped = defaultdict(list)
    for record in records:
        grouped[record["data_source"]].append(record)

    trigger_tail_rows = build_trigger_tail_rows(
        records=records,
        short_window=args.trigger_short_window,
        long_window=args.trigger_long_window,
        ema_alpha=args.ema_alpha,
        tail_tokens=args.trigger_tail_tokens,
    )
    pd.DataFrame(trigger_tail_rows).to_csv(
        output_dir / "entropy_trigger_tail_tokens.csv",
        index=False,
    )

    for source in sorted(grouped):
        output_path = output_dir / f"entropy_plot_{sanitize_source_name(source)}.png"
        plot_source_trajectories(
            source=source,
            records=grouped[source],
            output_path=output_path,
            wrap_width=args.token_wrap_width,
        )
        smooth_output_path = output_dir / (
            f"entropy_plot_{sanitize_source_name(source)}"
            f"_raw_rolling{args.rolling_window_size}_ema.png"
        )
        plot_source_trajectories_with_smoothing(
            source=source,
            records=grouped[source],
            output_path=smooth_output_path,
            wrap_width=args.token_wrap_width,
            rolling_window_size=args.rolling_window_size,
            ema_alpha=args.ema_alpha,
        )
        trigger_output_path = output_dir / (
            f"entropy_plot_{sanitize_source_name(source)}"
            f"_think_trigger_schemes.png"
        )
        plot_source_trigger_schemes(
            source=source,
            records=grouped[source],
            output_path=trigger_output_path,
            wrap_width=args.token_wrap_width,
            short_window=args.trigger_short_window,
            long_window=args.trigger_long_window,
            ema_alpha=args.ema_alpha,
        )

    print(output_dir)


if __name__ == "__main__":
    main()
