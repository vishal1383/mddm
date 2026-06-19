from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import re
import textwrap


def plot_single_anchor_results(rows: list[dict], out_dir: str | Path) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for example_id, group in _by_example(rows).items():
        group = sorted(group, key=lambda row: row["anchor_position"])
        best = max(group, key=lambda row: row["information_gain"])
        x = list(range(len(best["p_gt_before"])))

        _plot_before_after(
            plt,
            x,
            best["p_gt_before"],
            best["p_gt_after"],
            best["anchor_position"],
            "p_gt",
            f"{example_id}: p_gt before/after best-IG anchor",
            out / f"{_safe(example_id)}_pgt_position.png",
        )
        _plot_before_after(
            plt,
            x,
            _safe_log(best["p_gt_before"]),
            _safe_log(best["p_gt_after"]),
            best["anchor_position"],
            "log p_gt",
            f"{example_id}: log p_gt before/after best-IG anchor",
            out / f"{_safe(example_id)}_log_pgt_position.png",
        )

        plt.figure(figsize=(10, 4))
        plt.plot(
            [row["anchor_position"] for row in group],
            [row["information_gain"] for row in group],
            marker="o",
            linewidth=1,
        )
        plt.axvline(best["anchor_position"], color="black", linestyle="--", linewidth=1)
        plt.xlabel("anchor position")
        plt.ylabel("summed entropy IG")
        plt.title(f"{example_id}: single-anchor IG")
        plt.tight_layout()
        plt.savefig(out / f"{_safe(example_id)}_ig_by_anchor.png", dpi=160)
        plt.close()

def plot_greedy_results(rows: list[dict], out_dir: str | Path) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for example_id, group in _by_example(rows).items():
        group = sorted(group, key=lambda row: row["k"])
        plt.figure(figsize=(7, 4))
        plt.plot([row["k"] for row in group], [row["score_after"] for row in group], marker="o")
        plt.xlabel("k anchors")
        plt.ylabel("score: p_gt over remaining masks")
        plt.title(f"{example_id}: greedy p_gt score")
        plt.tight_layout()
        plt.savefig(out / f"{_safe(example_id)}_greedy_score.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot([row["k"] for row in group], [row["information_gain"] for row in group], marker="o")
        plt.xlabel("k anchors")
        plt.ylabel("summed entropy IG for selected anchor")
        plt.title(f"{example_id}: greedy selected-anchor IG")
        plt.tight_layout()
        plt.savefig(out / f"{_safe(example_id)}_greedy_ig.png", dpi=160)
        plt.close()

        for row in group:
            x = list(range(len(row["p_gt_before"])))
            label = f"{_safe(example_id)}_greedy_k{int(row['k']):02d}"
            _plot_before_after(
                plt,
                x,
                row["p_gt_before"],
                row["p_gt_after"],
                row["selected_anchor_position"],
                "p_gt",
                f"{example_id}: p_gt before/after greedy k={row['k']}",
                out / f"{label}_pgt_position.png",
            )
            _plot_before_after(
                plt,
                x,
                _safe_log(row["p_gt_before"]),
                _safe_log(row["p_gt_after"]),
                row["selected_anchor_position"],
                "log p_gt",
                f"{example_id}: log p_gt before/after greedy k={row['k']}",
                out / f"{label}_log_pgt_position.png",
            )


def plot_layout_results(rows: list[dict], out_dir: str | Path) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for example_id, group in _by_example(rows).items():
        by_layout: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_layout[row["layout"]].append(row)

        plt.figure(figsize=(8, 4))
        for layout, layout_rows in sorted(by_layout.items()):
            layout_rows = sorted(layout_rows, key=lambda row: row["k"])
            plt.plot(
                [row["k"] for row in layout_rows],
                [row["score_after"] for row in layout_rows],
                marker="o",
                label=layout,
            )
        plt.xlabel("k anchors")
        plt.ylabel("score: p_gt over remaining masks")
        plt.title(f"{example_id}: layout control")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"{_safe(example_id)}_layout_scores.png", dpi=160)
        plt.close()

        by_k: dict[int, list[dict]] = defaultdict(list)
        for row in group:
            by_k[int(row["k"])].append(row)
        for k, k_rows in sorted(by_k.items()):
            if _default_pgt(k_rows[0]) is None:
                continue
            x = list(range(len(_default_pgt(k_rows[0]))))
            _plot_layout_position(
                plt,
                x,
                k_rows,
                "p_gt",
                lambda values: values,
                f"{example_id}: layout p_gt k={k}",
                out / f"{_safe(example_id)}_layout_k{k:02d}_pgt_position.png",
            )
            _plot_layout_position(
                plt,
                x,
                k_rows,
                "log p_gt",
                _safe_log,
                f"{example_id}: layout log p_gt k={k}",
                out / f"{_safe(example_id)}_layout_k{k:02d}_log_pgt_position.png",
            )


def plot_layout_pgt_results(rows: list[dict], out_dir: str | Path) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for example_id, group in _by_example(rows).items():
        by_k: dict[int, list[dict]] = defaultdict(list)
        for row in group:
            by_k[int(row["k"])].append(row)
        for k, k_rows in sorted(by_k.items()):
            if _default_pgt(k_rows[0]) is None:
                continue
            x = list(range(len(_default_pgt(k_rows[0]))))
            _plot_layout_position(
                plt,
                x,
                k_rows,
                "p_gt",
                lambda values: values,
                f"{example_id}: layout p_gt k={k}",
                out / f"{_safe(example_id)}_layout_k{k:02d}_pgt_position.png",
            )
            _plot_layout_position(
                plt,
                x,
                k_rows,
                "log p_gt",
                _safe_log,
                f"{example_id}: layout log p_gt k={k}",
                out / f"{_safe(example_id)}_layout_k{k:02d}_log_pgt_position.png",
            )


def _by_example(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row["example_id"])].append(row)
    return out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "example"


def _safe_log(values: list[float]) -> list[float]:
    return [math.log(max(float(v), 1e-12)) for v in values]


def _plot_before_after(plt, x, before, after, anchor_position, ylabel, title, path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(x, before, label="before")
    plt.plot(x, after, label=f"after anchor {anchor_position}")
    plt.axvline(anchor_position, color="black", linestyle="--", linewidth=1)
    plt.xlabel("completion token position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_layout_position(plt, x, rows, ylabel, transform, title, path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["layout"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, transform(_default_pgt(rows[0])), label="default", color="black", linewidth=1.5)
    for row in rows:
        ax.plot(x, transform(row["p_gt_after"]), label=row["layout"], linewidth=1)
    greedy = next((row for row in rows if row["layout"] == "greedy_ig"), None)
    if greedy is not None:
        for anchor in greedy.get("anchors", []):
            ax.axvline(int(anchor), color="black", linestyle=":", linewidth=1, alpha=0.65)

    caption = _anchor_caption(rows)
    if caption:
        fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7, family="monospace")
    ax.set_xlabel("completion token position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout(rect=(0, 0.18 if caption else 0, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _default_pgt(row: dict) -> list[float] | None:
    return row.get("p_gt_default") or row.get("p_gt_before")


def _anchor_caption(rows: list[dict]) -> str:
    greedy = next((row for row in rows if row["layout"] == "greedy_ig"), None)
    if greedy is None:
        return ""
    return "\n".join(textwrap.wrap(f"greedy_ig anchors={greedy.get('anchors', [])}", width=140))


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to write plots, or run with --no-plots.") from exc
    return plt
