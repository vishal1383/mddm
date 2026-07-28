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
        selection_metric = group[0].get("selection_metric", "information_gain")
        best = max(group, key=lambda row: row[selection_metric])
        best_label = selection_metric.replace("_", " ")
        metric = _position_metric(best)
        x = list(range(len(best[metric["before"]])))
        _plot_before_after(
            plt,
            x,
            metric["transform"](best[metric["before"]]),
            metric["transform"](best[metric["after"]]),
            best["anchor_position"],
            metric["ylabel"],
            f"{example_id}: {metric['ylabel']} before/after best {best_label} anchor",
            out / f"{_safe(example_id)}_{metric['suffix']}_position.png",
        )

        plt.figure(figsize=(10, 4))
        plt.plot(
            [row["anchor_position"] for row in group],
            [row[metric["gain"]] for row in group],
            marker="o",
            linewidth=1,
        )
        plt.axvline(best["anchor_position"], color="black", linestyle="--", linewidth=1)
        plt.xlabel("anchor position")
        plt.ylabel(metric["gain"])
        plt.title(f"{example_id}: single-anchor {metric['gain']}")
        plt.tight_layout()
        plt.savefig(out / f"{_safe(example_id)}_{metric['gain']}_by_anchor.png", dpi=160)
        plt.close()

def plot_greedy_results(rows: list[dict], out_dir: str | Path) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for example_id, group in _by_example(rows).items():
        group = sorted(group, key=lambda row: row["k"])
        metric = _position_metric(group[0])
        _plot_k_curve(
            plt,
            group,
            metric["score"],
            f"score: {metric['ylabel']} over remaining masks",
            f"{example_id}: greedy {metric['ylabel']} score",
            out / f"{_safe(example_id)}_greedy_{metric['suffix']}_score.png",
        )
        _plot_k_curve(
            plt,
            group,
            metric["gain"],
            f"{metric['gain']} for selected anchor",
            f"{example_id}: greedy selected-anchor {metric['gain']}",
            out / f"{_safe(example_id)}_greedy_{metric['gain']}.png",
        )

        for row in group:
            x = list(range(len(row[metric["before"]])))
            label = f"{_safe(example_id)}_greedy_k{int(row['k']):02d}"
            _plot_before_after(
                plt,
                x,
                metric["transform"](row[metric["before"]]),
                metric["transform"](row[metric["after"]]),
                row.get("anchors", [row["selected_anchor_position"]]),
                metric["ylabel"],
                f"{example_id}: {metric['ylabel']} before/after greedy k={row['k']}",
                out / f"{label}_{metric['suffix']}_position.png",
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

        metric = _position_metric(group[0])
        _plot_layout_scores(
            plt,
            by_layout,
            metric["score"],
            f"score: {metric['ylabel']} over remaining masks",
            f"{example_id}: layout control {metric['ylabel']}",
            out / f"{_safe(example_id)}_layout_{metric['suffix']}_scores.png",
        )

        by_k: dict[int, list[dict]] = defaultdict(list)
        for row in group:
            by_k[int(row["k"])].append(row)
        for k, k_rows in sorted(by_k.items()):
            x = list(range(len(_default_values(k_rows[0], metric["default"]))))
            _plot_layout_position(
                plt,
                x,
                k_rows,
                metric["ylabel"],
                metric["transform"],
                f"{example_id}: layout {metric['ylabel']} k={k}",
                out / f"{_safe(example_id)}_layout_k{k:02d}_{metric['suffix']}_position.png",
                value_key=metric["after"],
                default_key=metric["default"],
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


def plot_decode_impact_results(rows: list[dict], out_dir: str | Path) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for example_id, group in _by_example(rows).items():
        by_policy: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_policy[str(row["policy"])].append(row)

        fig, (ax_answer, ax_changed, ax_tokens) = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
        for policy, policy_rows in sorted(by_policy.items()):
            policy_rows = sorted(policy_rows, key=lambda row: int(row["k"]))
            ks = [int(row["k"]) for row in policy_rows]
            answer_values = [
                float(row["answer_correct"]) if row.get("answer_correct") is not None else math.nan
                for row in policy_rows
            ]
            ax_answer.plot(ks, answer_values, marker="o", label=policy)
            ax_changed.plot(
                ks,
                [float(row["answer_changed_from_baseline"]) for row in policy_rows],
                marker="o",
                label=policy,
            )
            ax_tokens.plot(
                ks,
                [float(row["token_accuracy"]) for row in policy_rows],
                marker="o",
                label=policy,
            )

        caption = _decode_caption(group)
        if caption:
            fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7, family="monospace")
        ax_answer.set_title(f"{example_id}: cheated-anchor decode impact")
        ax_answer.set_ylabel("final answer correct")
        ax_answer.set_yticks([0, 1])
        ax_answer.set_ylim(-0.1, 1.1)
        ax_answer.legend()
        ax_changed.set_ylabel("answer changed")
        ax_changed.set_yticks([0, 1])
        ax_changed.set_ylim(-0.1, 1.1)
        ax_tokens.set_xlabel("gold anchors revealed (k)")
        ax_tokens.set_ylabel("token accuracy")
        ax_tokens.set_ylim(-0.05, 1.05)
        fig.tight_layout(rect=(0, 0.18 if caption else 0, 1, 1))
        fig.savefig(out / f"{_safe(example_id)}_decode_answer_impact.png", dpi=160)
        plt.close(fig)

    aggregate = _decode_aggregate(rows)
    if aggregate:
        by_policy: dict[str, list[dict]] = defaultdict(list)
        for row in aggregate:
            by_policy[str(row["policy"])].append(row)
        fig, (ax_answer, ax_changed, ax_tokens) = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
        for policy, policy_rows in sorted(by_policy.items()):
            policy_rows = sorted(policy_rows, key=lambda row: int(row["k"]))
            ks = [int(row["k"]) for row in policy_rows]
            ax_answer.plot(
                ks,
                [float(row["mean_answer_correct"]) for row in policy_rows],
                marker="o",
                label=policy,
            )
            ax_changed.plot(
                ks,
                [float(row["mean_answer_changed_from_baseline"]) for row in policy_rows],
                marker="o",
                label=policy,
            )
            ax_tokens.plot(
                ks,
                [float(row["mean_token_accuracy"]) for row in policy_rows],
                marker="o",
                label=policy,
            )
        ax_answer.set_ylabel("mean answer correct")
        ax_answer.set_ylim(-0.05, 1.05)
        ax_answer.set_title("Cheated-anchor decode impact across examples")
        ax_answer.legend()
        ax_changed.set_ylabel("mean answer changed")
        ax_changed.set_ylim(-0.05, 1.05)
        ax_tokens.set_xlabel("gold anchors revealed (k)")
        ax_tokens.set_ylabel("mean token accuracy")
        ax_tokens.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(out / "aggregate_decode_answer_impact.png", dpi=160)
        plt.close(fig)

        plt.figure(figsize=(8, 4))
        for policy, policy_rows in sorted(by_policy.items()):
            policy_rows = sorted(policy_rows, key=lambda row: int(row["k"]))
            plt.plot(
                [int(row["k"]) for row in policy_rows],
                [float(row["mean_answer_changed_from_baseline"]) for row in policy_rows],
                marker="o",
                label=policy,
            )
        plt.xlabel("gold anchors revealed (k)")
        plt.ylabel("mean answer changed from baseline")
        plt.ylim(-0.05, 1.05)
        plt.title("Mean final-answer change under cheated anchors")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "aggregate_mean_answer_change.png", dpi=160)
        plt.close()


def plot_greedy_standard_accuracy_change(
    aggregate_rows: list[dict],
    out_dir: str | Path,
    *,
    policy: str = "greedy_ig",
    example_rows: list[dict] | None = None,
    show_individual_examples: bool = False,
) -> None:
    focus_rows = [
        row for row in aggregate_rows
        if str(row.get("policy")) == policy
        and (
            row.get("mean_answer_accuracy_change_pp") is not None
            or row.get("mean_answer_correct") is not None
        )
    ]
    if not focus_rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    focus_rows = sorted(focus_rows, key=lambda row: int(row["k"]))

    xs: list[int] = []
    deltas_pp: list[float] = []
    ci95_pp: list[float] = []
    labels: list[str] = []
    baseline_acc = None
    if "mean_answer_accuracy_change_pp" not in focus_rows[0]:
        baseline = next((row for row in focus_rows if int(row["k"]) == 0), focus_rows[0])
        baseline_acc = _float_value(baseline.get("mean_answer_correct"))
        if baseline_acc is None:
            return
    for row in focus_rows:
        delta_pp = _float_value(row.get("mean_answer_accuracy_change_pp"))
        if delta_pp is None:
            mean_acc = _float_value(row.get("mean_answer_correct"))
            if mean_acc is None or baseline_acc is None:
                continue
            delta_pp = 100.0 * (mean_acc - baseline_acc)
        xs.append(int(row["k"]))
        deltas_pp.append(delta_pp)
        ci95_pp.append(_float_value(row.get("ci95_answer_accuracy_change_pp")) or 0.0)
        if int(row["k"]) == 0:
            labels.append("standard")
        else:
            labels.append(str(row["k"]))
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if example_rows and show_individual_examples:
        by_example: dict[str, list[dict]] = defaultdict(list)
        for row in example_rows:
            if str(row.get("policy")) == policy:
                by_example[str(row["example_id"])].append(row)
        for _, group in sorted(by_example.items()):
            group = sorted(group, key=lambda row: int(row["k"]))
            ex_x = [int(row["k"]) for row in group]
            ex_y = [
                _float_value(row.get("answer_accuracy_change_pp"))
                for row in group
            ]
            if any(value is None for value in ex_y):
                continue
            ax.plot(
                ex_x,
                ex_y,
                color="#bdbdbd",
                linewidth=0.9,
                alpha=0.35,
                zorder=1,
            )

    ax.axhline(0, color="black", linewidth=0.9)
    ax.errorbar(
        xs,
        deltas_pp,
        yerr=ci95_pp if any(value > 0 for value in ci95_pp) else None,
        marker="o",
        linewidth=2,
        color="#1976d2",
        capsize=3,
        label="mean greedy anchors",
        zorder=3,
    )
    ax.scatter(xs[0], deltas_pp[0], color="#424242", s=48, zorder=3, label="standard decode")
    if example_rows and show_individual_examples:
        ax.plot([], [], color="#bdbdbd", linewidth=1, alpha=0.6, label="individual examples")

    low = min([y - err for y, err in zip(deltas_pp, ci95_pp)] + [0.0])
    high = max([y + err for y, err in zip(deltas_pp, ci95_pp)] + [0.0])
    if low == high:
        ax.set_ylim(low - 5.0, high + 5.0)
    else:
        pad = max(3.0, 0.15 * (high - low))
        ax.set_ylim(max(-100.0, low - pad), min(100.0, high + pad))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("greedy anchors placed (k)")
    ax.set_ylabel("mean change in answer accuracy (pp)")
    ax.set_title(f"{policy}: accuracy change vs standard decode")
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(left=0.16)
    fig.savefig(out / "greedy_standard_mean_answer_accuracy_change.png", dpi=180)
    plt.close(fig)


def plot_policy_standard_accuracy_change(
    aggregate_rows: list[dict],
    out_dir: str | Path,
    *,
    policies: tuple[str, ...] = ("greedy_ig", "prefix"),
) -> None:
    focus_rows = [
        row for row in aggregate_rows
        if str(row.get("policy")) in set(policies)
        and row.get("mean_answer_accuracy_change_pp") is not None
    ]
    if not focus_rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in focus_rows:
        by_policy[str(row["policy"])].append(row)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.axhline(0, color="black", linewidth=0.9)
    colors = {
        "greedy_ig": "#1976d2",
        "prefix": "#ef6c00",
    }
    labels_by_policy = {
        "greedy_ig": "greedy IG anchors",
        "prefix": "left-to-right prefix anchors",
    }
    all_xs: set[int] = set()
    lows = [0.0]
    highs = [0.0]
    for policy in policies:
        rows = sorted(by_policy.get(policy, []), key=lambda row: int(row["k"]))
        if not rows:
            continue
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["mean_answer_accuracy_change_pp"]) for row in rows]
        errs = [_float_value(row.get("ci95_answer_accuracy_change_pp")) or 0.0 for row in rows]
        all_xs.update(xs)
        lows.extend(y - err for y, err in zip(ys, errs))
        highs.extend(y + err for y, err in zip(ys, errs))
        ax.errorbar(
            xs,
            ys,
            yerr=errs if any(err > 0 for err in errs) else None,
            marker="o",
            linewidth=2,
            capsize=3,
            color=colors.get(policy),
            label=labels_by_policy.get(policy, policy),
        )

    if not all_xs:
        plt.close(fig)
        return
    xs_sorted = sorted(all_xs)
    ax.set_xticks(xs_sorted)
    ax.set_xticklabels(["standard" if x == 0 else str(x) for x in xs_sorted])
    low = min(lows)
    high = max(highs)
    if low == high:
        ax.set_ylim(low - 5.0, high + 5.0)
    else:
        pad = max(3.0, 0.15 * (high - low))
        ax.set_ylim(max(-100.0, low - pad), min(100.0, high + pad))
    ax.set_xlabel("anchors placed (k)")
    ax.set_ylabel("mean change in answer accuracy vs standard (pp)")
    ax.set_title("Greedy IG anchors vs left-to-right control")
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(left=0.16)
    fig.savefig(out / "greedy_vs_left_to_right_accuracy_change.png", dpi=180)
    plt.close(fig)


def plot_decode_trajectory_results(
    rows: list[dict],
    out_dir: str | Path,
    *,
    max_groups: int = 50,
) -> None:
    if not rows or max_groups <= 0:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["example_id"]), str(row["policy"]), int(row["k"]))].append(row)

    for group_index, ((example_id, policy, k), group) in enumerate(sorted(grouped.items())):
        if group_index >= max_groups:
            break
        group = sorted(group, key=lambda row: (int(row["step"]), int(row["position"])))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        scatter = ax.scatter(
            [int(row["position"]) for row in group],
            [int(row["step"]) for row in group],
            c=[float(row["confidence"]) for row in group],
            cmap="viridis",
            s=24,
        )
        anchors = group[0].get("anchors", [])
        for anchor in anchors:
            ax.axvline(int(anchor), color="black", linestyle=":", linewidth=1, alpha=0.65)
        fig.colorbar(scatter, ax=ax, label="fill confidence")
        ax.set_xlabel("completion token position")
        ax.set_ylabel("decode step")
        ax.set_title(f"{example_id}: {policy} k={k} decode trajectory")
        fig.tight_layout()
        fig.savefig(out / f"{_safe(example_id)}_{_safe(policy)}_k{k:02d}_decode_trajectory.png", dpi=160)
        plt.close(fig)


def plot_anchor_answer_timeline(
    rows: list[dict],
    token_rows: list[dict],
    out_dir: str | Path,
) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokens_by_example: dict[str, list[dict]] = defaultdict(list)
    for row in token_rows:
        tokens_by_example[str(row["example_id"])].append(row)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["example_id"]), str(row["policy"]))].append(row)

    for (example_id, policy), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["k"]))
        tokens = sorted(tokens_by_example.get(example_id, []), key=lambda row: int(row["position"]))
        if tokens:
            T = max(int(row["position"]) for row in tokens) + 1
        else:
            T = max((max(row.get("anchors", []) or [-1]) for row in group), default=-1) + 1
        if T <= 0:
            continue

        width = max(10, min(24, 0.28 * T))
        height = max(4.5, 1.0 + 0.55 * len(group))
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.75, max(int(row["k"]) for row in group) + 0.85)

        for row in group:
            k = int(row["k"])
            correct = _bool_value(row.get("answer_correct"))
            changed = _bool_value(row.get("answer_changed_from_baseline"))
            color = "#2e7d32" if correct else "#c62828"
            ax.axhline(k, color="#e0e0e0", linewidth=0.8, zorder=0)
            for anchor in row.get("anchors", []) or []:
                ax.scatter(
                    int(anchor),
                    k,
                    s=42,
                    marker="s",
                    color="#90caf9",
                    edgecolors="#1565c0",
                    linewidths=0.6,
                    zorder=2,
                )
            new_positions = row.get("new_anchor_positions", []) or []
            for anchor, text in zip(new_positions, row.get("new_anchor_token_texts", []) or []):
                ax.scatter(
                    int(anchor),
                    k,
                    s=115,
                    marker="*",
                    color=color,
                    edgecolors="black" if changed else color,
                    linewidths=0.8,
                    zorder=3,
                )
                ax.annotate(
                    _short_token(text),
                    (int(anchor), k),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    rotation=35,
                )
            answer = str(row.get("decoded_answer", ""))
            status = "correct" if correct else "wrong"
            changed_label = "changed" if changed else "same"
            ax.text(
                T - 0.5,
                k,
                f"  k={k}: ans={answer!r} ({status}, {changed_label})",
                va="center",
                ha="left",
                fontsize=8,
                clip_on=False,
            )

        tick_positions = list(range(T))
        if T > 80:
            stride = max(1, T // 40)
            tick_positions = list(range(0, T, stride))
        token_labels = {int(row["position"]): _short_token(row.get("token_text", "")) for row in tokens}
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([token_labels.get(pos, str(pos)) for pos in tick_positions], rotation=90, fontsize=7)
        ax.set_yticks([int(row["k"]) for row in group])
        ax.set_ylabel("anchors placed (k)")
        ax.set_xlabel("gold completion token position")
        ax.set_title(f"{example_id}: {policy} anchor placement -> decoded answer")
        fig.subplots_adjust(right=0.72, bottom=0.3)
        fig.savefig(out / f"{_safe(example_id)}_{_safe(policy)}_anchor_answer_timeline.png", dpi=170)
        plt.close(fig)


def plot_anchor_canvas_results(
    rows: list[dict],
    token_rows: list[dict],
    out_dir: str | Path,
) -> None:
    if not rows:
        return
    plt = _require_matplotlib()
    from matplotlib.patches import Rectangle

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokens_by_example: dict[str, list[dict]] = defaultdict(list)
    for row in token_rows:
        tokens_by_example[str(row["example_id"])].append(row)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["example_id"]), str(row["policy"]))].append(row)

    for (example_id, policy), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["k"]))
        tokens = sorted(tokens_by_example.get(example_id, []), key=lambda row: int(row["position"]))
        if not tokens:
            continue
        T = len(tokens)

        cell_font_size = 6.5 if T <= 90 else 5.2
        fig_width = max(12, min(36, 0.32 * T))
        fig_height = max(3.5, 1.05 + 0.62 * len(group))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, T + 13)
        ax.set_ylim(-0.7, len(group) + 0.4)
        ax.axis("off")

        for col, token in enumerate(tokens):
            ax.text(
                col + 0.5,
                len(group) - 0.1,
                str(token["position"]),
                ha="center",
                va="bottom",
                fontsize=6,
                color="#666666",
            )

        for row_idx, row in enumerate(group):
            y = len(group) - row_idx - 1
            anchors = {int(anchor) for anchor in row.get("anchors", []) or []}
            new_anchors = {int(anchor) for anchor in row.get("new_anchor_positions", []) or []}
            token_text_by_pos = {int(token["position"]): str(token["token_text"]) for token in tokens}
            for pos in range(T):
                is_anchor = pos in anchors
                is_new = pos in new_anchors
                face = "#fff8e1" if is_new else ("#e3f2fd" if is_anchor else "#f8f8f8")
                edge = "#f9a825" if is_new else ("#64b5f6" if is_anchor else "#dddddd")
                ax.add_patch(Rectangle((pos, y), 0.92, 0.45, facecolor=face, edgecolor=edge, linewidth=0.8))
                label = _short_token(token_text_by_pos[pos], limit=7) if is_anchor else "_"
                ax.text(pos + 0.46, y + 0.225, label, ha="center", va="center", fontsize=cell_font_size)

            correct = _bool_value(row.get("answer_correct"))
            changed = _bool_value(row.get("answer_changed_from_baseline"))
            status = "correct" if correct else "wrong"
            changed_label = "changed" if changed else "same"
            ans = str(row.get("decoded_answer", ""))
            score = row.get("gold_answer_token_confidence")
            delta = _float_value(row.get("answer_correct_delta_from_baseline"))
            score_text = f", p_gold_ans={float(score):.3f}" if isinstance(score, (int, float)) else ""
            delta_text = f", Δbase={delta:+.0f}" if delta is not None else ""
            ax.text(
                T + 0.4,
                y + 0.225,
                f"k={int(row['k'])}: ans={ans!r} ({status}, {changed_label}{delta_text}{score_text})",
                ha="left",
                va="center",
                fontsize=8,
                color="#2e7d32" if correct else "#c62828",
            )

        ax.set_title(f"{example_id}: {policy} blank-canvas anchor decode", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"{_safe(example_id)}_{_safe(policy)}_anchor_canvas.png", dpi=180)
        plt.close(fig)


def plot_anchor_token_effects(
    aggregate_rows: list[dict],
    out_dir: str | Path,
    *,
    focus_policy: str = "greedy_ig",
) -> None:
    if not aggregate_rows:
        return
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        by_policy[str(row["policy"])].append(row)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for policy, rows in sorted(by_policy.items()):
        rows = sorted(rows, key=lambda row: int(row["position_bin"]))
        ax.scatter(
            [int(row["position_bin"]) + 1 for row in rows],
            [float(row["mean_correct_delta_from_baseline"]) for row in rows],
            s=52,
            label=policy,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"t{i}" for i in range(1, 11)])
    ax.set_xlabel("anchor token slot in completion (normalized decile)")
    ax.set_ylabel("mean Δ correct_ans vs baseline")
    ax.set_title("Correct-answer change by anchor token position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "aggregate_correct_change_by_anchor_position.png", dpi=170)
    plt.close(fig)

    focus_rows = by_policy.get(focus_policy, [])
    if focus_rows:
        focus_rows = sorted(focus_rows, key=lambda row: int(row["position_bin"]))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            [int(row["position_bin"]) + 1 for row in focus_rows],
            [float(row["mean_correct_delta_from_baseline"]) for row in focus_rows],
            width=0.75,
            color="#1976d2",
            alpha=0.8,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(1, 11))
        ax.set_xticklabels([f"t{i}" for i in range(1, 11)])
        ax.set_xlabel("greedy anchor token slot (normalized decile)")
        ax.set_ylabel("mean Δ correct_ans vs baseline")
        ax.set_title(f"{focus_policy}: correct-answer change by anchor position")
        fig.tight_layout()
        fig.savefig(out / f"{_safe(focus_policy)}_correct_change_by_anchor_position.png", dpi=170)
        plt.close(fig)


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _float_value(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _by_example(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row["example_id"])].append(row)
    return out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "example"


def _safe_log(values: list[float]) -> list[float]:
    return [math.log(max(float(v), 1e-12)) for v in values]


def _short_token(value, limit: int = 10) -> str:
    text = str(value)
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if text == " ":
        text = "<space>"
    if len(text) > limit:
        text = text[: limit - 1] + "..."
    return text


def _plot_before_after(plt, x, before, after, anchor_positions, ylabel, title, path: Path) -> None:
    anchors = _as_anchor_list(anchor_positions)
    plt.figure(figsize=(10, 4))
    plt.plot(x, before, label="before")
    plt.plot(x, after, label=f"after anchors {anchors}")
    for anchor in anchors:
        plt.axvline(anchor, color="black", linestyle="--", linewidth=1)
    plt.xlabel("completion token position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _as_anchor_list(anchor_positions) -> list[int]:
    if isinstance(anchor_positions, (list, tuple)):
        return [int(anchor) for anchor in anchor_positions]
    return [int(anchor_positions)]


def _plot_k_curve(plt, rows, metric_key: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot([row["k"] for row in rows], [row[metric_key] for row in rows], marker="o")
    plt.xlabel("k anchors")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_layout_scores(plt, by_layout, metric_key: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for layout, layout_rows in sorted(by_layout.items()):
        layout_rows = sorted(layout_rows, key=lambda row: row["k"])
        plt.plot(
            [row["k"] for row in layout_rows],
            [row[metric_key] for row in layout_rows],
            marker="o",
            label=layout,
        )
    plt.xlabel("k anchors")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_layout_position(
    plt,
    x,
    rows,
    ylabel,
    transform,
    title,
    path: Path,
    *,
    value_key: str = "p_gt_after",
    default_key: str = "p_gt_default",
) -> None:
    rows = sorted(rows, key=lambda row: row["layout"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, transform(_default_values(rows[0], default_key)), label="default", color="black", linewidth=1.5)
    for row in rows:
        ax.plot(x, transform(row[value_key]), label=row["layout"], linewidth=1)
    greedy = _greedy_layout_row(rows)
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


def _default_max_p(row: dict) -> list[float] | None:
    return row.get("max_p_default") or row.get("max_p_before")


def _default_values(row: dict, default_key: str) -> list[float] | None:
    if default_key == "max_p_default":
        return _default_max_p(row)
    return _default_pgt(row)


def _position_metric(row: dict) -> dict:
    if "max_p_before" in row or "max_p_default" in row:
        return {
            "before": "max_p_before",
            "after": "max_p_after",
            "default": "max_p_default",
            "score": "max_p_score_after",
            "gain": "max_p_gain",
            "ylabel": "max p",
            "suffix": "maxp",
            "transform": lambda values: values,
        }
    return {
        "before": "p_gt_before",
        "after": "p_gt_after",
        "default": "p_gt_default",
        "score": "score_after",
        "gain": row.get("selection_metric", "information_gain"),
        "ylabel": "p_gt",
        "suffix": "pgt",
        "transform": lambda values: values,
    }


def _anchor_caption(rows: list[dict]) -> str:
    greedy = _greedy_layout_row(rows)
    if greedy is None:
        return ""
    return "\n".join(textwrap.wrap(f"{greedy['layout']} anchors={greedy.get('anchors', [])}", width=140))


def _greedy_layout_row(rows: list[dict]) -> dict | None:
    return next((row for row in rows if str(row.get("layout", "")).startswith("greedy_")), None)


def _decode_caption(rows: list[dict]) -> str:
    first = rows[0]
    gold = str(first.get("gold_answer", ""))
    baseline = next((row for row in rows if int(row.get("k", -1)) == 0), first)
    decoded = str(baseline.get("decoded_answer", ""))
    caption = f"gold_answer={gold!r}; baseline_answer={decoded!r}"
    return "\n".join(textwrap.wrap(caption, width=140))


def _decode_aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), int(row["k"]))].append(row)

    aggregate: list[dict] = []
    for (policy, k), group in sorted(grouped.items()):
        answer_values = [
            float(row["answer_correct"])
            for row in group
            if row.get("answer_correct") is not None
        ]
        token_values = [float(row["token_accuracy"]) for row in group]
        changed_values = [
            float(row["answer_changed_from_baseline"])
            for row in group
            if row.get("answer_changed_from_baseline") is not None
        ]
        if not answer_values or not token_values:
            continue
        aggregate.append(
            {
                "policy": policy,
                "k": k,
                "mean_answer_correct": sum(answer_values) / len(answer_values),
                "mean_answer_changed_from_baseline": (
                    sum(changed_values) / len(changed_values) if changed_values else math.nan
                ),
                "mean_token_accuracy": sum(token_values) / len(token_values),
            }
        )
    return aggregate


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to write plots, or run with --no-plots.") from exc
    return plt
