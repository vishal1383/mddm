from __future__ import annotations


def prefix_anchors(positions: list[int], k: int) -> list[int]:
    return positions[:k]


def suffix_anchors(positions: list[int], k: int) -> list[int]:
    return positions[-k:]


def middle_cluster_anchors(positions: list[int], k: int) -> list[int]:
    start = (len(positions) - k) // 2
    return positions[start : start + k]


def maximally_separated_anchors(positions: list[int], k: int) -> list[int]:
    if k == 1:
        return [positions[len(positions) // 2]]
    anchors = []
    for i in range(k):
        pos = positions[round(i * (len(positions) - 1) / (k - 1))]
        if pos not in anchors:
            anchors.append(pos)
    return anchors + [pos for pos in positions if pos not in anchors][: k - len(anchors)]


def layout_anchors(
    positions: list[int],
    k: int,
    greedy_chain: list[int] | None = None,
) -> dict[str, list[int]]:
    k = min(k, len(positions))
    layouts = {
        "prefix": prefix_anchors(positions, k),
        "suffix": suffix_anchors(positions, k),
        "middle_cluster": middle_cluster_anchors(positions, k),
        "maximally_separated": maximally_separated_anchors(positions, k),
    }
    if greedy_chain is not None:
        layouts["greedy_ig"] = sorted(greedy_chain[:k])
    return layouts
