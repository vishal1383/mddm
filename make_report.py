#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
from html import escape
import re
import struct
from pathlib import Path
import zipfile


RUNS = [
    ("llada-8b_gsm8k", "LLaDA-8B-Instruct", "GSM8K"),
    ("llada-8b_humaneval", "LLaDA-8B-Instruct", "HumanEval"),
    ("dream-7b_gsm8k", "Dream-v0-Instruct-7B", "GSM8K"),
    ("dream-7b_humaneval", "Dream-v0-Instruct-7B", "HumanEval"),
]
LAYOUTS = ["greedy_ig", "maximally_separated", "prefix", "suffix", "middle_cluster"]
PLOT_KS = [1, 2, 4, 8]


def main() -> None:
    args = parse_args()
    if args.markdown:
        build_markdown_report(Path(args.markdown), Path(args.out))
        print(args.out)
        return

    root = Path(args.root)
    report = DocxReport()

    report.heading("MDM Spread-Anchor Probe", 1)
    report.paragraph("Cached experiment report for the masked diffusion anchor probe.")

    report.heading("Setup", 2)
    report.paragraph(
        "For each prompt/completion pair, completion tokens are masked except for an anchor set. "
        "An anchor reveals the true token at its own token position. The experiment measures how "
        "revealing anchors changes the model distribution at the remaining masked token positions."
    )
    report.paragraph(
        "Greedy anchors are selected by entropy reduction after revealing the true token at candidate "
        "anchor position p, not by p_gt. For each target position q, H(q) is the entropy of the full "
        "vocabulary distribution at q. We report this gold-reveal metric as AggregatedIG_gt: "
        "sum_q [H_before(q) - H_after(q)], summed over remaining masked target positions q."
    )
    report.paragraph(
        "Because anchors are fixed to their ground-truth token values, AggregatedIG_gt is a gold-anchor "
        "entropy-change/influence measure rather than Shannon information gain, so it can be negative. "
        "True information gain would require averaging over all possible token values at the anchor "
        "position under P(X_p | z_A); implementing that expectation changes the probe cost from roughly "
        "K*T to K*T*V, adding a vocabulary-size factor, so we leave it as a later variant if needed."
    )
    report.paragraph(
        "The p_gt plots are diagnostic: they show teacher-forced gold-token confidence before anchors "
        "and after each layout at matched k. Black dotted vertical lines mark the greedy_ig anchor positions."
    )

    report.heading("Models And Datasets", 2)
    setup_rows = []
    for run, model, dataset in RUNS:
        examples = read_csv(root / run / "examples.csv")
        n = len(examples)
        mean_prompt = mean_float(row["prompt_tokens"] for row in examples)
        mean_completion = mean_float(row["completion_tokens"] for row in examples)
        setup_rows.append([model, dataset, str(n), f"{mean_prompt:.1f}", f"{mean_completion:.1f}"])
    report.table(
        ["Model", "Dataset", "Prompts", "Mean prompt tokens", "Mean completion tokens"],
        setup_rows,
    )

    report.heading("Layout-Control Scores", 2)
    report.paragraph(
        "Each model/dataset table reports the entropy-reduction score used for anchor selection, "
        "AggregatedIG_gt = sum_q H_before(q) - H_after(q). This gold-anchor influence metric can be "
        "negative because it conditions on fixed ground-truth anchor values rather than averaging over "
        "all possible anchor-token values."
    )
    for run, model, dataset in RUNS:
        report.heading(f"{model} / {dataset}", 3)
        aggregate_path = root / run / "layout_control_aggregate.csv"
        report.paragraph("AggregatedIG_gt: sum(H_before - H_after)")
        report.table(
            ["k", "greedy_ig", "max separated", "prefix", "suffix", "middle cluster"],
            layout_table_rows(aggregate_path, "mean_information_gain"),
        )

    report.heading("Layout p_gt Examples", 2)
    report.paragraph(
        "One cached prompt is shown for each model/dataset pair at k = 1, 2, 4, and 8."
    )
    for k in PLOT_KS:
        report.heading(f"k = {k}", 3)
        for run, model, dataset in RUNS:
            example_id = first_example_id(root / run / "examples.csv")
            image = plot_path(root / run, example_id, k)
            report.paragraph(f"{model} / {dataset}, example {example_id}")
            if image.exists():
                report.image(image)
            else:
                report.paragraph(f"Missing cached plot: {image}")

    report.heading("Conclusion And Next Steps", 2)
    report.paragraph(
        "The layout-control tables show whether greedy AggregatedIG_gt anchors produce larger summed "
        "gold entropy reduction than "
        "simple prefix, suffix, middle-cluster, or maximally separated anchors. This supports the core "
        "anchor-position hypothesis: where the revealed true tokens are placed matters."
    )
    report.paragraph(
        "The GSM8K effect is especially strong, which may partly reflect that GSM8K solutions are more "
        "structured and easier for these models once a few informative tokens are revealed. Code generation "
        "is more mixed: Dream shows a much smaller HumanEval effect, while LLaDA still benefits but the "
        "layout controls narrow the gap at larger k. The weaker or mixed code signal is surprising and "
        "worth checking on more prompts and other code tasks."
    )
    report.paragraph(
        "Next steps: test additional domains such as MATH500, bio, legal, creative writing, and text-to-SQL; "
        "add alternative confidence scores such as max_w p(w | z_A, q), p_gt, and log p_gt/NLL; and only after "
        "the anchor mechanism is established, evaluate downstream decoding metrics such as GSM8K exact match "
        "and HumanEval pass@1."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save(out)
    print(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the MDM probe Word report.")
    parser.add_argument("--root", default="outputs/full_probe")
    parser.add_argument("--out", default="reports/mdm_spread_anchor_probe_report_v2.docx")
    parser.add_argument("--markdown", help="Convert a Markdown analysis report to Word.")
    return parser.parse_args()


def build_markdown_report(source: Path, out: Path) -> None:
    report = DocxReport()
    lines = source.read_text(encoding="utf-8").splitlines()
    index = 0
    in_code = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            index += 1
            continue
        if in_code:
            report.code_paragraph(line)
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            report.page_break()
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            report.heading(clean_markdown(heading.group(2)), min(len(heading.group(1)), 3))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and is_markdown_separator(lines[index + 1]):
            table_lines = [line]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [parse_markdown_row(item) for item in table_lines]
            report.table(rows[0], rows[1:])
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            report.label(clean_markdown(stripped))
        elif stripped.startswith(">"):
            report.paragraph(clean_markdown(stripped[1:].strip()), italic=True)
        elif stripped:
            report.paragraph(clean_markdown(stripped))
        index += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    report.save(out)


def is_markdown_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_markdown_row(line: str) -> list[str]:
    return [clean_markdown(cell.strip()) for cell in line.strip().strip("|").split("|")]


def clean_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


def layout_table_rows(path: Path, metric: str) -> list[list[str]]:
    rows = read_csv(path)
    by_k: dict[str, dict[str, float]] = {}
    for row in rows:
        by_k.setdefault(row["k"], {})[row["layout"]] = float(row.get(metric, "nan"))
    out = []
    for k in sorted(by_k, key=lambda value: int(value)):
        vals = by_k[k]
        out.append([k] + [f"{vals.get(layout, float('nan')):.2f}" for layout in LAYOUTS])
    return out


def first_example_id(path: Path) -> str:
    rows = read_csv(path)
    return rows[0]["example_id"]


def plot_path(run_dir: Path, example_id: str, k: int) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(example_id)).strip("_") or "example"
    return run_dir / "plots_cached_layout_pgt" / f"{safe}_layout_k{k:02d}_pgt_position.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_float(values) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


class DocxReport:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.images: list[tuple[Path, str]] = []

    def heading(self, text: str, level: int) -> None:
        size = {1: "32", 2: "26", 3: "22"}.get(level, "22")
        style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(level, "Heading3")
        self.parts.append(
            f"<w:p><w:pPr><w:pStyle w:val=\"{style}\"/><w:keepNext/><w:spacing w:before=\"240\" w:after=\"120\"/></w:pPr>"
            f"<w:r><w:rPr><w:b/><w:sz w:val=\"{size}\"/></w:rPr>{wtext(text)}</w:r></w:p>"
        )

    def paragraph(self, text: str, *, italic: bool = False) -> None:
        run_props = "<w:rPr><w:i/></w:rPr>" if italic else ""
        self.parts.append(f"<w:p><w:r>{run_props}{wtext(text)}</w:r></w:p>")

    def label(self, text: str) -> None:
        self.parts.append(
            f"<w:p><w:pPr><w:spacing w:before=\"120\" w:after=\"40\"/></w:pPr>"
            f"<w:r><w:rPr><w:b/></w:rPr>{wtext(text)}</w:r></w:p>"
        )

    def code_paragraph(self, text: str) -> None:
        self.parts.append(
            "<w:p><w:pPr><w:spacing w:before=\"0\" w:after=\"0\"/>"
            "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F3F4F6\"/>"
            "<w:ind w:left=\"120\" w:right=\"120\"/></w:pPr>"
            "<w:r><w:rPr><w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\"/>"
            f"<w:sz w:val=\"16\"/></w:rPr>{wtext(text)}</w:r></w:p>"
        )

    def page_break(self) -> None:
        self.parts.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        xml = [
            "<w:tbl><w:tblPr><w:tblBorders>"
            "<w:top w:val=\"single\" w:sz=\"4\"/><w:left w:val=\"single\" w:sz=\"4\"/>"
            "<w:bottom w:val=\"single\" w:sz=\"4\"/><w:right w:val=\"single\" w:sz=\"4\"/>"
            "<w:insideH w:val=\"single\" w:sz=\"4\"/><w:insideV w:val=\"single\" w:sz=\"4\"/>"
            "</w:tblBorders></w:tblPr>",
            table_row(headers, bold=True),
        ]
        xml.extend(table_row(row) for row in rows)
        xml.append("</w:tbl>")
        self.parts.append("".join(xml))

    def image(self, path: Path) -> None:
        rid = f"rId{len(self.images) + 1}"
        self.images.append((path, rid))
        width_px, height_px = png_size(path)
        max_width_emu = int(6.6 * 914400)
        width_emu = int(width_px / 96 * 914400)
        height_emu = int(height_px / 96 * 914400)
        if width_emu > max_width_emu:
            scale = max_width_emu / width_emu
            width_emu = max_width_emu
            height_emu = int(height_emu * scale)
        doc_pr_id = len(self.images)
        self.parts.append(
            f"<w:p><w:r><w:drawing><wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\">"
            f"<wp:extent cx=\"{width_emu}\" cy=\"{height_emu}\"/>"
            f"<wp:docPr id=\"{doc_pr_id}\" name=\"Picture {doc_pr_id}\"/>"
            f"<a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">"
            f"<pic:pic><pic:nvPicPr><pic:cNvPr id=\"{doc_pr_id}\" name=\"{escape(path.name)}\"/>"
            f"<pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed=\"{rid}\"/>"
            f"<a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
            f"<pic:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"{width_emu}\" cy=\"{height_emu}\"/>"
            f"</a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></pic:spPr>"
            f"</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
        )

    def save(self, path: Path) -> None:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as docx:
            docx.writestr("[Content_Types].xml", content_types(self.images))
            docx.writestr("_rels/.rels", package_rels())
            docx.writestr("word/document.xml", document_xml(self.parts))
            docx.writestr("word/styles.xml", styles_xml())
            docx.writestr("word/_rels/document.xml.rels", document_rels(self.images))
            for idx, (image, _) in enumerate(self.images, start=1):
                docx.write(image, f"word/media/image{idx}.png")


def table_row(values: list[str], bold: bool = False) -> str:
    cells = []
    for value in values:
        rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
        cells.append(f"<w:tc><w:p><w:r>{rpr}{wtext(str(value))}</w:r></w:p></w:tc>")
    return "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(cells) + "</w:tr>"


def wtext(text: str) -> str:
    return f"<w:t xml:space=\"preserve\">{escape(text)}</w:t>"


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        data = handle.read(24)
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return 1200, 700
    return struct.unpack(">II", data[16:24])


def content_types(images: list[tuple[Path, str]]) -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Default Extension=\"png\" ContentType=\"image/png\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "<Override PartName=\"/word/styles.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>"
        "</Types>"
    )


def package_rels() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/></Relationships>"
    )


def document_rels(images: list[tuple[Path, str]]) -> str:
    rels = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
    ]
    for idx, (_, rid) in enumerate(images, start=1):
        rels.append(
            f"<Relationship Id=\"{rid}\" "
            "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" "
            f"Target=\"media/image{idx}.png\"/>"
        )
    rels.append(
        "<Relationship Id=\"rIdStyles\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" "
        "Target=\"styles.xml\"/>"
    )
    rels.append("</Relationships>")
    return "".join(rels)


def styles_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\">"
        "<w:name w:val=\"Normal\"/><w:qFormat/></w:style>"
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading1\">"
        "<w:name w:val=\"heading 1\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>"
        "<w:uiPriority w:val=\"9\"/><w:qFormat/><w:pPr><w:outlineLvl w:val=\"0\"/></w:pPr>"
        "<w:rPr><w:b/><w:sz w:val=\"32\"/></w:rPr></w:style>"
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading2\">"
        "<w:name w:val=\"heading 2\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>"
        "<w:uiPriority w:val=\"9\"/><w:qFormat/><w:pPr><w:outlineLvl w:val=\"1\"/></w:pPr>"
        "<w:rPr><w:b/><w:sz w:val=\"26\"/></w:rPr></w:style>"
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading3\">"
        "<w:name w:val=\"heading 3\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>"
        "<w:uiPriority w:val=\"9\"/><w:qFormat/><w:pPr><w:outlineLvl w:val=\"2\"/></w:pPr>"
        "<w:rPr><w:b/><w:sz w:val=\"22\"/></w:rPr></w:style>"
        "</w:styles>"
    )


def document_xml(parts: list[str]) -> str:
    body = "".join(parts)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
        "xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">"
        f"<w:body>{body}<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"720\" w:right=\"720\" w:bottom=\"720\" w:left=\"720\"/>"
        "</w:sectPr></w:body></w:document>"
    )


if __name__ == "__main__":
    main()
