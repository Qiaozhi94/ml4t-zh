"""Convert one book chapter from the PDF into a Markdown study note.

Layout mapping (3rd edition PDF):
    Jost-Medium   84/28pt  -> chapter opener (dropped; H1 comes from the PDF outline)
    Jost-Bold     16pt     -> ## section (1.1, 1.2, ...)
    Jost-Bold     14pt     -> ### subsection
    Jost-Bold     13pt     -> #### minor heading
    SourceSerif4-Regular/Semibold/It 9.5pt -> body (bold/italic inline)
    SourceSans3-It 9pt     -> figure captions
    Consolas               -> inline code
    SourceSerif4-* 8.5pt   -> running header / page number (dropped)
    list markers ("•", "1.") are hanging-indent rows that share y with their
    first content line and are rebuilt into Markdown lists

Usage:
    python scripts/pdf_chapter_to_md.py 1            # convert chapter 1
    python scripts/pdf_chapter_to_md.py 1 --stdout   # preview only
"""

import argparse
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
MARKER_RE = re.compile(r"^\s*(?:([•▪◦])|(\d{1,2})[.)])\s+")
# de-hyphenation over-merges compounds whose own hyphen landed on the line break
DEHYPHENATION_FIXES = {"timeaware": "time-aware", "changepoint": "change point"}


def find_pdf() -> Path:
    pdfs = [p for p in ROOT.glob("*.pdf") if p.stat().st_size > 1_000_000]
    if not pdfs:
        raise SystemExit("No book PDF found in repository root.")
    return pdfs[0]


def chapter_pages(doc: fitz.Document, ch: int):
    """Return (start_page_1based, end_page_1based, outline title)."""
    toc = doc.get_toc()
    matches = [(p, t) for lvl, t, p in toc if lvl == 1 and re.match(rf"Chapter {ch:02d}\b", t)]
    if not matches:
        raise SystemExit(f"Chapter {ch:02d} not found in PDF outline.")
    start, title = matches[0]
    later = [p for lvl, _t, p in toc if lvl == 1 and p > start]
    end = min(later) - 1 if later else doc.page_count
    return start, end, re.sub(r"^Chapter 0+(\d+):\s*", r"Chapter \1: ", title)


def span_md(s: dict) -> str:
    t = s["text"].replace("\u00a0", " ").replace("\u00ad", "")
    f, stripped = s["font"], s["text"].strip()
    if not stripped:
        return " " if t else ""
    if f == "SourceSerif4-Semibold":
        return f"**{stripped}**" + (" " if t.endswith(" ") else "")
    if f == "SourceSerif4-It":
        return f"*{stripped}*" + (" " if t.endswith(" ") else "")
    if f == "Consolas":
        return f"`{stripped}`" + (" " if t.endswith(" ") else "")
    return t


def line_md(line: dict) -> str:
    text = "".join(span_md(s) for s in line["spans"])
    return re.sub(r"\s+", " ", text).strip()


def heading_level(text: str, line: dict):
    s = line["spans"][0]
    if not s["font"].startswith("Jost") or s["size"] >= 20:
        return None
    if s["size"] >= 15:
        return 2
    if s["size"] >= 13.5:
        return 3
    return 4


def is_furniture(line: dict) -> bool:
    """Running header + page number: serif, <=8.75pt."""
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return True
    return all(s["font"].startswith("SourceSerif") and s["size"] <= 8.75 for s in spans)


def is_opener(line: dict) -> bool:
    s = line["spans"][0]
    return s["font"].startswith("Jost") and s["size"] >= 20


def is_caption(line: dict) -> bool:
    return line["spans"][0]["font"].startswith("SourceSans3")


def is_promo_block(block: dict) -> bool:
    for l in block.get("lines", []):
        for s in l["spans"]:
            if s["text"].strip():
                return s["text"].strip().startswith("Your purchase includes")
    return False


def join_lines(a: str, b: str) -> str:
    if not a:
        return b
    if a.endswith("-"):
        if b[:1].islower():
            return a[:-1] + b  # soft hyphenation across a line break
        return a + b  # true compound split (e.g. CRISP-/ML(Q))
    if a.endswith("**") and b.startswith("**"):
        head, rest = a[:-2], b[2:]  # bold continuing across the break
        if head.endswith("-") and rest[:1].islower():
            return head[:-1] + rest
        return head + " " + rest
    if a.endswith("*") and not a.endswith("**") and b.startswith("*") and not b.startswith("**"):
        head, rest = a[:-1], b[1:]  # italic continuing across the break
        if head.endswith("-") and rest[:1].islower():
            return head[:-1] + rest
        return head + " " + rest
    if a.endswith(("–", "—")):
        return a + b
    return a + " " + b


def page_rows(page: fitz.Page):
    """Flatten a page into visual rows: same-baseline fragments (hanging list
    markers and their content) merged, ordered top-to-bottom, left-to-right.
    Also returns the page's body-left and body-right text margins (the book
    mirrors margins on odd/even pages, so thresholds must be relative)."""
    items = []  # (y0, x0, y1, x1, payload, text) payload: line dict or image block
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 1:
            items.append((block["bbox"][1], block["bbox"][0], block["bbox"][3], block["bbox"][2], block, None))
            continue
        if is_promo_block(block):
            continue
        for line in block["lines"]:
            text = line_md(line)
            if not text or is_furniture(line) or is_opener(line):
                continue
            items.append((line["bbox"][1], line["bbox"][0], line["bbox"][3], line["bbox"][2], line, text))
    items.sort(key=lambda t: (t[0], t[1]))
    rows = []
    for it in items:
        if rows and it[4] is not None and rows[-1][-1][4] is not None \
                and abs(it[0] - rows[-1][-1][0]) < 3.0:
            rows[-1].append(it)
            rows[-1].sort(key=lambda t: t[1])
        else:
            rows.append([it])
    text_rows = [it for it in items if it[4] is not None]
    body_left = min(it[1] for it in text_rows)
    body_right = max(it[3] for it in text_rows)
    return rows, body_left, body_right


def extract_image(page: fitz.Page, doc: fitz.Document, bbox, out_dir: Path, name: str):
    best, best_overlap = None, 0.0
    for info in page.get_image_info(xrefs=True):
        b = fitz.Rect(info["bbox"])
        overlap = (b & fitz.Rect(bbox)).get_area() if b.intersects(fitz.Rect(bbox)) else 0
        if overlap > best_overlap:
            best, best_overlap = info, overlap
    if best is None or best["xref"] == 0:
        return None
    img = doc.extract_image(best["xref"])
    path = out_dir / f"{name}.{img['ext']}"
    path.write_bytes(img["image"])
    return path


def convert(ch: int, to_stdout: bool = False):
    pdf = find_pdf()
    doc = fitz.open(pdf)
    start, end, title = chapter_pages(doc, ch)

    folder = next(ROOT.glob(f"{ch:02d}_*"), ROOT / "docs")
    assets = folder / "assets"
    if not to_stdout:
        assets.mkdir(parents=True, exist_ok=True)

    out: list[str] = [f"# {title}", ""]
    fig_counter = 0
    para = ""            # paragraph text accumulated with join_lines
    items = []           # list items: (kind, text) with kind in {"bullet", "num"}
    caption = ""         # figure caption text accumulated with join_lines

    def flush_para():
        nonlocal para
        text = para.replace("—", " - ").strip()
        if text:
            out.extend(["", text])
        para = ""

    def flush_items():
        nonlocal items
        if items:
            out.append("")
            n = 0
            for kind, text in items:
                text = text.replace("—", " - ")
                if kind == "num":
                    n += 1
                    out.append(f"{n}. {text}")
                else:
                    out.append(f"- {text}")
            items = []

    def flush_caption():
        nonlocal caption
        text = caption.replace("—", " - ").strip()
        if text:
            out.extend(["", f"*{text}*"])
        caption = ""

    for pno in range(start - 1, end):
        page = doc[pno]
        prev_top, prev_x1 = None, None
        rows, body_left, body_right = page_rows(page)
        for row in rows:
            if row[0][5] is None:  # image block
                flush_para()
                flush_items()
                flush_caption()
                fig_counter += 1
                stem = f"figure_{ch}_{fig_counter}"
                path = extract_image(page, doc, row[0][4]["bbox"], assets, stem)
                if path:
                    alt = f"Figure {ch}.{fig_counter}"
                    out.extend(["", f"![{alt}]({path.relative_to(folder).as_posix()})"])
                prev_top, prev_x1 = row[0][0], row[0][3]
                continue

            line, text = row[0][4], " ".join(it[5] for it in row)
            top, x0 = row[0][0], row[0][1]
            # baseline-to-baseline distance: ~13.3 line spacing, ~19.5 paragraph gap
            gap = None if prev_top is None else top - prev_top
            # page break: gap unknown; list continuations sit ~31pt right of the
            # body margin; a justified line ending short closes its paragraph
            first_of_page = gap is None

            if is_caption(line):
                flush_para()
                flush_items()
                caption = join_lines(caption, text)
                prev_top, prev_x1 = top, row[0][3]
                continue
            flush_caption()

            lvl = heading_level(text, line)
            if lvl is not None:
                flush_para()
                flush_items()
                if out and out[-1] == "" and len(out) >= 2 and out[-2].startswith("#" * lvl + " ") \
                        and gap is not None and gap < 21:
                    out[-2] = join_lines(out[-2], text)  # heading wrapped across two lines
                else:
                    out.extend(["", f"{'#' * lvl} {text}", ""])
                prev_top, prev_x1 = top, row[0][3]
                continue

            marker = MARKER_RE.match(text)
            if marker:
                flush_para()
                items.append(("bullet" if marker.group(1) else "num", text[marker.end():].strip()))
                prev_top, prev_x1 = top, row[0][3]
                continue

            # paragraph break, or a page break landing on body margin, closes a list
            if (gap is not None and gap >= 17.5) or (first_of_page and items and x0 < body_left + 20):
                flush_para()
                flush_items()
            elif items:  # tight line inside a list -> continuation of the last item
                items[-1] = (items[-1][0], join_lines(items[-1][1], text))
                prev_top, prev_x1 = top, row[0][3]
                continue
            elif first_of_page and para and prev_x1 is not None and prev_x1 < body_right - 40:
                flush_para()

            para = join_lines(para, text)
            prev_top, prev_x1 = top, row[0][3]

    flush_para()
    flush_items()
    flush_caption()

    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    for wrong, right in DEHYPHENATION_FIXES.items():
        md = md.replace(wrong, right)

    if to_stdout:
        print(md)
    else:
        dest = folder / f"chapter_{ch:02d}.md"
        dest.write_text(md, encoding="utf-8")
        print(f"Chapter {ch:02d}: pages {start}-{end} -> {dest.relative_to(ROOT)}")
        print(f"Sections: {md.count(chr(10) + '## ')}, subsections: {md.count(chr(10) + '### ')}, figures: {fig_counter}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chapter", type=int)
    ap.add_argument("--stdout", action="store_true", help="print instead of writing files")
    args = ap.parse_args()
    convert(args.chapter, to_stdout=args.stdout)
