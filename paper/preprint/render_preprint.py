from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Callable, Iterable

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    PageBreak,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name("LIP_PREPRINT_001.md")
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "LIP_PREPRINT_001.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#276FBF")
TEAL = colors.HexColor("#2A9D8F")
ORANGE = colors.HexColor("#E07A2D")
RED = colors.HexColor("#C74B50")
INK = colors.HexColor("#1D2733")
MID = colors.HexColor("#5E6A75")
LIGHT = colors.HexColor("#E9EEF3")
PALE_BLUE = colors.HexColor("#EEF5FB")
PALE_ORANGE = colors.HexColor("#FFF3E8")
WHITE = colors.white


def register_fonts() -> dict[str, str]:
    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/ariali.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        ),
    ]
    for regular, bold, italic in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont("LIP-Regular", str(regular)))
            pdfmetrics.registerFont(TTFont("LIP-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("LIP-Italic", str(italic)))
            pdfmetrics.registerFontFamily(
                "LIP", normal="LIP-Regular", bold="LIP-Bold", italic="LIP-Italic"
            )
            return {
                "regular": "LIP-Regular",
                "bold": "LIP-Bold",
                "italic": "LIP-Italic",
                "mono": "Courier",
            }
    return {
        "regular": "Helvetica",
        "bold": "Helvetica-Bold",
        "italic": "Helvetica-Oblique",
        "mono": "Courier",
    }


FONTS = register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "LIPTitle",
            parent=sample["Title"],
            fontName=FONTS["bold"],
            fontSize=23,
            leading=27,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "LIPSubtitle",
            parent=sample["Normal"],
            fontName=FONTS["regular"],
            fontSize=13,
            leading=17,
            textColor=MID,
            alignment=TA_CENTER,
            spaceAfter=17,
        ),
        "author": ParagraphStyle(
            "LIPAuthor",
            parent=sample["Normal"],
            fontName=FONTS["bold"],
            fontSize=12.5,
            leading=16,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "meta": ParagraphStyle(
            "LIPMeta",
            parent=sample["Normal"],
            fontName=FONTS["regular"],
            fontSize=8.4,
            leading=11,
            textColor=MID,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "LIPH1",
            parent=sample["Heading1"],
            fontName=FONTS["bold"],
            fontSize=14.2,
            leading=17,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "LIPH2",
            parent=sample["Heading2"],
            fontName=FONTS["bold"],
            fontSize=11.2,
            leading=14,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "LIPH3",
            parent=sample["Heading3"],
            fontName=FONTS["bold"],
            fontSize=9.7,
            leading=12,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "LIPBody",
            parent=sample["BodyText"],
            fontName=FONTS["regular"],
            fontSize=9.15,
            leading=12.25,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
            splitLongWords=True,
        ),
        "abstract": ParagraphStyle(
            "LIPAbstract",
            parent=sample["BodyText"],
            fontName=FONTS["regular"],
            fontSize=8.75,
            leading=11.7,
            textColor=INK,
            alignment=TA_JUSTIFY,
            leftIndent=10,
            rightIndent=10,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "LIPBullet",
            parent=sample["BodyText"],
            fontName=FONTS["regular"],
            fontSize=9,
            leading=12,
            textColor=INK,
            alignment=TA_LEFT,
            leftIndent=16,
            firstLineIndent=-9,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "numbered": ParagraphStyle(
            "LIPNumbered",
            parent=sample["BodyText"],
            fontName=FONTS["regular"],
            fontSize=9,
            leading=12,
            textColor=INK,
            alignment=TA_LEFT,
            leftIndent=18,
            firstLineIndent=-13,
            spaceAfter=3,
        ),
        "equation": ParagraphStyle(
            "LIPEquation",
            parent=sample["Code"],
            fontName=FONTS["mono"],
            fontSize=8.2,
            leading=11,
            textColor=NAVY,
            alignment=TA_CENTER,
            leftIndent=18,
            rightIndent=18,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "LIPCaption",
            parent=sample["Normal"],
            fontName=FONTS["regular"],
            fontSize=7.7,
            leading=9.7,
            textColor=MID,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "LIPTable",
            parent=sample["Normal"],
            fontName=FONTS["regular"],
            fontSize=7.15,
            leading=8.7,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "table_head": ParagraphStyle(
            "LIPTableHead",
            parent=sample["Normal"],
            fontName=FONTS["bold"],
            fontSize=7.1,
            leading=8.5,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),
        "reference": ParagraphStyle(
            "LIPReference",
            parent=sample["BodyText"],
            fontName=FONTS["regular"],
            fontSize=7.7,
            leading=10,
            textColor=INK,
            alignment=TA_LEFT,
            leftIndent=13,
            firstLineIndent=-13,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "LIPCallout",
            parent=sample["Normal"],
            fontName=FONTS["bold"],
            fontSize=9.4,
            leading=12.5,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
    }


STYLES = make_styles()


def parse_source(path: Path) -> tuple[dict[str, str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Manuscript must begin with metadata front matter.")
    metadata: dict[str, str] = {}
    index = 1
    while index < len(lines) and lines[index].strip() != "---":
        line = lines[index]
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        index += 1
    if index == len(lines):
        raise ValueError("Unterminated metadata front matter.")
    return metadata, lines[index + 1 :]


def inline_markup(text: str) -> str:
    safe = html.escape(text, quote=True)
    code_fragments: list[str] = []

    def save_code(match: re.Match[str]) -> str:
        code_fragments.append(
            f'<font name="{FONTS["mono"]}" color="#17324D">{match.group(1)}</font>'
        )
        return f"@@CODE{len(code_fragments) - 1}@@"

    safe = re.sub(r"`([^`]+)`", save_code, safe)
    safe = re.sub(r"\*\*([^*]+)\*\*", rf'<font name="{FONTS["bold"]}">\1</font>', safe)
    safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", rf'<font name="{FONTS["italic"]}">\1</font>', safe)

    def link_url(match: re.Match[str]) -> str:
        url = match.group(0)
        return f'<link href="{url}" color="#276FBF">{url}</link>'

    safe = re.sub(r"https://[^\s<]+", link_url, safe)
    for index, fragment in enumerate(code_fragments):
        safe = safe.replace(f"@@CODE{index}@@", fragment)
    return safe


def box(
    drawing: Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: Iterable[str],
    fill: colors.Color,
    stroke: colors.Color = BLUE,
    text_color: colors.Color = INK,
) -> None:
    drawing.add(
        Rect(
            x,
            y,
            width,
            height,
            rx=6,
            ry=6,
            fillColor=fill,
            strokeColor=stroke,
            strokeWidth=1,
        )
    )
    line_list = list(lines)
    start_y = y + height / 2 + (len(line_list) - 1) * 5
    for index, line_text in enumerate(line_list):
        drawing.add(
            String(
                x + width / 2,
                start_y - index * 10,
                line_text,
                fontName=FONTS["bold"] if index == 0 else FONTS["regular"],
                fontSize=7.2,
                fillColor=text_color,
                textAnchor="middle",
            )
        )


def arrow(
    drawing: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: colors.Color = MID,
) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.4))
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = [x2, y2, x2 - 6 * direction, y2 + 3, x2 - 6 * direction, y2 - 3]
    else:
        direction = 1 if y2 >= y1 else -1
        points = [x2, y2, x2 + 3, y2 - 6 * direction, x2 - 3, y2 - 6 * direction]
    drawing.add(Polygon(points, fillColor=color, strokeColor=color))


def pipeline_figure(width: float) -> KeepTogether:
    height = 198
    d = Drawing(width, height)
    d.add(String(0, 184, "STAGE 1 — VALIDATE THE RECEIVER CARRIER", fontName=FONTS["bold"], fontSize=8.2, fillColor=NAVY))
    margin = 4
    gap = 18
    first_w = (width - margin * 2 - gap * 2) / 3
    y1 = 121
    box(d, margin, y1, first_w, 45, ["Native packets", "matched / shuffled"], PALE_BLUE)
    box(d, margin + first_w + gap, y1, first_w, 45, ["Fixed receiver", "carrier + executor"], PALE_BLUE)
    box(d, margin + 2 * (first_w + gap), y1, first_w, 45, ["Oracle identity effect", "Is the carrier viable?"], colors.HexColor("#E7F5F1"), TEAL)
    arrow(d, margin + first_w, y1 + 22.5, margin + first_w + gap - 3, y1 + 22.5)
    arrow(d, margin + 2 * first_w + gap, y1 + 22.5, margin + 2 * (first_w + gap) - 3, y1 + 22.5)

    d.add(String(0, 101, "STAGE 2 — TEST THE LEARNED IDENTITY EFFECT", fontName=FONTS["bold"], fontSize=8.2, fillColor=NAVY))
    second_gap = 13
    second_w = (width - margin * 2 - second_gap * 3) / 4
    y2 = 42
    starts = [margin + i * (second_w + second_gap) for i in range(4)]
    box(d, starts[0], y2, second_w, 42, ["Sender states", "matched / shuffled"], PALE_ORANGE, ORANGE)
    box(d, starts[1], y2, second_w, 42, ["Learned bridge", "sealed selection"], PALE_ORANGE, ORANGE)
    box(d, starts[2], y2, second_w, 42, ["Predicted packets", "+ mean / random"], PALE_ORANGE, ORANGE)
    box(d, starts[3], y2, second_w, 42, ["Functional effect", "same carrier"], colors.HexColor("#FDEBEC"), RED)
    for index in range(3):
        arrow(d, starts[index] + second_w, y2 + 21, starts[index + 1] - 3, y2 + 21)

    d.add(String(width / 2, 13, "A positive oracle anchor does not substitute for a positive learned matched-versus-shuffled effect.", fontName=FONTS["bold"], fontSize=7.5, fillColor=RED, textAnchor="middle"))
    caption = Paragraph(
        "<b>Figure 1.</b> Receiver-anchored evaluation separates target-carrier viability from learned cross-model transport. All learned-side conditions reuse the validated carrier and functional evaluator.",
        STYLES["caption"],
    )
    return KeepTogether([d, caption])


def bar_figure(
    width: float,
    labels: list[str],
    values: list[float],
    fills: list[colors.Color],
    caption_text: str,
) -> KeepTogether:
    height = 218
    d = Drawing(width, height)
    left = 35
    right = 10
    bottom = 58
    top = 15
    plot_w = width - left - right
    plot_h = height - bottom - top
    for tick in [0, 25, 50, 75, 100]:
        y = bottom + plot_h * tick / 100
        d.add(Line(left, y, width - right, y, strokeColor=LIGHT, strokeWidth=0.8))
        d.add(String(left - 5, y - 2.5, str(tick), fontName=FONTS["regular"], fontSize=6.7, fillColor=MID, textAnchor="end"))
    d.add(String(left, height - 7, "Functional pass rate (%)", fontName=FONTS["bold"], fontSize=7.1, fillColor=MID))
    slot = plot_w / len(labels)
    bar_w = min(31, slot * 0.57)
    for index, (label, value, fill) in enumerate(zip(labels, values, fills)):
        x = left + slot * index + (slot - bar_w) / 2
        visual_h = plot_h * value / 100
        if 0 < value < 1:
            visual_h = max(visual_h, 1.2)
        d.add(Rect(x, bottom, bar_w, visual_h, fillColor=fill, strokeColor=colors.white, strokeWidth=0.4))
        value_label = "0" if value == 0 else (f"{value:.2f}" if value < 10 else f"{value:.1f}")
        d.add(String(x + bar_w / 2, bottom + max(visual_h, 0) + 4, value_label, fontName=FONTS["bold"], fontSize=6.5, fillColor=INK, textAnchor="middle"))
        d.add(String(x + bar_w / 2 - 2, bottom - 7, label, fontName=FONTS["regular"], fontSize=6.4, fillColor=INK, textAnchor="end", angle=36))
    d.add(Line(left, bottom, width - right, bottom, strokeColor=MID, strokeWidth=1))
    caption = Paragraph(caption_text, STYLES["caption"])
    return KeepTogether([d, caption])


def p013_figure(width: float) -> KeepTogether:
    return bar_figure(
        width,
        ["K32 M", "K32 S", "MMM", "SMM", "MSM", "MMS", "SMS", "SSS"],
        [88.54, 0.0, 82.29, 3.12, 0.0, 81.25, 4.17, 0.0],
        [BLUE, MID, TEAL, ORANGE, ORANGE, TEAL, ORANGE, MID],
        "<b>Figure 2.</b> Study A functional pass rates. M and S denote matched and same-stratum shuffled identity. The equal-capacity K=24 factorial collapses when core or function-name identity is replaced, while the observed MMS rate remains near MMM.",
    )


def bridge_architecture_figure(width: float) -> KeepTogether:
    height = 270
    d = Drawing(width, height)
    margin = 4

    d.add(
        String(
            0,
            257,
            "QUERY-CONDITIONED NONLINEAR BRIDGE",
            fontName=FONTS["bold"],
            fontSize=8.2,
            fillColor=NAVY,
        )
    )

    row1_gap = 14
    row1_w = (width - margin * 2 - row1_gap * 2) / 3
    row1_y = 198
    row1_x = [margin + i * (row1_w + row1_gap) for i in range(3)]
    box(
        d,
        row1_x[0],
        row1_y,
        row1_w,
        43,
        ["DeepSeek packet", "24 layers x 32 positions", "x 2048 features"],
        PALE_BLUE,
    )
    box(
        d,
        row1_x[1],
        row1_y,
        row1_w,
        43,
        ["Project + normalize", "2048 -> 512; add learned", "layer / position embeddings"],
        PALE_BLUE,
    )
    box(
        d,
        row1_x[2],
        row1_y,
        row1_w,
        43,
        ["Source memory", "768 sites x 512"],
        PALE_BLUE,
    )
    arrow(d, row1_x[0] + row1_w, row1_y + 21.5, row1_x[1] - 3, row1_y + 21.5)
    arrow(d, row1_x[1] + row1_w, row1_y + 21.5, row1_x[2] - 3, row1_y + 21.5)

    row2_gap = 18
    row2_w = (width - margin * 2 - row2_gap * 2) / 3
    row2_y = 120
    row2_x = [margin + i * (row2_w + row2_gap) for i in range(3)]
    box(
        d,
        row2_x[0],
        row2_y,
        row2_w,
        45,
        ["Protocol queries", "32 learned slots x 512"],
        PALE_ORANGE,
        ORANGE,
    )
    box(
        d,
        row2_x[1],
        row2_y,
        row2_w,
        45,
        ["Sender encoder", "2 pre-norm decoder blocks", "8-head cross-attention"],
        PALE_ORANGE,
        ORANGE,
    )
    box(
        d,
        row2_x[2],
        row2_y,
        row2_w,
        45,
        ["LIP code", "32 slots x 512"],
        PALE_ORANGE,
        ORANGE,
    )
    arrow(d, row2_x[0] + row2_w, row2_y + 22.5, row2_x[1] - 3, row2_y + 22.5)
    arrow(d, row2_x[1] + row2_w, row2_y + 22.5, row2_x[2] - 3, row2_y + 22.5)
    d.add(
        Line(
            row1_x[2] + row1_w / 2,
            row1_y,
            row1_x[2] + row1_w / 2,
            row2_y + 60,
            strokeColor=MID,
            strokeWidth=1.4,
        )
    )
    arrow(
        d,
        row1_x[2] + row1_w / 2,
        row2_y + 60,
        row2_x[1] + row2_w / 2,
        row2_y + 45,
    )

    row3_gap = 10
    row3_w = (width - margin * 2 - row3_gap * 3) / 4
    row3_y = 38
    row3_x = [margin + i * (row3_w + row3_gap) for i in range(4)]
    receiver_fill = colors.HexColor("#E7F5F1")
    box(
        d,
        row3_x[0],
        row3_y,
        row3_w,
        48,
        ["Receiver queries", "8 x 24 sites x 512"],
        receiver_fill,
        TEAL,
    )
    box(
        d,
        row3_x[1],
        row3_y,
        row3_w,
        48,
        ["Receiver decoder", "2 pre-norm blocks", "attend to LIP code"],
        receiver_fill,
        TEAL,
    )
    box(
        d,
        row3_x[2],
        row3_y,
        row3_w,
        48,
        ["Normalized residual", "8 x 24 x 4096"],
        receiver_fill,
        TEAL,
    )
    box(
        d,
        row3_x[3],
        row3_y,
        row3_w,
        48,
        ["Reconstruct + inject", "H_hat = mu + sigma * Delta", "residual inputs, blocks 0-7"],
        colors.HexColor("#FDEBEC"),
        RED,
    )
    for index in range(3):
        arrow(
            d,
            row3_x[index] + row3_w,
            row3_y + 24,
            row3_x[index + 1] - 3,
            row3_y + 24,
        )
    d.add(
        Line(
            row2_x[2] + row2_w / 2,
            row2_y,
            row2_x[2] + row2_w / 2,
            row3_y + 63,
            strokeColor=MID,
            strokeWidth=1.4,
        )
    )
    arrow(
        d,
        row2_x[2] + row2_w / 2,
        row3_y + 63,
        row3_x[1] + row3_w / 2,
        row3_y + 48,
    )
    d.add(
        String(
            width / 2,
            14,
            "The mean scaffold and site scales are computed from training tasks only.",
            fontName=FONTS["bold"],
            fontSize=7.3,
            fillColor=RED,
            textAnchor="middle",
        )
    )
    caption = Paragraph(
        "<b>Figure 3.</b> Architecture of the registered query-conditioned nonlinear bridge used in Study B. The diagram describes the tested system and its training-only scaffold reconstruction; it does not imply successful functional semantic transport.",
        STYLES["caption"],
    )
    return KeepTogether([d, caption])


def p014_figure(width: float) -> KeepTogether:
    return bar_figure(
        width,
        ["Text", "Oracle M", "Oracle S", "Mean", "Learned M", "Learned S", "Random"],
        [92.7083, 87.5, 0.0, 0.0, 0.3472, 0.3472, 0.0],
        [BLUE, TEAL, MID, MID, ORANGE, RED, MID],
        "<b>Figure 4.</b> Study B sealed functional confirmation. The receiver-native oracle identity contrast is large; learned matched and learned shuffled rates are identical. Exact denominators are reported in the table below.",
    )


FIGURES: dict[str, Callable[[float], KeepTogether]] = {
    "PIPELINE": pipeline_figure,
    "P013": p013_figure,
    "BRIDGE": bridge_architecture_figure,
    "P014": p014_figure,
}


def table_widths(column_count: int, available: float) -> list[float]:
    if column_count == 2:
        weights = [2.7, 1.0]
    elif column_count == 3:
        weights = [2.5, 1.0, 1.0]
    elif column_count == 4:
        weights = [2.4, 2.0, 1.0, 1.0]
    elif column_count == 5:
        weights = [2.5, 1.05, 1.25, 0.8, 1.0]
    else:
        weights = [1.0] * column_count
    scale = available / sum(weights)
    return [weight * scale for weight in weights]


def make_table(rows: list[list[str]], width: float) -> Table:
    rendered: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = STYLES["table_head"] if row_index == 0 else STYLES["table"]
        rendered.append([Paragraph(inline_markup(cell.strip()), style) for cell in row])
    table = Table(
        rendered,
        colWidths=table_widths(len(rows[0]), width),
        repeatRows=1,
        hAlign="LEFT",
        spaceBefore=4,
        spaceAfter=8,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C3CF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F5F7F9")))
    table.setStyle(TableStyle(commands))
    return table


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def manuscript_story(metadata: dict[str, str], lines: list[str], width: float) -> list:
    story: list = []
    story.append(Spacer(1, 9 * mm))
    story.append(Paragraph(inline_markup(metadata["title"]), STYLES["title"]))
    story.append(Paragraph(inline_markup(metadata["subtitle"]), STYLES["subtitle"]))
    story.append(Paragraph(inline_markup(metadata["author"]), STYLES["author"]))
    story.append(Paragraph(inline_markup(metadata["affiliation"]), STYLES["meta"]))
    story.append(Spacer(1, 5))
    story.append(
        Paragraph(
            inline_markup(
                f'{metadata["project"]} · {metadata["status"]} · version {metadata["version"]} · {metadata["date"]}'
            ),
            STYLES["meta"],
        )
    )
    story.append(Paragraph(inline_markup(metadata["repository"]), STYLES["meta"]))
    story.append(Paragraph(inline_markup(metadata["license"]), STYLES["meta"]))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.1, color=BLUE, spaceBefore=3, spaceAfter=9))
    callout = Table(
        [[Paragraph("Receiver-native residual packets carried task identity; the registered learned cross-model bridge did not.", STYLES["callout"])]],
        colWidths=[width],
    )
    callout.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B9D2E8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend([callout, Spacer(1, 9)])

    paragraph: list[str] = []
    current_section = ""
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = " ".join(part.strip() for part in paragraph).strip()
        paragraph = []
        if not text:
            return
        if current_section == "Abstract":
            style = STYLES["abstract"]
        elif current_section == "References" and re.match(r"^\[\d+\]", text):
            style = STYLES["reference"]
        else:
            style = STYLES["body"]
        story.append(Paragraph(inline_markup(text), style))

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped == "[[PAGEBREAK]]":
            flush_paragraph()
            story.append(PageBreak())
            index += 1
            continue

        figure_match = re.fullmatch(r"\[\[FIGURE:([A-Z0-9_-]+)\]\]", stripped)
        if figure_match:
            flush_paragraph()
            figure_id = figure_match.group(1)
            if figure_id not in FIGURES:
                raise ValueError(f"Unknown figure marker: {figure_id}")
            story.append(FIGURES[figure_id](width))
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2)
            current_section = title if level == 1 else current_section
            story.append(Paragraph(inline_markup(title), STYLES[f"h{level}"]))
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and is_separator(lines[index + 1]):
            flush_paragraph()
            raw_rows = [split_table_row(stripped)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                raw_rows.append(split_table_row(lines[index]))
                index += 1
            if any(len(row) != len(raw_rows[0]) for row in raw_rows):
                raise ValueError("Inconsistent Markdown table width.")
            story.append(make_table(raw_rows, width))
            continue

        bullet = re.match(r"^-\s+(.+)$", stripped)
        if bullet:
            flush_paragraph()
            story.append(Paragraph(inline_markup(bullet.group(1)), STYLES["bullet"], bulletText="•"))
            index += 1
            continue

        numbered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if numbered:
            flush_paragraph()
            numbered_items = []
            while index < len(lines):
                item_match = re.match(r"^(\d+)\.\s+(.+)$", lines[index].strip())
                if not item_match:
                    break
                numbered_items.append(
                    Paragraph(
                        inline_markup(item_match.group(2)),
                        STYLES["numbered"],
                        bulletText=f'{item_match.group(1)}.',
                    )
                )
                index += 1
            story.append(KeepTogether(numbered_items))
            continue

        if line.startswith("    "):
            flush_paragraph()
            code_lines = [line[4:]]
            index += 1
            while index < len(lines) and (lines[index].startswith("    ") or not lines[index].strip()):
                code_lines.append(lines[index][4:] if lines[index].startswith("    ") else "")
                index += 1
            story.append(Preformatted("\n".join(code_lines).rstrip(), STYLES["equation"]))
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return story


def page_decor(canvas, doc=None) -> None:
    canvas.saveState()
    canvas.setTitle("Receiver-Anchored Tests for Latent Communication")
    canvas.setAuthor("Cristiano Silva")
    canvas.setSubject("LIP receiver-anchored latent communication evaluation protocol")
    canvas.setKeywords("latent communication, activation intervention, residual packet, cross-model alignment")
    page = canvas.getPageNumber()
    page_width, page_height = A4
    canvas.setFont(FONTS["regular"], 6.7)
    canvas.setFillColor(MID)
    canvas.drawRightString(page_width - 18 * mm, 9.5 * mm, str(page))
    canvas.restoreState()


def build(source: Path, output: Path) -> None:
    metadata, lines = parse_source(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f'{metadata["title"]}: {metadata["subtitle"]}',
        author=metadata["author"],
        subject="Receiver-anchored evaluation of latent communication",
        creator="LIP preprint renderer",
        pageCompression=1,
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="manuscript",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates(
        [PageTemplate(id="paper", frames=[frame], onPageEnd=page_decor)]
    )
    story = manuscript_story(metadata, lines, doc.width)
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the LIP preprint as a publication-ready PDF.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()} ({args.output.resolve().stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
