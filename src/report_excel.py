"""
Excel model report: one workbook per training run, tables and figures on the same sheets.

Usage
-----
    report = ExcelReport()
    report.add_text("Summary", "Model comparison", style="title")
    report.add_table("Summary", df, title="Hold-out metrics")
    report.add_image("Summary", "figures/roc.png")
    report.save("runs/<run_id>/model_report_<run_id>.xlsx")

Blocks are written top to bottom in the order they are added to each sheet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=14, color="1F4E78")
SUBTITLE_FONT = Font(bold=True, size=11, color="1F4E78")
THIN = Side(style="thin", color="BFBFBF")
PASS_FILL = PatternFill("solid", fgColor="E2EFDA")
FAIL_FILL = PatternFill("solid", fgColor="FCE4D6")

PCT_HINTS = ("rate", "pct", "share", "approval", "mean_pd", "ratio_obs")
INT_HINTS = ("rows", "bad", "trial", "n_", "best_iter", "grade", "df")
IMG_WIDTH_PX = 820
ROW_PX = 20


def _fmt(col: str, series: pd.Series) -> str:
    c = col.lower()
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return "General"
    if any(h in c for h in PCT_HINTS) and not c.startswith("ratio") and "ratio_obs" not in c:
        return "0.00%"
    if pd.api.types.is_integer_dtype(series) or (any(c.startswith(h) or c == h for h in INT_HINTS)
                                                 and np.allclose(series.dropna() % 1, 0)):
        return "#,##0"
    if "p_value" in c:
        return "0.0000E+00"
    if "psi" in c or "csi" in c:
        return "0.00000"
    if "score" in c or "cutoff" in c:
        return "0.0"
    return "0.0000"


class ExcelReport:
    def __init__(self):
        self.sheets: dict[str, list] = {}

    def _blocks(self, sheet):
        return self.sheets.setdefault(sheet, [])

    def add_text(self, sheet, text, style="normal"):
        self._blocks(sheet).append(("text", text, style))

    def add_kv(self, sheet, mapping: dict, title=None):
        df = pd.DataFrame({"item": list(mapping), "value": [str(v) for v in mapping.values()]})
        self._blocks(sheet).append(("table", df, title))

    def add_table(self, sheet, df: pd.DataFrame, title=None):
        self._blocks(sheet).append(("table", df.reset_index(drop=True), title))

    def add_image(self, sheet, path, title=None):
        self._blocks(sheet).append(("image", str(path), title))

    # ------------------------------------------------------------------ write
    def save(self, path):
        wb = Workbook()
        wb.remove(wb.active)
        for name, blocks in self.sheets.items():
            ws = wb.create_sheet(name[:31])
            ws.sheet_view.showGridLines = False
            r = 1
            widths: dict[int, int] = {}
            for b in blocks:
                if b[0] == "text":
                    _, text, style = b
                    cell = ws.cell(r, 1, text)
                    cell.font = TITLE_FONT if style == "title" else SUBTITLE_FONT if style == "subtitle" else Font()
                    if style == "normal":
                        cell.alignment = Alignment(wrap_text=False)
                    r += 2 if style == "title" else 1
                elif b[0] == "table":
                    _, df, title = b
                    if title:
                        ws.cell(r, 1, title).font = SUBTITLE_FONT
                        r += 1
                    for j, col in enumerate(df.columns, 1):
                        c = ws.cell(r, j, str(col))
                        c.fill, c.font, c.border = HEADER_FILL, HEADER_FONT, Border(bottom=THIN)
                        c.alignment = Alignment(horizontal="center", wrap_text=True)
                        widths[j] = max(widths.get(j, 0), min(len(str(col)) + 2, 28))
                    fmts = [_fmt(str(c), df[c]) for c in df.columns]
                    for i, row in enumerate(df.itertuples(index=False), 1):
                        for j, v in enumerate(row, 1):
                            if isinstance(v, (np.generic,)):
                                v = v.item()
                            if v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                                v = None
                            c = ws.cell(r + i, j, v)
                            c.number_format = fmts[j - 1]
                            c.border = Border(bottom=THIN)
                            if isinstance(v, bool):
                                c.fill = PASS_FILL if v else FAIL_FILL
                            widths[j] = max(widths.get(j, 0), min(len(str(v)) + 2 if not isinstance(v, float) else 12, 60))
                    r += len(df) + 2
                elif b[0] == "image":
                    _, p, title = b
                    if title:
                        ws.cell(r, 1, title).font = SUBTITLE_FONT
                        r += 1
                    if not Path(p).exists():
                        ws.cell(r, 1, f"(missing figure: {p})")
                        r += 2
                        continue
                    img = XLImage(p)
                    scale = IMG_WIDTH_PX / img.width
                    img.width, img.height = int(img.width * scale), int(img.height * scale)
                    ws.add_image(img, f"A{r}")
                    r += int(img.height / ROW_PX) + 2
            for j, w in widths.items():
                ws.column_dimensions[get_column_letter(j)].width = max(w, 10)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
        return str(path)
