"""
HTML Table to Markdown Converter.
Converts standard HTML tables to GitHub-Flavored Markdown (GFM) format.
"""

from typing import List, Optional
from bs4 import BeautifulSoup


def html_table_to_markdown(html_content: str) -> str:
    """
    Parses an HTML string and converts table elements into GitHub-Flavored Markdown.

    Args:
        html_content: String containing HTML table markup.

    Returns:
        Converted Markdown string.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        return ""

    markdown_tables: List[str] = []

    for table in tables:
        md_table = _convert_single_table(table)
        if md_table:
            markdown_tables.append(md_table)

    return "\n\n".join(markdown_tables)


def _convert_single_table(table_tag) -> str:
    """
    Converts a single BeautifulSoup <table> tag into a Markdown table,
    handling colspan, rowspan, and pruning phantom empty columns.
    """
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    grid_dict = {}
    r = 0
    for row in rows:
        c = 0
        cells = row.find_all(["th", "td"])
        for cell in cells:
            while (r, c) in grid_dict:
                c += 1

            text = cell.get_text(separator=" ", strip=True)
            text = text.replace("|", "\\|").replace("\n", "<br>").strip()

            try:
                colspan = int(cell.get("colspan", 1))
            except (ValueError, TypeError):
                colspan = 1

            try:
                rowspan = int(cell.get("rowspan", 1))
            except (ValueError, TypeError):
                rowspan = 1

            for i in range(rowspan):
                for j in range(colspan):
                    grid_dict[(r + i, c + j)] = text if (i == 0 and j == 0) else ""

            c += colspan
        r += 1

    if not grid_dict:
        return ""

    max_r = max(k[0] for k in grid_dict.keys())
    max_c = max(k[1] for k in grid_dict.keys())
    max_cols = max_c + 1

    grid = []
    for i in range(max_r + 1):
        row_data = [grid_dict.get((i, j), "") for j in range(max_cols)]
        grid.append(row_data)

    # Prune columns that are completely empty across all rows
    if grid and max_cols > 0:
        cols_with_data = [
            col_idx for col_idx in range(max_cols)
            if any(bool(grid[row_idx][col_idx].strip()) for row_idx in range(len(grid)))
        ]
        if cols_with_data:
            grid = [[row[c] for c in cols_with_data] for row in grid]
            max_cols = len(cols_with_data)
        else:
            return ""

    col_widths = [3] * max_cols
    for row in grid:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    lines: List[str] = []

    # First row is treated as header
    header_row = grid[0]
    header_line = "| " + " | ".join(header_row[i].ljust(col_widths[i]) for i in range(max_cols)) + " |"
    lines.append(header_line)

    # Separator row
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(max_cols)) + " |"
    lines.append(separator_line)

    # Subsequent rows are body rows
    for row in grid[1:]:
        body_line = "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(max_cols)) + " |"
        lines.append(body_line)

    return "\n".join(lines)
