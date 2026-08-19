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
    Converts a single BeautifulSoup <table> tag into a Markdown table.
    """
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    grid: List[List[str]] = []
    max_cols = 0

    for row in rows:
        cells = row.find_all(["th", "td"])
        row_data: List[str] = []
        for cell in cells:
            # Clean cell text: strip whitespace and replace internal line breaks with <br>
            cell_text = cell.get_text(separator=" ", strip=True)
            cell_text = cell_text.replace("|", "\\|").replace("\n", "<br>")
            row_data.append(cell_text)

        if row_data:
            grid.append(row_data)
            max_cols = max(max_cols, len(row_data))

    if not grid or max_cols == 0:
        return ""

    # Normalize row lengths to max_cols
    for row in grid:
        while len(row) < max_cols:
            row.append("")

    # Determine column widths for aligned formatting
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
