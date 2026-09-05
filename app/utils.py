import logging
import numpy as np

logger = logging.getLogger("ettin-reranker")
logging.basicConfig(level=logging.INFO)

try:
    from scipy.special import erf

    def _gelu_numpy(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
except ImportError:

    def _gelu_numpy(x: np.ndarray) -> np.ndarray:
        """GELU activation in pure NumPy using tanh approximation."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def html_table_to_markdown(html_content: str) -> str:
    """Converts HTML table markup into GitHub-Flavored Markdown (GFM) format."""
    if not html_content or "<table" not in html_content.lower():
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return ""

        md_tables = []
        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            grid_dict = {}
            r = 0
            for row in rows:
                c = 0
                cells = row.find_all(["th", "td"])
                for cell in cells:
                    # Skip cells that are already filled by a previous rowspan/colspan
                    while (r, c) in grid_dict:
                        c += 1
                    
                    text = cell.get_text(separator=" ", strip=True)
                    text = " ".join(text.split())
                    text = text.replace("|", "\\|")
                    
                    try:
                        colspan = int(cell.get("colspan", 1))
                    except (ValueError, TypeError):
                        colspan = 1
                        
                    try:
                        rowspan = int(cell.get("rowspan", 1))
                    except (ValueError, TypeError):
                        rowspan = 1
                        
                    # Populate the grid coordinates covered by this cell
                    for i in range(rowspan):
                        for j in range(colspan):
                            grid_dict[(r + i, c + j)] = text if (i == 0 and j == 0) else ""
                    
                    c += colspan
                r += 1

            if not grid_dict:
                continue

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
                    continue

            col_widths = [3] * max_cols
            for row in grid:
                for col_idx, cell in enumerate(row):
                    col_widths[col_idx] = max(col_widths[col_idx], len(cell))

            lines = []
            header = grid[0]
            lines.append("| " + " | ".join(header[i].ljust(col_widths[i]) for i in range(max_cols)) + " |")
            lines.append("| " + " | ".join("-" * col_widths[i] for i in range(max_cols)) + " |")
            for row in grid[1:]:
                lines.append("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(max_cols)) + " |")

            md_tables.append("\n".join(lines))

        return "\n\n".join(md_tables)
    except Exception as e:
        logger.warning(f"Failed to convert HTML table to Markdown: {e}")
        return ""


def _get_tensor(tensor_dict: dict, *candidate_keys):
    """Safely retrieves a tensor matching candidate keys from a safetensors dictionary."""
    if not tensor_dict:
        return None
    for k in candidate_keys:
        if k in tensor_dict:
            return tensor_dict[k]
    for k, v in tensor_dict.items():
        for cand in candidate_keys:
            if k.endswith(cand):
                return v
    return None
