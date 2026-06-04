import os
import re
from collections import defaultdict
from fastapi import HTTPException
import fitz  # PyMuPDF

ALLOWED_EXTENSIONS = (".pdf", ".txt")

# UNICODE_TO_LATEX mapping removed to simplify parsing.



def validate_file_extension(filename: str | None) -> None:
    """Raise 400 if the filename is missing or not an allowed type."""
    if not filename or not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="invalid file format")


def unicode_to_latex(text: str) -> str:
    """Convert unicode mathematical/Greek symbols to standard LaTeX notation."""
    # Simplified parsing logic: let the LLM handle LaTeX formatting and syntax conversion during generation
    return text



def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping vertical intervals with a small vertical tolerance."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        tolerance = 5.0  # Vertical grouping sensitivity in pixels
        if curr_start <= prev_end + tolerance:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    return merged


def cluster_and_sort_columns(blocks_list: list[dict]) -> list[str]:
    """Group blocks into horizontal columns and sort left-to-right."""
    if not blocks_list:
        return []
    
    columns = []
    # Sort blocks from left to right first to find columns
    sorted_blocks = sorted(blocks_list, key=lambda b: b["x0"])
    
    for block in sorted_blocks:
        placed = False
        for col in columns:
            col_x0 = min(b["x0"] for b in col)
            col_x1 = max(b["x1"] for b in col)
            
            overlap = max(0, min(block["x1"], col_x1) - max(block["x0"], col_x0))
            block_width = block["x1"] - block["x0"]
            
            # If horizontal overlap is significant (> 30% of block width)
            if block_width > 0 and (overlap / block_width) > 0.3:
                col.append(block)
                placed = True
                break
        if not placed:
            columns.append([block])
            
    # Sort blocks inside each column top-to-bottom
    sorted_columns = []
    for col in columns:
        sorted_col = sorted(col, key=lambda b: b["y0"])
        col_x0 = min(b["x0"] for b in sorted_col)
        sorted_columns.append((col_x0, sorted_col))
        
    # Sort columns from left to right
    sorted_columns.sort(key=lambda c: c[0])
    
    out = []
    for _, col in sorted_columns:
        for b in col:
            out.append(b["text"])
    return out


def extract_smart_layout(page) -> str:
    """Smart extraction of a single PDF page handling columns, tables, headings, and margins."""
    width = page.rect.width
    height = page.rect.height
    
    # Establish standard margins (top 8%, bottom 8% of page height)
    top_margin = height * 0.08
    bottom_margin = height * 0.92
    
    toc_lines_count = 0
    toc_dot_pattern = re.compile(r'\.\s*\.\s*\.\s*\.\s*\.')
    
    # 1. Table Detection
    tables = page.find_tables(strategy="lines")
    table_bboxes = []
    formatted_tables = []
    
    for tab in tables.tables:
        table_bboxes.append(tab.bbox)
        data = tab.extract()
        if not data or len(data) < 1:
            continue
        
        headers = [str(h or "").strip() for h in data[0]]
        headers = [h if h else f"Col {idx+1}" for idx, h in enumerate(headers)]
        
        table_rows = []
        for row in data[1:]:
            row_cells = []
            for col_idx, val in enumerate(row):
                if col_idx < len(headers):
                    header = headers[col_idx]
                    cell_val = str(val or "").strip()
                    row_cells.append(f"{header}: {cell_val}")
            table_rows.append("; ".join(row_cells))
            
        table_text = "[Table Start]\n" + "\n".join(table_rows) + "\n[Table End]"
        y_center = (tab.bbox[1] + tab.bbox[3]) / 2
        formatted_tables.append({
            "bbox": tab.bbox,
            "text": table_text,
            "y0": tab.bbox[1],
            "y1": tab.bbox[3],
            "x0": tab.bbox[0],
            "x1": tab.bbox[2],
            "type": "table"
        })

    # 2. Extract layout dict
    page_dict = page.get_text("dict")
    
    # Calculate body size (most frequent rounded size)
    font_sizes = defaultdict(int)
    for block in page_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        size = round(span.get("size", 10))
                        font_sizes[size] += len(text)
    
    body_size = max(font_sizes.keys(), key=lambda s: font_sizes[s]) if font_sizes else 10

    # Process block layouts
    text_blocks = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip images/drawings
            continue
            
        bbox = block.get("bbox", (0, 0, 0, 0))
        bx0, by0, bx1, by1 = bbox
        
        # Filter margins
        if by1 < top_margin or by0 > bottom_margin:
            continue
            
        # Check table overlap
        in_table = False
        for t_bbox in table_bboxes:
            tx0, ty0, tx1, ty1 = t_bbox
            overlap_x = max(0, min(bx1, tx1) - max(bx0, tx0))
            overlap_y = max(0, min(by1, ty1) - max(by0, ty0))
            block_area = (bx1 - bx0) * (by1 - by0)
            if block_area > 0 and (overlap_x * overlap_y) > 0.5 * block_area:
                in_table = True
                break
                
        if in_table:
            continue
            
        # Reconstruct block text and detect headings
        block_lines = []
        is_heading = False
        
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                if not span_text:
                    continue
                
                span_size = span.get("size", 10)
                if span_size >= body_size * 1.25:  # Font size is 25% larger than body
                    is_heading = True
                    
                line_text += " " + span.get("text", "")
            
            cleaned_line = line_text.strip()
            if cleaned_line:
                if toc_dot_pattern.search(cleaned_line):
                    toc_lines_count += 1
                    continue
                block_lines.append(cleaned_line)
                
        block_text = "\n".join(block_lines)
        if not block_text:
            continue
            
        if is_heading:
            block_text = f"[Heading: {block_text}]"
            
        text_blocks.append({
            "text": block_text,
            "x0": bx0,
            "y0": by0,
            "x1": bx1,
            "y1": by1,
            "type": "text"
        })

    if toc_lines_count > 5:
        return ""

    # Combine text and tables
    all_elements = text_blocks + formatted_tables
    if not all_elements:
        # Fallback to default raw text in case of empty layout groups
        flags = (
            fitz.TEXT_PRESERVE_WHITESPACE
            | fitz.TEXT_PRESERVE_LIGATURES
            | fitz.TEXT_MEDIABOX_CLIP
        )
        return str(page.get_text("text", flags=flags)).strip()

    # 3. Interval Merging
    intervals = [(float(el["y0"]), float(el["y1"])) for el in all_elements]
    merged_intervals = merge_intervals(intervals)

    band_elements = defaultdict(list)
    for el in all_elements:
        mid_y = (float(el["y0"]) + float(el["y1"])) / 2
        for band in merged_intervals:
            if band[0] <= mid_y <= band[1]:
                band_elements[band].append(el)
                break

    final_output = []
    for band in sorted(merged_intervals, key=lambda x: x[0]):
        elements_in_band = band_elements[band]
        if len(elements_in_band) == 1:
            final_output.append(elements_in_band[0]["text"])
        else:
            texts_in_band = [el for el in elements_in_band if el["type"] == "text"]
            tables_in_band = [el for el in elements_in_band if el["type"] == "table"]
            
            sorted_texts = cluster_and_sort_columns(texts_in_band)
            final_output.extend(sorted_texts)
            for tab in tables_in_band:
                final_output.append(tab["text"])
                
    return "\n\n".join(final_output)


def extract_text(filename: str | None, content: bytes) -> str:
    """
    Extract plain text from raw file bytes.

    Supports:
      - .pdf  → parsed with Smart PyMuPDF Layout Parser (and converted to LaTeX notation)
      - .txt  → decoded as UTF-8

    Raises HTTPException 400 on parse / encoding failure.
    """
    if filename and filename.endswith(".pdf"):
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            pages: list[str] = []
            for page in doc:
                page_text = extract_smart_layout(page)
                if page_text:
                    pages.append(page_text)
            doc.close()
            full_text = "\n\n".join(pages)
            return unicode_to_latex(full_text)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"invalid PDF file or extraction failed: {str(e)}",
            )
    else:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="invalid file encoding")


# ---------------------------------------------------------------------------
# Storage-item helpers (used by AttachmentService.get_documents)
# ---------------------------------------------------------------------------

def parse_storage_item_name(name: str) -> tuple[str, str]:
    """
    Split a storage object name into (file_id, original_filename).

    Names are stored as  "<uuid>_<original_filename>".
    If no underscore is present the whole name is used for both fields.
    """
    if "_" in name:
        file_id, name_part = name.split("_", 1)
    else:
        file_id = name_part = name
    return file_id, name_part


def format_size_kb(size_bytes: int) -> str:
    """Return a human-readable kilobyte string, e.g. '123.4 KB'."""
    return f"{size_bytes / 1024:.1f} KB"


def extract_text_preview(file_bytes: bytes, filename: str, max_chars: int = 5000) -> str:
    """
    Return up to *max_chars* characters of plain text from *file_bytes*.

    - PDF  → parsed page-by-page with PyMuPDF; stops as soon as the limit is reached.
    - Other → decoded as UTF-8 (replacing bad bytes) and sliced.

    Returns an empty string on any error.
    """
    try:
        if filename.endswith(".pdf"):
            text = ""
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += str(page.get_text("text"))
                if len(text) >= max_chars:
                    text = text[:max_chars]
                    break
            doc.close()
            return unicode_to_latex(text)
        else:
            return file_bytes.decode("utf-8", errors="ignore")[:max_chars]
    except Exception as ex:
        print("Error extracting text preview:", ex)
        return ""
