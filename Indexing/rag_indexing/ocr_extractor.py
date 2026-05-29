"""
OCR Document Extraction Module
Dùng EasyOCR thay PaddleOCR (tương thích Windows, không cần paddle)
+ PyMuPDF để render PDF
"""
import logging
import re
from pathlib import Path
from typing import Optional
import numpy as np

from .config import OCRConfig
from .models import DocumentRegion, BoundingBox, RegionType

logger = logging.getLogger(__name__)


class PaddleOCRExtractor:
    """
    OCR Extractor dùng EasyOCR + PyMuPDF.
    Giữ nguyên interface để pipeline.py không cần sửa.
    """

    def __init__(self, config: OCRConfig):
        self.config = config
        self._ocr = None       # EasyOCR reader
        self._init_engines()

    def _init_engines(self):
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            self._ocr = easyocr.Reader(
                [self.config.lang if self.config.lang != "ch" else "ch_sim"],
                gpu=use_gpu,
                verbose=False,
            )
            device = "GPU" if use_gpu else "CPU"
            logger.info(f"EasyOCR initialized ({device} mode)")
        except ImportError:
            raise ImportError(
                "easyocr not installed.\n"
                "Run: pip install easyocr pymupdf"
            )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def extract_from_file(self, file_path: Path) -> list[DocumentRegion]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            return self._extract_image_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # ------------------------------------------------------------------ #
    # PDF extraction                                                        #
    # ------------------------------------------------------------------ #

    def _extract_pdf(self, pdf_path: Path) -> list[DocumentRegion]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required: pip install pymupdf")

        import concurrent.futures, torch

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        # Pass 1: lấy text layer (nhanh, single-thread) + render scan pages
        text_results: list[tuple[int, list]] = []
        scan_jobs:   list[tuple[int, bytes]] = []

        for page_num, page in enumerate(doc):
            text_regions = self._extract_text_layer(page, page_num)
            if text_regions:
                text_results.append((page_num, text_regions))
            else:
                mat = fitz.Matrix(2.0, 2.0)
                img_bytes = page.get_pixmap(matrix=mat).tobytes("png")
                scan_jobs.append((page_num, img_bytes))
        doc.close()

        logger.info(f"  {len(text_results)} text pages, {len(scan_jobs)} scan pages (OCR)")

        all_regions: list[DocumentRegion] = []
        for _, regions in text_results:
            all_regions.extend(regions)

        # Pass 2: OCR scan pages song song
        # EasyOCR GPU không thread-safe → 1 worker; CPU → tối đa 4
        if scan_jobs:
            max_workers = 1 if torch.cuda.is_available() else min(4, len(scan_jobs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(self._extract_image_bytes, img, pn): pn
                    for pn, img in scan_jobs
                }
                for fut in concurrent.futures.as_completed(futures):
                    pn = futures[fut]
                    try:
                        all_regions.extend(fut.result())
                    except Exception as e:
                        logger.warning(f"  Page {pn+1} OCR failed: {e}")

        all_regions.sort(key=lambda r: (r.page_num, r.bbox.y1 if r.bbox else 0))
        logger.info(f"Extraction complete: {len(all_regions)} regions from {total_pages} pages")
        return all_regions

    def _extract_text_layer(self, page, page_num: int) -> list[DocumentRegion]:
        """
        Dùng PyMuPDF để lấy text block trực tiếp từ PDF có text layer.
        Trả về [] nếu trang không có text (PDF scan).
        """
        blocks = page.get_text("dict")["blocks"]
        regions = []

        for block in blocks:
            btype = block.get("type", -1)

            # type 0 = text block
            if btype == 0:
                lines_text = []
                for line in block.get("lines", []):
                    line_text = " ".join(
                        span["text"] for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    )
                    if line_text.strip():
                        lines_text.append(line_text)

                text = "\n".join(lines_text).strip()
                if not text:
                    continue

                bbox_raw = block.get("bbox", (0, 0, 0, 0))
                bbox = BoundingBox(
                    x1=bbox_raw[0], y1=bbox_raw[1],
                    x2=bbox_raw[2], y2=bbox_raw[3],
                    page=page_num,
                )

                # Phân loại region
                region_type = self._classify_text_block(text, block)

                regions.append(DocumentRegion(
                    region_type=region_type,
                    text=text,
                    raw_text=text,
                    confidence=1.0,
                    bbox=bbox,
                    page_num=page_num,
                    metadata={"source": "text_layer"},
                ))

            # type 1 = image block
            elif btype == 1:
                bbox_raw = block.get("bbox", (0, 0, 0, 0))
                regions.append(DocumentRegion(
                    region_type=RegionType.IMAGE,
                    text="",
                    confidence=1.0,
                    bbox=BoundingBox(*bbox_raw[:4], page=page_num),
                    page_num=page_num,
                    metadata={"source": "image_block"},
                ))

        # Detect inline math ($...$) trong text regions
        math_regions = self._detect_math_regions(regions, page_num)
        regions.extend(math_regions)

        # Detect tables (heuristic: nhiều tab / | )
        regions = self._detect_tables(regions, page_num)

        return regions

    def _classify_text_block(self, text: str, block: dict) -> RegionType:
        """Phân loại block dựa trên font size và nội dung."""
        # Lấy font size lớn nhất trong block
        max_size = 0.0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                max_size = max(max_size, span.get("size", 0))

        # Heuristic: font lớn = title/header
        if max_size >= 16:
            return RegionType.TITLE
        if max_size >= 13:
            return RegionType.HEADER

        # Table heuristic
        if text.count("|") >= 4 or text.count("\t") >= 3:
            return RegionType.TABLE

        # Formula heuristic
        if re.search(r'\$[^$]+\$|\\frac|\\sum|\\int|\\alpha', text):
            return RegionType.FORMULA

        return RegionType.TEXT

    # ------------------------------------------------------------------ #
    # Image / scan OCR with EasyOCR                                        #
    # ------------------------------------------------------------------ #

    def _extract_image_file(self, img_path: Path) -> list[DocumentRegion]:
        img_bytes = img_path.read_bytes()
        return self._extract_image_bytes(img_bytes, page_num=0)

    def _extract_image_bytes(
        self, img_bytes: bytes, page_num: int = 0
    ) -> list[DocumentRegion]:
        """Run EasyOCR on raw image bytes."""
        import cv2

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Could not decode image for page {page_num}")
            return []

        try:
            results = self._ocr.readtext(img, detail=1, paragraph=False)
        except Exception as e:
            logger.error(f"EasyOCR failed on page {page_num}: {e}")
            return []

        regions = []
        for (bbox_pts, text, conf) in results:
            text = text.strip()
            if not text or conf < 0.3:
                continue

            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            bbox = BoundingBox(
                x1=min(xs), y1=min(ys),
                x2=max(xs), y2=max(ys),
                page=page_num,
            )

            region_type = self._classify_ocr_text(text, conf)
            regions.append(DocumentRegion(
                region_type=region_type,
                text=text,
                raw_text=text,
                confidence=float(conf),
                bbox=bbox,
                page_num=page_num,
                metadata={"source": "easyocr"},
            ))

        # Group nearby lines into paragraphs
        regions = self._group_into_paragraphs(regions, page_num)

        # Detect math
        math_regions = self._detect_math_regions(regions, page_num)
        regions.extend(math_regions)

        return regions

    def _classify_ocr_text(self, text: str, conf: float) -> RegionType:
        """Phân loại text từ EasyOCR."""
        if re.search(r'\$[^$]+\$|\\frac|\\sum|\\int', text):
            return RegionType.FORMULA
        if text.count("|") >= 3:
            return RegionType.TABLE
        if len(text.split()) <= 5 and text.isupper():
            return RegionType.TITLE
        return RegionType.TEXT

    def _group_into_paragraphs(
        self, regions: list[DocumentRegion], page_num: int
    ) -> list[DocumentRegion]:
        """
        Gộp các text line gần nhau (cùng cột, y gần) thành paragraph.
        Giữ nguyên formula/table regions.
        """
        text_regions = [r for r in regions if r.region_type == RegionType.TEXT]
        other_regions = [r for r in regions if r.region_type != RegionType.TEXT]

        if not text_regions:
            return regions

        # Sắp xếp theo y (top-to-bottom)
        text_regions.sort(key=lambda r: r.bbox.y1 if r.bbox else 0)

        merged = []
        current_group: list[DocumentRegion] = [text_regions[0]]

        for region in text_regions[1:]:
            prev = current_group[-1]
            prev_y2 = prev.bbox.y2 if prev.bbox else 0
            curr_y1 = region.bbox.y1 if region.bbox else 0
            line_height = (prev.bbox.y2 - prev.bbox.y1) if prev.bbox else 20

            # Nếu khoảng cách dọc nhỏ hơn 1.5x line height → cùng paragraph
            if (curr_y1 - prev_y2) < line_height * 1.5:
                current_group.append(region)
            else:
                merged.append(self._merge_group(current_group, page_num))
                current_group = [region]

        if current_group:
            merged.append(self._merge_group(current_group, page_num))

        return merged + other_regions

    def _merge_group(
        self, group: list[DocumentRegion], page_num: int
    ) -> DocumentRegion:
        text = " ".join(r.text for r in group)
        conf = float(np.mean([r.confidence for r in group]))
        bboxes = [r.bbox for r in group if r.bbox]
        if bboxes:
            bbox = BoundingBox(
                x1=min(b.x1 for b in bboxes),
                y1=min(b.y1 for b in bboxes),
                x2=max(b.x2 for b in bboxes),
                y2=max(b.y2 for b in bboxes),
                page=page_num,
            )
        else:
            bbox = None
        return DocumentRegion(
            region_type=RegionType.TEXT,
            text=text,
            raw_text=text,
            confidence=conf,
            bbox=bbox,
            page_num=page_num,
            metadata={"source": "merged_paragraph", "line_count": len(group)},
        )

    # ------------------------------------------------------------------ #
    # Math & Table detection                                                #
    # ------------------------------------------------------------------ #

    def _detect_math_regions(
        self, regions: list[DocumentRegion], page_num: int
    ) -> list[DocumentRegion]:
        """Tìm inline math trong text regions ($...$ hoặc LaTeX commands)."""
        math_pattern = re.compile(
            r'\$+[^$]+\$+|'
            r'\\(?:frac|sum|int|prod|lim|sqrt|alpha|beta|gamma|delta|theta|lambda|sigma|pi|mu)\b[^.]{0,200}'
        )
        math_regions = []

        for region in regions:
            if region.region_type not in (RegionType.TEXT, RegionType.TITLE):
                continue
            matches = math_pattern.findall(region.text)
            for match in matches:
                clean = match.strip("$").strip()
                if len(clean) > 3:
                    math_regions.append(DocumentRegion(
                        region_type=RegionType.FORMULA,
                        text=clean,
                        raw_text=match,
                        confidence=0.85,
                        bbox=region.bbox,
                        page_num=page_num,
                        metadata={
                            "source": "inline_math",
                            "parent_region": region.id,
                        },
                    ))
        return math_regions

    def _detect_tables(
        self, regions: list[DocumentRegion], page_num: int
    ) -> list[DocumentRegion]:
        """
        Heuristic: gộp các lines liền kề có nhiều | thành TABLE region.
        """
        result = []
        table_buffer: list[DocumentRegion] = []

        for region in regions:
            is_table_line = (
                region.region_type == RegionType.TEXT and
                (region.text.count("|") >= 2 or region.text.count("\t") >= 2)
            )
            if is_table_line:
                table_buffer.append(region)
            else:
                if len(table_buffer) >= 2:
                    # Merge thành TABLE region
                    merged = self._merge_group(table_buffer, page_num)
                    merged.region_type = RegionType.TABLE
                    result.append(merged)
                elif table_buffer:
                    result.extend(table_buffer)
                table_buffer = []
                result.append(region)

        if len(table_buffer) >= 2:
            merged = self._merge_group(table_buffer, page_num)
            merged.region_type = RegionType.TABLE
            result.append(merged)
        elif table_buffer:
            result.extend(table_buffer)

        return result

    # ------------------------------------------------------------------ #
    # Crop helper (dùng cho LaTeX parser)                                  #
    # ------------------------------------------------------------------ #

    def _crop_image_bytes(
        self, img_bytes: bytes, bbox: BoundingBox
    ) -> Optional[bytes]:
        try:
            import cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))
            if x2 <= x1 or y2 <= y1:
                return None
            crop = img[y1:y2, x1:x2]
            _, buf = cv2.imencode(".png", crop)
            return bytes(buf)
        except Exception as e:
            logger.warning(f"Crop failed: {e}")
            return None