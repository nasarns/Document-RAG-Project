import uuid
from typing import List, Dict, Any


class TextChunker:
    """
    Splits text blocks into overlapping chunks while preserving
    document provenance and citation metadata.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document_blocks(self, blocks: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """
        Takes extracted blocks from DocumentExtractor and produces fine-grained chunks.

        Returns:
            List of chunk dicts:
            [
                {
                    "chunk_id": str,
                    "doc_id": str,
                    "text": str,
                    "metadata": {
                        "doc_id": str,
                        "filename": str,
                        "file_type": str,
                        "page_number": int | None,
                        "sheet_name": str | None,
                        "section": str | None,
                        "chunk_index": int,
                        "citation_label": str
                    }
                }
            ]
        """
        all_chunks = []
        global_chunk_idx = 0

        for block in blocks:
            raw_text = block["text"].strip()
            base_meta = block["metadata"]

            if not raw_text:
                continue

            # If block text is small enough, keep as single chunk
            if len(raw_text) <= self.chunk_size:
                text_splits = [raw_text]
            else:
                text_splits = self._split_text(raw_text)

            for split_idx, split_text in enumerate(text_splits):
                # Build human-readable citation label
                citation_label = self._build_citation_label(base_meta, split_idx + 1)

                chunk_dict = {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "text": split_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": base_meta.get("filename", "Unknown"),
                        "file_type": base_meta.get("file_type", "txt"),
                        "page_number": base_meta.get("page_number"),
                        "sheet_name": base_meta.get("sheet_name"),
                        "section": base_meta.get("section"),
                        "chunk_index": global_chunk_idx,
                        "citation_label": citation_label
                    }
                }
                all_chunks.append(chunk_dict)
                global_chunk_idx += 1

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Splits text on paragraph/sentence boundaries with sliding overlap.
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try finding a natural break point (newline, period, space)
            break_point = -1
            # 1. Try paragraph break
            p_break = text.rfind("\n\n", start + self.chunk_size // 2, end)
            if p_break != -1:
                break_point = p_break + 2
            else:
                # 2. Try single newline
                nl_break = text.rfind("\n", start + self.chunk_size // 2, end)
                if nl_break != -1:
                    break_point = nl_break + 1
                else:
                    # 3. Try sentence end
                    s_break = text.rfind(". ", start + self.chunk_size // 2, end)
                    if s_break != -1:
                        break_point = s_break + 2
                    else:
                        # 4. Try whitespace
                        w_break = text.rfind(" ", start + self.chunk_size // 2, end)
                        if w_break != -1:
                            break_point = w_break + 1
                        else:
                            break_point = end

            chunk = text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            start = max(start + 1, break_point - self.chunk_overlap)

        return chunks

    @staticmethod
    def _build_citation_label(meta: Dict[str, Any], part_num: int) -> str:
        fname = meta.get("filename", "Doc")
        page = meta.get("page_number")
        sheet = meta.get("sheet_name")
        section = meta.get("section")

        if page is not None:
            return f"{fname} (Page {page})"
        elif sheet is not None:
            return f"{fname} [{sheet}]"
        elif section is not None and section != "General":
            return f"{fname} - {section}"
        else:
            return f"{fname} (Part {part_num})"
