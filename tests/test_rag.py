import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pymupdf
import docx
import pandas as pd

from app.core.extractors import DocumentExtractor
from app.core.chunker import TextChunker
from app.core.embeddings import embedding_manager
from app.core.vector_db import vector_db
from app.core.rag_engine import rag_engine


class TestDocumentRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = BASE_DIR / "data" / "test_samples"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create sample PDF using textboxes to ensure complete multi-line preservation
        cls.pdf_path = cls.test_dir / "sample_rag_paper.pdf"
        doc = pymupdf.open()
        
        # Page 1
        page1 = doc.new_page()
        rect1 = pymupdf.Rect(50, 50, 500, 400)
        p1_text = (
            "Title: Retrieval Augmented Generation for Question Answering\n"
            "Author: Abdul Nasar\n"
            "Retrieval-Augmented Generation (RAG) combines dense semantic retrieval "
            "with generative LLMs to reduce hallucination and ground responses in source documents."
        )
        page1.insert_textbox(rect1, p1_text, fontsize=12)

        # Page 2
        page2 = doc.new_page()
        rect2 = pymupdf.Rect(50, 50, 500, 400)
        p2_text = (
            "Section 2: Architecture\n"
            "The architecture utilizes FastAPI, PyMuPDF for extraction, Sentence Transformers for embeddings, "
            "and Qdrant as vector database. Accuracy achieved on test benchmarks is 94.8%."
        )
        page2.insert_textbox(rect2, p2_text, fontsize=12)
        
        doc.save(str(cls.pdf_path))
        doc.close()

        # 2. Create sample TXT
        cls.txt_path = cls.test_dir / "sample_notes.txt"
        cls.txt_path.write_text("Project Name: Document-RAG.\nFeatures: Fast extraction, Qdrant vector search, multi-format parsing.", encoding="utf-8")

        # 3. Create sample CSV
        cls.csv_path = cls.test_dir / "sample_sales.csv"
        df_csv = pd.DataFrame({
            "Product": ["Laptop Pro", "Wireless Mouse", "4K Monitor"],
            "Price": [1200, 25, 350],
            "Quarter": ["Q1", "Q1", "Q2"]
        })
        df_csv.to_csv(cls.csv_path, index=False)

    def test_01_pdf_extraction(self):
        blocks = DocumentExtractor.extract(self.pdf_path)
        self.assertEqual(len(blocks), 2, "PDF should have 2 page blocks")
        self.assertEqual(blocks[0]["metadata"]["page_number"], 1)
        self.assertIn("Retrieval-Augmented Generation", blocks[0]["text"])
        self.assertEqual(blocks[1]["metadata"]["page_number"], 2)
        self.assertIn("94.8%", blocks[1]["text"])
        print("[PASS] Test 1: PDF Extraction & Page metadata verified.")

    def test_02_csv_extraction(self):
        blocks = DocumentExtractor.extract(self.csv_path)
        self.assertTrue(len(blocks) >= 1)
        self.assertIn("Laptop Pro", blocks[0]["text"])
        print("[PASS] Test 2: CSV Tabular Extraction verified.")

    def test_03_chunker_and_metadata(self):
        blocks = DocumentExtractor.extract(self.pdf_path)
        chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document_blocks(blocks, doc_id="test-doc-123")
        self.assertTrue(len(chunks) >= 2)
        self.assertEqual(chunks[0]["metadata"]["doc_id"], "test-doc-123")
        self.assertTrue("Page" in chunks[0]["metadata"]["citation_label"])
        print("[PASS] Test 3: Text Chunking and Citation Label creation verified.")

    def test_04_embedding_and_qdrant_indexing(self):
        blocks = DocumentExtractor.extract(self.pdf_path)
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_document_blocks(blocks, doc_id="test-pdf-id")

        texts = [c["text"] for c in chunks]
        embeddings = embedding_manager.embed_texts(texts)
        self.assertEqual(len(embeddings), len(chunks))
        self.assertEqual(len(embeddings[0]), embedding_manager.vector_dimension)

        # Upsert into Qdrant
        count = vector_db.upsert_chunks(chunks, embeddings)
        self.assertEqual(count, len(chunks))

        # Perform Search
        query = "What accuracy benchmark was achieved in architecture?"
        q_emb = embedding_manager.embed_query(query)
        results = vector_db.search(q_emb, top_k=2)

        self.assertTrue(len(results) > 0)
        self.assertIn("94.8%", results[0]["text"])
        self.assertEqual(results[0]["page_number"], 2)
        print(f"[PASS] Test 4: Embedding & Qdrant Search verified (Top hit: {results[0]['citation_label']}, Score: {results[0]['score']:.3f}).")

    def test_05_insufficient_evidence_guardrail(self):
        # 1. Test empty search / empty query filter triggering insufficient evidence
        result_empty = rag_engine.query("What is photosynthesis?", doc_id_filter="non-existent-id")
        self.assertTrue(
            result_empty["insufficient_evidence"],
            "Empty filter search must trigger Insufficient Evidence"
        )

        # 2. Test direct prompt grounding logic
        unrelated_context = "--- [SOURCE 1: doc.pdf | Page 1] ---\nThis document only talks about Python loops."
        prompt_answer = rag_engine._generate_llm_response("What is the recipe for chocolate cake?", unrelated_context)
        self.assertTrue(
            "insufficient evidence" in prompt_answer.lower() or "api key missing" in prompt_answer.lower(),
            "Prompt grounding or missing key notification should be returned."
        )
        print("[PASS] Test 5: Insufficient Evidence & Grounding logic verified.")


if __name__ == "__main__":
    unittest.main()
