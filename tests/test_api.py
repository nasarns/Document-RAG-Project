import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app

client = TestClient(app)

class TestFastAPIEndpoints(unittest.TestCase):

    def test_01_health(self):
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        print("[PASS] API Test 1: Health endpoint verified.")

    def test_02_upload_pdf(self):
        sample_pdf = BASE_DIR / "data" / "samples" / "RAG_Technology_Whitepaper.pdf"
        with open(sample_pdf, "rb") as f:
            response = client.post(
                "/api/upload",
                files={"file": ("RAG_Technology_Whitepaper.pdf", f, "application/pdf")}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["total_chunks_created"] > 0)
        print(f"[PASS] API Test 2: Upload PDF endpoint verified (Created {data['total_chunks_created']} chunks).")

    def test_03_list_documents(self):
        response = client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        docs = response.json()
        self.assertTrue(len(docs) > 0)
        print(f"[PASS] API Test 3: List documents endpoint verified (Found {len(docs)} documents).")

    def test_04_query_rag(self):
        response = client.post(
            "/api/query",
            json={"question": "What is the retrieval latency and accuracy benchmark?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("retrieved_chunks", data)
        self.assertTrue(len(data["retrieved_chunks"]) > 0)
        print(f"[PASS] API Test 4: Query RAG endpoint verified.")


if __name__ == "__main__":
    unittest.main()
