import os
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings
from app.core.embeddings import embedding_manager
from app.core.vector_db import vector_db


class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) orchestration engine.
    Handles semantic search, prompt grounding, LLM generation,
    citation attribution, and 'Insufficient Evidence' guardrails.
    """

    def __init__(self):
        self.vector_db = vector_db
        self.embeddings = embedding_manager
        self._init_llm_client()

    def _init_llm_client(self):
        """
        Initializes OpenAI-compatible client for Groq, OpenAI, Gemini, or Ollama.
        """
        provider = settings.LLM_PROVIDER.lower()
        base_url = None
        api_key = "dummy_key"

        if provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
            api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        elif provider == "openai":
            base_url = "https://api.openai.com/v1"
            api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        elif provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        elif provider == "ollama":
            base_url = settings.LLM_BASE_URL or "http://localhost:11434/v1"
            api_key = "ollama"
        elif settings.LLM_BASE_URL:
            base_url = settings.LLM_BASE_URL
            api_key = settings.OPENAI_API_KEY or "custom_key"

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "no_key_provided"
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        doc_id_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline:
        1. Embeds question
        2. Retrieves relevant chunks from Qdrant
        3. Validates confidence thresholds
        4. Synthesizes grounded answer using LLM
        5. Formats verifiable citations
        """
        k = top_k or settings.TOP_K_RESULTS
        threshold = settings.SIMILARITY_THRESHOLD

        # 1. Embed query
        query_vector = self.embeddings.embed_query(question)

        # 2. Retrieve top-K chunks from Qdrant
        retrieved_chunks = self.vector_db.search(
            query_vector=query_vector,
            top_k=k,
            doc_id_filter=doc_id_filter
        )

        # 3. Guardrail: No chunks found or low similarity
        if not retrieved_chunks or (retrieved_chunks and retrieved_chunks[0]["score"] < threshold):
            return {
                "question": question,
                "answer": "Insufficient Evidence. The uploaded document(s) do not contain information related to this question.",
                "insufficient_evidence": True,
                "citations": [],
                "retrieved_chunks": retrieved_chunks,
                "provider": settings.LLM_PROVIDER,
                "model": settings.LLM_MODEL
            }

        # 4. Format context for LLM prompt
        context_str = self._build_context_prompt(retrieved_chunks)

        # 5. Call LLM
        answer = self._generate_llm_response(question, context_str)

        # 6. Check if LLM explicitly determined lack of evidence
        is_insufficient = (
            "insufficient evidence" in answer.lower() or
            "not mentioned in the provided" in answer.lower() or
            "does not contain information" in answer.lower()
        )

        # 7. Extract citations
        citations = []
        if not is_insufficient:
            seen_citations = set()
            for chunk in retrieved_chunks:
                label = chunk.get("citation_label") or chunk.get("filename")
                if label and label not in seen_citations:
                    seen_citations.add(label)
                    citations.append({
                        "label": label,
                        "filename": chunk.get("filename"),
                        "page_number": chunk.get("page_number"),
                        "sheet_name": chunk.get("sheet_name"),
                        "section": chunk.get("section"),
                        "score": round(chunk.get("score", 0.0), 3),
                        "snippet": chunk.get("text", "")[:200] + "..."
                    })

        return {
            "question": question,
            "answer": answer,
            "insufficient_evidence": is_insufficient,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL
        }

    def _build_context_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Builds clear, tagged context blocks for the LLM.
        """
        formatted_blocks = []
        for i, chunk in enumerate(chunks, 1):
            label = chunk.get("citation_label", f"Chunk {i}")
            text = chunk.get("text", "").strip()
            score = round(chunk.get("score", 0.0), 2)
            formatted_blocks.append(f"--- [SOURCE {i}: {label} | Relevance: {score}] ---\n{text}\n")

        return "\n".join(formatted_blocks)

    def _generate_llm_response(self, question: str, context: str) -> str:
        """
        Sends the grounded question answering prompt to the LLM.
        """
        system_instruction = (
            "You are an expert Document-Based Question Answering AI assistant.\n"
            "Your task is to answer the user's question using ONLY the provided document sources.\n\n"
            "STRICT RULES:\n"
            "1. Ground every single claim in the provided context.\n"
            "2. If the provided context DOES NOT contain the answer, reply EXACTLY with:\n"
            "   'Insufficient Evidence: The uploaded document(s) do not contain sufficient information to answer this question.'\n"
            "3. Do NOT make assumptions, guess, or use any outside pre-trained knowledge.\n"
            "4. Mention the source document name and page/section where applicable (e.g. '[Source: document.pdf, Page 2]').\n"
            "5. Keep the answer clear, structured, and factual."
        )

        user_content = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nANSWER:"

        # Check if API key is provided
        api_key = self.client.api_key
        if not api_key or api_key == "no_key_provided" or api_key == "dummy_key":
            if settings.LLM_PROVIDER in ["groq", "openai", "gemini"]:
                return (
                    f"⚠️ **API Key Missing**: Please set your `{settings.LLM_PROVIDER.upper()}_API_KEY` in the `.env` file.\n\n"
                    f"**Retrieved Context Summary:**\nFound {len(context.split('--- [SOURCE')) - 1} relevant chunks for your query."
                )

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error communicating with LLM ({settings.LLM_PROVIDER}): {str(e)}"


# Global instance
rag_engine = RAGEngine()
