"""Working Context Manager - enforces token limits by trimming."""

from typing import Dict, List, Optional

import tiktoken

from src.agents.schemas import RetrievedChunk


class WorkingContextManager:
    """Trims conversation history and chunks to fit within a token budget."""

    def __init__(
        self,
        max_prompt_tokens: int = 6000,
        max_history_turns: int = 3,
        max_chunks: int = 3,
        model_name: str = "gpt-4o",
    ):
        self.max_prompt_tokens = max_prompt_tokens
        self.max_history_turns = max_history_turns
        self.max_chunks = max_chunks

        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def prepare_context(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Build a trimmed context string for the LLM."""
        # 1. Keep only top max_chunks by score
        sorted_chunks = sorted(
            retrieved_chunks, key=lambda x: x.score, reverse=True
        )[: self.max_chunks]

        # 2. Keep only last N turns
        history_text = ""
        if conversation_history:
            recent = conversation_history[-self.max_history_turns:]
            history_text = "\n".join(
                f"User: {t['query']}\nAssistant: {t['response']}" for t in recent
            )

        def build() -> str:
            chunks_text = "\n\n".join(
                f"[Source: {c.doc_id} | Score: {c.score:.2f}]\n{c.text}"
                for c in sorted_chunks
            )
            parts = []
            if history_text:
                parts.append(f"### Conversation History:\n{history_text}")
            parts.append(f"### Retrieved Documents:\n{chunks_text}")
            parts.append(f"### User Question:\n{query}")
            return "\n\n".join(parts)

        full_context = build()
        current_tokens = self.count_tokens(full_context)

        # 3. Drop lowest-scoring chunks until within budget
        while current_tokens > self.max_prompt_tokens and len(sorted_chunks) > 1:
            dropped = sorted_chunks.pop()
            print(f"[ContextManager] Dropped chunk {dropped.doc_id} (score {dropped.score:.2f})")
            full_context = build()
            current_tokens = self.count_tokens(full_context)

        # 4. Drop oldest history turns if still over budget
        while current_tokens > self.max_prompt_tokens and history_text:
            lines = history_text.split("\n")
            if len(lines) > 2:
                history_text = "\n".join(lines[2:])
                full_context = build()
                current_tokens = self.count_tokens(full_context)
            else:
                history_text = ""
                full_context = build()
                current_tokens = self.count_tokens(full_context)

        print(f"[ContextManager] Final tokens: {current_tokens} / {self.max_prompt_tokens}")
        return full_context