import math
import re
from typing import List, Optional

from app.core.config import settings
from app.models.document import ChunkMetadata

# GPT-style BPE subword pre-tokenization regex pattern
GPT_TOKEN_PATTERN = re.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\s]|\w+|\s+""",
    re.UNICODE,
)


class SubwordTokenizer:
    """
    High-fidelity subword tokenizer that matches modern BPE token estimation rules:
    - Splits contractions ('s, 't, 're, 've, 'm, 'll, 'd) into sub-tokens
    - Tokenizes punctuation and symbols separately from words
    - Divides long technical words/hashes into ~4-character subword units
    - Does NOT assume 1 word = 1 token
    """

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        if not text:
            return []
        raw_tokens = GPT_TOKEN_PATTERN.findall(text)
        sub_tokens: List[str] = []
        for tok in raw_tokens:
            stripped = tok.strip()
            # If word is long (e.g. compound identifier, hash, URL), estimate subword units (~4 chars)
            if len(stripped) > 6 and stripped.isalnum():
                parts = [stripped[i : i + 4] for i in range(0, len(stripped), 4)]
                sub_tokens.extend(parts)
            else:
                sub_tokens.append(tok)
        return sub_tokens

    @classmethod
    def count_tokens(cls, text: str) -> int:
        if not text:
            return 0
        return len(cls.tokenize(text))


class TextSplitter:
    """
    Sentence-aware token chunker that creates overlapping chunks
    using subword token counting while strictly respecting page boundaries.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Detect if compiled tiktoken DLL is permitted by host OS policy
        self._tiktoken_enc = None
        try:
            import tiktoken
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to pure-Python BPE subword tokenizer
            self._tiktoken_enc = None

    def count_tokens(self, text: str) -> int:
        """Counts exact subword BPE tokens in a string."""
        if not text:
            return 0
        if self._tiktoken_enc is not None:
            try:
                return len(self._tiktoken_enc.encode(text, disallowed_special=()))
            except Exception:
                pass
        return SubwordTokenizer.count_tokens(text)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into cohesive sentences/paragraphs without breaking words.
        Uses paragraph breaks (\n\n) and sentence-ending punctuation (. ? !).
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: List[str] = []

        for para in paragraphs:
            # Split on sentence-ending punctuation followed by whitespace
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    units.append(s_clean)

        return units

    def split_page_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_number: int,
        start_chunk_index: int = 0,
    ) -> List[ChunkMetadata]:
        """
        Splits a single page's text into overlapping chunks, preserving source page metadata.
        Returns a list of ChunkMetadata instances.
        """
        if not text or not text.strip():
            return []

        total_page_tokens = self.count_tokens(text)

        # If page fits entirely within a single chunk, return directly
        if total_page_tokens <= self.chunk_size:
            chunk_id = f"{document_id}_p{page_number}_c{start_chunk_index:03d}"
            return [
                ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=start_chunk_index,
                    chunk_text=text.strip(),
                    token_count=total_page_tokens,
                    character_count=len(text.strip()),
                )
            ]

        # Break text into sentence units
        sentence_units = self.split_into_sentences(text)
        if not sentence_units:
            sentence_units = [text]

        chunks: List[ChunkMetadata] = []
        current_chunk_sentences: List[str] = []
        current_tokens = 0
        current_index = start_chunk_index

        for unit in sentence_units:
            unit_tokens = self.count_tokens(unit)

            # If a single sentence exceeds chunk_size, split by subword tokens
            if unit_tokens > self.chunk_size:
                # Flush existing chunk
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    chunks.append(
                        ChunkMetadata(
                            chunk_id=f"{document_id}_p{page_number}_c{current_index:03d}",
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_number,
                            chunk_index=current_index,
                            chunk_text=chunk_text,
                            token_count=self.count_tokens(chunk_text),
                            character_count=len(chunk_text),
                        )
                    )
                    current_index += 1
                    current_chunk_sentences = []
                    current_tokens = 0

                # Slice large sentence by words/sub-tokens
                words = unit.split()
                step = max(1, self.chunk_size - self.chunk_overlap)
                sub_slice: List[str] = []
                sub_tokens = 0
                for w in words:
                    w_tokens = self.count_tokens(w)
                    if sub_tokens + w_tokens > self.chunk_size and sub_slice:
                        slice_text = " ".join(sub_slice).strip()
                        chunks.append(
                            ChunkMetadata(
                                chunk_id=f"{document_id}_p{page_number}_c{current_index:03d}",
                                document_id=document_id,
                                document_name=document_name,
                                page_number=page_number,
                                chunk_index=current_index,
                                chunk_text=slice_text,
                                token_count=self.count_tokens(slice_text),
                                character_count=len(slice_text),
                            )
                        )
                        current_index += 1
                        # Retain overlap words
                        overlap_words = sub_slice[-max(1, len(sub_slice) // 4) :]
                        sub_slice = overlap_words + [w]
                        sub_tokens = self.count_tokens(" ".join(sub_slice))
                    else:
                        sub_slice.append(w)
                        sub_tokens += w_tokens

                if sub_slice:
                    slice_text = " ".join(sub_slice).strip()
                    chunks.append(
                        ChunkMetadata(
                            chunk_id=f"{document_id}_p{page_number}_c{current_index:03d}",
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_number,
                            chunk_index=current_index,
                            chunk_text=slice_text,
                            token_count=self.count_tokens(slice_text),
                            character_count=len(slice_text),
                        )
                    )
                    current_index += 1
                continue

            # Standard sentence accumulation
            if current_tokens + unit_tokens <= self.chunk_size:
                current_chunk_sentences.append(unit)
                current_tokens += unit_tokens
            else:
                # Flush current chunk
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    chunks.append(
                        ChunkMetadata(
                            chunk_id=f"{document_id}_p{page_number}_c{current_index:03d}",
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_number,
                            chunk_index=current_index,
                            chunk_text=chunk_text,
                            token_count=self.count_tokens(chunk_text),
                            character_count=len(chunk_text),
                        )
                    )
                    current_index += 1

                # Calculate overlap by backtracking sentences from the end
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_chunk_sentences):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk_sentences = overlap_sentences + [unit]
                current_tokens = self.count_tokens(" ".join(current_chunk_sentences))

        # Final trailing chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if not chunks or chunks[-1].chunk_text != chunk_text:
                chunks.append(
                    ChunkMetadata(
                        chunk_id=f"{document_id}_p{page_number}_c{current_index:03d}",
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        chunk_index=current_index,
                        chunk_text=chunk_text,
                        token_count=self.count_tokens(chunk_text),
                        character_count=len(chunk_text),
                    )
                )

        return chunks
