from typing import List, Dict, Any

from src.init.logger import logger
from src.init.config import ChunkConfig
from .dialog_chunk_processor import DialogChunkProcessor

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkProcessor:
    def __init__(self, config: ChunkConfig, data_type: str = "LongmemEval"):
        self.config = config
        self.data_type = data_type
        self.dialog_processor = DialogChunkProcessor(data_type)
        logger.info(f"Chunk processor initialized for {data_type} dataset")

    def chunk(self, text: str):
        if not text:
            return []

        chunks = []
        chunk_size = self.config.chunk_token_size
        overlap = self.config.chunk_overlap_token_size

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        split_texts = splitter.split_text(text)

        current_pos = 0
        for chunk_text in split_texts:
            start_idx = text.find(chunk_text, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(chunk_text)

            chunks.append({
                "text": chunk_text,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "token_count": len(chunk_text.split()),
            })

            current_pos = max(start_idx, end_idx - overlap)

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

    def chunk_by_dialog_turns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk dialog data by conversation turns."""
        return self.dialog_processor.create_dialog_chunks(session_data)

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        dialogue_input = getattr(self.config, "dialogue_input", False)

        if dialogue_input:
            chunks = self.chunk_by_dialog_turns(doc)
            for chunk in chunks:
                chunk["doc_id"] = doc.get("doc_id", 0)
            return chunks

        else:
            title = doc.get('title', '')
            content = doc.get('content', '')
            full_text = f"{title}\n\n{content}" if title else content
            chunks = self.chunk(full_text)
            for chunk in chunks:
                chunk.update({
                    'doc_id': doc.get('doc_id', 0),
                    'title': title})
            return chunks

    def process_corpus(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        global_chunk_id = 0

        for doc in corpus:
            doc_chunks = self.chunk_document(doc)

            for chunk in doc_chunks:
                chunk["chunk_id"] = global_chunk_id
                global_chunk_id += 1

            all_chunks.extend(doc_chunks)

        logger.info(f"Processed {len(corpus)} documents into {len(all_chunks)} chunks with unique chunk IDs")
        return all_chunks