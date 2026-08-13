import os
import glob
from typing import List, Dict, Optional


class documentLoader:
   

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path

    def load_document(self, extensions: Optional[List[str]] = None) -> List[Dict]:
        if extensions is None:
            extensions = [".md", ".txt"]

        documents: List[Dict] = []
        for ext in extensions:
            pattern = os.path.join(self.corpus_path, f"*{ext}")
            for filepath in glob.glob(pattern):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({"content": content, "source": filepath})

        return documents


class textchunker:
   

    def chunk_document(self, documents: List[Dict], sep: str = "\n\n") -> List[Dict]:
        chunks: List[Dict] = []
        for doc in documents:
            raw_chunks = doc["content"].split(sep)
            for chunk in raw_chunks:
                text = chunk.strip()
                if not text:
                    continue
                chunks.append({"chunk_text": text, "source": doc.get("source")})
        return chunks


class simpleindexer:
    def __init__(self):
        self.bm25_model = None
        self.chunks: List[Dict] = []

    def _tokenize(self, text: str):
        return text.lower().split()

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        tokenized_chunks = [self._tokenize(c["chunk_text"]) for c in self.chunks]
       
        self.tokenized_chunks = tokenized_chunks

        try:
            from rank_bm25 import BM25Okapi

            self.bm25_model = BM25Okapi(tokenized_chunks)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "rank_bm25 is not installed. Install with: pip install rank-bm25"
            )


if __name__ == "__main__":
   
    loader = documentLoader(corpus_path="corpus")
    docs = loader.load_document()

    chunker = textchunker()
    chunks = chunker.chunk_document(docs)

    indexer = simpleindexer()
    try:
        indexer.build_index(chunks)
    except ModuleNotFoundError as e:
        # Build index silently when run as a script
        _init_index("corpus")


class retriever:
    def __init__(self,indexer:simpleindexer):
        self.indexer = indexer

    def _tokenize(self, text: str):
        return text.lower().split()

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:

        if top_k <= 0:
            raise ValueError("top_k must be >= 1")


        q_tokens = self._tokenize(question)

        if getattr(self.indexer, "bm25_model", None) is not None:
            try:
                scores = self.indexer.bm25_model.get_scores(q_tokens)
                # pair scores with indices
                scored = list(enumerate(scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                results = []
                for idx, score in scored[:top_k]:
                    chunk = self.indexer.chunks[idx]
                    results.append({"chunk_text": chunk["chunk_text"], "source": chunk.get("source"), "score": float(score)})
                return results
            except Exception:
                # fall through to fallback scorer
                pass

        # fallback: simple token-overlap scoring using tokenized_chunks if available
        tokenized = getattr(self.indexer, "tokenized_chunks", None)
        if tokenized is None:
            tokenized = [self._tokenize(c["chunk_text"]) for c in self.indexer.chunks]

        q_set = set(q_tokens)
        scored = []
        for i, tokens in enumerate(tokenized):
            common = q_set.intersection(tokens)
            score = len(common)
            scored.append((i, score))

        # Also compute substring matches (helps match DIM -> dimensional)
        q_subs = [t for t in q_tokens if len(t) >= 3]
        substr_scores = []
        for i, chunk in enumerate(self.indexer.chunks):
            text = chunk["chunk_text"].lower()
            subcount = 0
            for sub in q_subs:
                if sub in text:
                    subcount += 1
            substr_scores.append((i, subcount))

        # merge scores (token overlap + substring matches)
        merged = {}
        for i, s in scored:
            merged[i] = merged.get(i, 0) + s
        for i, s in substr_scores:
            merged[i] = merged.get(i, 0) + (s * 2)  # give substring matches higher weight

        merged_list = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in merged_list[:top_k]:
            chunk = self.indexer.chunks[idx]
            results.append({"chunk_text": chunk["chunk_text"], "source": chunk.get("source"), "score": float(score)})

        return results


# Global index cache for convenience
_GLOBAL_INDEX = None


def _init_index(corpus_dir: str = "corpus"):
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is not None:
        return _GLOBAL_INDEX

    loader = documentLoader(corpus_dir)
    docs = loader.load_document()
    chunker = textchunker()
    chunks = chunker.chunk_document(docs)

    idx = simpleindexer()
    # keep chunks and tokenized chunks for fallback
    idx.chunks = chunks
    idx.tokenized_chunks = [idx._tokenize(c["chunk_text"]) for c in chunks]
    # try to build BM25 index but continue if dependency missing
    try:
        idx.build_index(chunks)
    except ModuleNotFoundError:
        pass

    _GLOBAL_INDEX = {"indexer": idx}
    return _GLOBAL_INDEX


def answer(question: str) -> Dict:
  
    index_bundle = _init_index()
    idx = index_bundle["indexer"]
    r = retriever(idx)
    top_chunks = r.retrieve(question, top_k=5)

    supported = any(h.get("score", 0) > 0 for h in top_chunks)
    # gather unique citation basenames in order
    citations = []
    for h in top_chunks:
        src = h.get("source")
        if src:
            name = os.path.basename(src)
            if name not in citations:
                citations.append(name)

    answer_text = ""
    if supported:
        try:
            import importlib

            lm_mod = importlib.import_module("LLM")
            OpenRouterLLM = getattr(lm_mod, "OpenRouterLLM", None)
            if OpenRouterLLM is not None:
                llm = OpenRouterLLM()
                answer_text = llm.generate_with_context(question, top_chunks, top_k=5)
            else:
                raise RuntimeError("OpenRouterLLM class not found in LLM module")
        except Exception:

            texts = [h.get("chunk_text", "") for h in top_chunks][:3]
            answer_text = "\n\n".join(texts)
    else:
        answer_text = "I don't know. The answer is not contained in the provided documents."

    return {"answer": answer_text, "citations": citations, "supported": bool(supported)}
