import os
import glob
import re
from typing import List, Dict, Optional

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}


def _tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    # Extract lowercased alphanumeric tokens
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    clean = []
    for t in tokens:
        if remove_stopwords and t in ENGLISH_STOP_WORDS:
            continue
        # drop single-letter non-numeric tokens like 's' or 't'
        if len(t) == 1 and not t.isdigit():
            continue
        clean.append(t)
    return clean


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
    def chunk_document(self, documents: List[Dict], min_chunk_len: int = 400) -> List[Dict]:
        chunks: List[Dict] = []
        for doc in documents:
            source_path = doc.get("source", "")
            filename = os.path.basename(source_path)
            content = doc["content"].strip()
            
            # Short documents (under 1200 chars) are kept intact as full-context chunks
            if len(content) <= 1200:
                formatted_text = f"Document: {filename}\n{content}"
                chunks.append({
                    "chunk_text": formatted_text,
                    "raw_text": content,
                    "source": source_path
                })
            else:
                # Merge small paragraphs into coherent chunks of at least min_chunk_len
                raw_paragraphs = content.split("\n\n")
                curr_para = []
                curr_len = 0
                for p in raw_paragraphs:
                    p_text = p.strip()
                    if not p_text:
                        continue
                    curr_para.append(p_text)
                    curr_len += len(p_text)
                    if curr_len >= min_chunk_len:
                        combined = "\n\n".join(curr_para)
                        chunks.append({
                            "chunk_text": f"Document: {filename}\n{combined}",
                            "raw_text": combined,
                            "source": source_path
                        })
                        curr_para = []
                        curr_len = 0
                if curr_para:
                    combined = "\n\n".join(curr_para)
                    chunks.append({
                        "chunk_text": f"Document: {filename}\n{combined}",
                        "raw_text": combined,
                        "source": source_path
                    })
        return chunks


class simpleindexer:
    def __init__(self):
        self.bm25_model = None
        self.tfidf_model = None
        self.tfidf_matrix = None
        self.chunks: List[Dict] = []
        self.tokenized_chunks: List[List[str]] = []

    def _tokenize(self, text: str):
        return _tokenize_text(text, remove_stopwords=True)

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        self.tokenized_chunks = [self._tokenize(c["chunk_text"]) for c in self.chunks]

        # Try sklearn TF-IDF first for robust retrieval
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus_texts = [c["chunk_text"] for c in chunks]
            self.tfidf_model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.tfidf_model.fit_transform(corpus_texts)
        except Exception:
            self.tfidf_model = None
            self.tfidf_matrix = None

        # Try rank_bm25
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_model = BM25Okapi(self.tokenized_chunks)
        except ModuleNotFoundError:
            self.bm25_model = None


class retriever:
    def __init__(self, indexer: simpleindexer):
        self.indexer = indexer

    def _tokenize(self, text: str):
        return _tokenize_text(text, remove_stopwords=True)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")

        q_tokens = self._tokenize(question)

        # 1. Try TF-IDF vectorizer if available
        if getattr(self.indexer, "tfidf_model", None) is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                q_vec = self.indexer.tfidf_model.transform([question])
                sims = cosine_similarity(q_vec, self.indexer.tfidf_matrix).flatten()
                
                # Bonus for exact key token matches
                boosted_sims = []
                for i, base_sim in enumerate(sims):
                    chunk_lower = self.indexer.chunks[i]["chunk_text"].lower()
                    bonus = 0.0
                    for token in q_tokens:
                        if len(token) >= 3 and token in chunk_lower:
                            bonus += 0.05
                    boosted_sims.append((i, float(base_sim + bonus)))

                boosted_sims.sort(key=lambda x: x[1], reverse=True)
                results = []
                for idx, score in boosted_sims[:top_k]:
                    chunk = self.indexer.chunks[idx]
                    results.append({
                        "chunk_text": chunk["chunk_text"],
                        "raw_text": chunk.get("raw_text", chunk["chunk_text"]),
                        "source": chunk.get("source"),
                        "score": score
                    })
                return results
            except Exception:
                pass

        # 2. Try BM25 if available
        if getattr(self.indexer, "bm25_model", None) is not None:
            try:
                scores = self.indexer.bm25_model.get_scores(q_tokens)
                scored = list(enumerate(scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                results = []
                for idx, score in scored[:top_k]:
                    chunk = self.indexer.chunks[idx]
                    results.append({
                        "chunk_text": chunk["chunk_text"],
                        "raw_text": chunk.get("raw_text", chunk["chunk_text"]),
                        "source": chunk.get("source"),
                        "score": float(score)
                    })
                return results
            except Exception:
                pass

        # 3. Fallback: Token overlap with stop-word filtering & sub-term matching
        scored = []
        q_set = set(q_tokens)
        for i, tokens in enumerate(self.indexer.tokenized_chunks):
            common = q_set.intersection(tokens)
            score = len(common) * 2.0
            
            # Substring matching for terms
            chunk_text_lower = self.indexer.chunks[i]["chunk_text"].lower()
            for qt in q_tokens:
                if len(qt) >= 3 and qt in chunk_text_lower:
                    score += 1.0
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[:top_k]:
            chunk = self.indexer.chunks[idx]
            results.append({
                "chunk_text": chunk["chunk_text"],
                "raw_text": chunk.get("raw_text", chunk["chunk_text"]),
                "source": chunk.get("source"),
                "score": float(score)
            })
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
    idx.build_index(chunks)

    _GLOBAL_INDEX = {"indexer": idx}
    return _GLOBAL_INDEX


def answer(question: str) -> Dict:
    index_bundle = _init_index()
    idx = index_bundle["indexer"]
    r = retriever(idx)
    top_chunks = r.retrieve(question, top_k=5)

    # Key non-stopword tokens in question
    q_non_stopwords = _tokenize_text(question, remove_stopwords=True)

    # Generic terms that appear broadly in logistics handbooks
    generic_domain_words = {"meridian", "company", "policy", "policies", "shipment", "shipments", "handbook"}
    specific_query_terms = [t for t in q_non_stopwords if t not in generic_domain_words and len(t) >= 3]
    if not specific_query_terms:
        specific_query_terms = [t for t in q_non_stopwords if len(t) >= 2]

    best_hit = top_chunks[0] if top_chunks else None
    max_score = best_hit["score"] if best_hit else 0.0

    # Verification: check if specific query terms exist in top retrieved chunk
    top_text_lower = best_hit["chunk_text"].lower() if best_hit else ""
    matching_terms = [t for t in specific_query_terms if t in top_text_lower]

    # Refusal logic: must have matching specific query terms and positive score
    is_answerable = len(matching_terms) > 0 and max_score > 0.01

    if not is_answerable:
        return {
            "answer": "I don't know. The handbook does not contain the answer to this question.",
            "citations": [],
            "supported": False
        }

    # Filter top_chunks to relevant ones
    relevant_chunks = [
        c for c in top_chunks
        if any(t in c["chunk_text"].lower() for t in specific_query_terms)
    ]
    if not relevant_chunks:
        relevant_chunks = top_chunks[:3]

    # Clean citations
    citations = []
    for h in relevant_chunks:
        src = h.get("source")
        if src:
            name = os.path.basename(src)
            if name not in citations:
                citations.append(name)

    answer_text = ""
    try:
        import importlib
        lm_mod = importlib.import_module("LLM")
        OpenRouterLLM = getattr(lm_mod, "OpenRouterLLM", None)
        if OpenRouterLLM is not None:
            llm = OpenRouterLLM()
            answer_text = llm.generate_with_context(question, relevant_chunks, top_k=5)
        else:
            raise RuntimeError("OpenRouterLLM class not found in LLM module")
    except Exception:
        # Fallback when LLM API call fails/unreachable: extract best raw text
        best_raw = relevant_chunks[0].get("raw_text", relevant_chunks[0].get("chunk_text", ""))
        answer_text = f"Based on {citations[0] if citations else 'handbook'}:\n{best_raw}"

    return {
        "answer": answer_text,
        "citations": citations,
        "supported": True
    }


if __name__ == "__main__":
    _init_index("corpus")
