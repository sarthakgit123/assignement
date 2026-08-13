import json
from rag_system import documentLoader, textchunker, simpleindexer, retriever


def main():
    with open("questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    loader = documentLoader("corpus")
    docs = loader.load_document()
    chunker = textchunker()
    chunks = chunker.chunk_document(docs)

    idx = simpleindexer()
    idx.chunks = chunks
    idx.tokenized_chunks = [idx._tokenize(c["chunk_text"]) for c in chunks]

    r = retriever(idx)


    # Print concise retrieval results for first 3 answerable questions
    printed = 0
    for q in questions:
        if not q.get("answerable", False):
            continue
        print(f"{q.get('id')}: {q.get('question')}")
        res = r.retrieve(q.get("question"), top_k=5)
        for i, hit in enumerate(res, 1):
            print(f"  {i}. {hit['score']:.2f}  {hit['source']}")
        printed += 1
        if printed >= 3:
            break


if __name__ == "__main__":
    main()
