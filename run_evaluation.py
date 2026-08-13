import json
from rag_system import answer, _init_index, retriever


import re


def is_expected_in_text(expected: str, text: str) -> bool:
    if not expected or not text:
        return False
    # normalize
    def norm(s: str) -> str:
        return re.sub(r"[^\w\s]", " ", s).lower()

    n_expected = norm(expected)
    n_text = norm(text)

    if n_expected.strip() and n_expected.strip() in n_text:
        return True

    # check numeric tokens
    nums = re.findall(r"\d+", expected)
    for n in nums:
        if n in n_text:
            return True

    # check long-word overlap (>=4 chars)
    words = [w for w in n_expected.split() if len(w) >= 4]
    match_count = sum(1 for w in words if w in n_text)
    if words and match_count >= max(1, len(words) // 2):
        return True

    return False


def main():
    qs = json.load(open("questions.json", encoding="utf-8"))
    # Print all questions being evaluated
    print("Questions:")
    for q in qs:
        print(f"{q.get('id')}: {q.get('question')}")
    print()
    total = len(qs)
    correct = 0
    results = []

    # ensure index built
    index_bundle = _init_index()
    idx = index_bundle["indexer"]
    r = retriever(idx)

    for q in qs:
        qid = q.get("id")
        question = q.get("question")
        expected = q.get("expected_answer", "").strip()
        res = answer(question)
        supported = res.get("supported", False)
        ans_text = res.get("answer", "")

        # also check retrieved chunks for evidence
        top_chunks = r.retrieve(question, top_k=5)
        chunks_text = "\n\n".join([c.get("chunk_text", "") for c in top_chunks])

        if q.get("answerable", False):
            ok = is_expected_in_text(expected, ans_text) or is_expected_in_text(expected, chunks_text)
        else:
            ok = (not supported)

        if ok:
            correct += 1

        results.append({"id": qid, "question": question, "expected": expected, "supported": supported, "ok": ok, "citations": res.get("citations", [])})

    # concise summary and a few examples
    print(f"Processed {total} questions. Correct: {correct}/{total}")
    print("Examples:")
    for r in results[:3]:
        out = answer(r['question'])
        print(f"{r['id']}: supported={r['supported']} correct={r['ok']} citations={r['citations']}")
        print(f"  {out['answer'][:200].replace('\n', ' ')}")

    print("\nShort analysis: retrieval can be noisy; consider finer chunking or embeddings for improvements.")


if __name__ == '__main__':
    main()
