import json
from rag_system import answer


def run_tests():
    qs = json.load(open("questions.json", encoding="utf-8"))
    results = []
    for q in qs:
        res = answer(q["question"])
        results.append({
            "id": q["id"],
            "supported": res.get("supported", False),
            "citations": res.get("citations", []),
            "answer_excerpt": res.get("answer", "")[:200].replace("\n", " "),
        })

    # print concise per-question results
    for r in results:
        print(f"{r['id']}: supported={r['supported']} citations={r['citations']}\n  {r['answer_excerpt']}\n")


if __name__ == "__main__":
    run_tests()
