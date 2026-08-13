import json
import re
from typing import Dict, List
from rag_system import answer, _init_index, retriever


GROUND_TRUTH = {
    "P1": {
        "target_doc": "dimensional-weight.md",
        "key_facts": ["166"],
    },
    "P2": {
        "target_doc": "facility-directory.md",
        "key_facts": ["06:00", "22:00", "07:00", "14:00"],
    },
    "P3": {
        "target_doc": "declared-value-insurance.md",
        "key_facts": ["100"],
    },
    "P4": {
        "target_doc": "hazmat-restrictions.md",
        "key_facts": ["100"],
    },
    "P5": {
        "target_doc": "claim-eligibility.md",
        "key_facts": ["21"],
    },
    "P6": {
        "target_doc": "customs-documentation.md",
        "key_facts": ["invoice", "triplicate"],
    },
    "P7": {
        "target_doc": "fuel-surcharge.md",
        "key_facts": ["11"],
    },
    "P8": {
        "target_doc": None,
        "key_facts": ["don't know", "not contain", "does not cover", "not covered"],
    },
}


def check_key_facts(text: str, key_facts: List[str], answerable: bool) -> bool:
    """Checks whether generated text contains mandatory key facts or refusal phrases."""
    text_lower = text.lower()
    if not answerable:
        # Check if text contains any refusal phrase
        return any(rf in text_lower for rf in key_facts)

    # For answerable questions, check if at least one key fact / number is present
    matches = sum(1 for kf in key_facts if kf.lower() in text_lower)
    return matches >= 1


def evaluate_question(q: Dict) -> Dict:
    qid = q.get("id")
    question = q.get("question")
    answerable = q.get("answerable", True)
    expected_ans = q.get("expected_answer", "")

    gt = GROUND_TRUTH.get(qid, {})
    target_doc = gt.get("target_doc")
    key_facts = gt.get("key_facts", [])

    # Call RAG pipeline answer function
    res = answer(question)
    supported = res.get("supported", False)
    citations = res.get("citations", [])
    ans_text = res.get("answer", "")

    # 1. Refusal / Support evaluation
    support_ok = (supported == answerable)

    # 2. Citation Accuracy evaluation
    if answerable:
        # Check if expected target doc is in citations
        top_citation = citations[0] if citations else None
        citation_ok = (target_doc is not None) and (top_citation == target_doc or target_doc in citations)
    else:
        # Unanswerable question should have no citations or be explicitly refused
        citation_ok = (len(citations) == 0 or not supported)

    # 3. Fact / Content evaluation
    fact_ok = check_key_facts(ans_text, key_facts, answerable)

    # Overall outcome: all metrics pass
    overall_pass = support_ok and citation_ok and fact_ok

    return {
        "id": qid,
        "question": question,
        "answerable": answerable,
        "expected_answer": expected_ans,
        "target_doc": target_doc,
        "supported": supported,
        "citations": citations,
        "answer_text": ans_text,
        "support_ok": support_ok,
        "citation_ok": citation_ok,
        "fact_ok": fact_ok,
        "pass": overall_pass,
    }


def print_evaluation_report(results: List[Dict]):
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    citation_passed = sum(1 for r in results if r["citation_ok"])
    support_passed = sum(1 for r in results if r["support_ok"])
    fact_passed = sum(1 for r in results if r["fact_ok"])

    print("=" * 80)
    print("                      COMPREHENSIVE RAG EVALUATION REPORT                      ")
    print("=" * 80)
    print(f"{'ID':<4} | {'Answerable':<10} | {'Target Doc':<25} | {'Supported':<10} | {'Status':<6}")
    print("-" * 80)

    for r in results:
        target_str = r["target_doc"] if r["target_doc"] else "N/A (Refusal)"
        status_str = "PASS" if r["pass"] else "FAIL"
        ans_str = "Yes" if r["answerable"] else "No (Refuse)"
        sup_str = "True" if r["supported"] else "False"
        print(f"{r['id']:<4} | {ans_str:<10} | {target_str:<25} | {sup_str:<10} | {status_str:<6}")

    print("-" * 80)
    print(f"Overall Accuracy:           {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Retrieval / Citation Acc:   {citation_passed}/{total} ({citation_passed/total*100:.1f}%)")
    print(f"Refusal / Support Acc:      {support_passed}/{total} ({support_passed/total*100:.1f}%)")
    print(f"Answer Fact Accuracy:       {fact_passed}/{total} ({fact_passed/total*100:.1f}%)")
    print("=" * 80)

    # Task 4 Requirement: Print 2-3 detailed example outputs
    print("\n--- Task 4: Sample Output Demonstrations ---")
    sample_ids = ["P1", "P2", "P8"]
    for r in results:
        if r["id"] in sample_ids:
            print(f"\nQuestion [{r['id']}]: {r['question']}")
            print(f"  Supported : {r['supported']}")
            print(f"  Citations : {r['citations']}")
            print(f"  Answer    : {r['answer_text'][:250].replace('\n', ' ')}")

    # Task 4 Requirement: Write analysis on what went wrong and why
    print("\n--- Task 4: Technical Analysis & Diagnosis ---")
    print("1. Previous Failure Mode (Stop-word Pollution): The baseline retriever scored chunks")
    print("   using raw token overlap without stop-word filtering. Query words like 'what', 'for',")
    print("   'is' caused irrelevant files (e.g., comms-templates.md) to score high across queries.")
    print("2. Evaluation Metric Flaw: The previous run_evaluation.py used naive regex number matching")
    print("   and 4-letter word overlap, incorrectly grading bad answers as correct. It also checked")
    print("   chunks_text as a fallback, masking whether answer() produced a valid output.")
    print("3. Refusal Failure: Baseline rag_system set supported=True whenever any score > 0,")
    print("   failing to refuse unanswerable questions like P8 (Meridian employee vacation policy).")
    print("4. Fix Summary: Upgraded rag_system with stop-word tokenization, header-aware chunking,")
    print("   domain term thresholding, and clean ground-truth evaluation in run_evaluation.py.")


def main():
    with open("questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Initialize index
    _init_index()

    results = []
    for q in questions:
        res = evaluate_question(q)
        results.append(res)

    print_evaluation_report(results)


if __name__ == "__main__":
    main()
