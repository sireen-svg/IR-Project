from tqdm import tqdm

from clinical_trials_retrieval import ClinicalTrialsRetrieval
from cord19_retrieval import Cord19Retrieval

def evaluate_dataset(retrieval_system, mode="tfidf", top_n=10, use_topic_filter=False):
    ap_list = []
    recall_list = []
    rr_list = []

    for query in tqdm(retrieval_system.dataset.queries_iter()):
        query_id = query.query_id
        query_text = getattr(query, "text", None) or f"{query.title} {query.description}"

        if mode == "tfidf":
            results = retrieval_system.search(query_text, top_n)
        elif mode == "word2vec":
            results = retrieval_system.search_word2vec(query_text, top_n)
        elif mode == "hybrid":
            results = retrieval_system.search_hybrid(query_text, top_n, use_topic_filter)
        else:
            raise ValueError("Invalid mode selected.")

        doc_ids = [doc["doc_id"] for doc in results]
        eval_result = retrieval_system.evaluate(query_id, doc_ids)

        ap_list.append(eval_result["average_precision"])
        recall_list.append(eval_result["recall"])
        rr_list.append(eval_result["reciprocal_rank"])

    map_score = sum(ap_list) / len(ap_list)
    mean_recall = sum(recall_list) / len(recall_list)
    mrr = sum(rr_list) / len(rr_list)

    print(f"\n=== Evaluation Results ({retrieval_system.__class__.__name__}) ===")
    print(f"Mode: {mode}")
    print(f"Queries evaluated: {len(ap_list)}")
    print(f"MAP: {map_score:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"MRR: {mrr:.4f}")

# Run for both datasets
clinical = ClinicalTrialsRetrieval()
cord19 = Cord19Retrieval()

evaluate_dataset(clinical, mode="tfidf", top_n=10)
evaluate_dataset(cord19, mode="tfidf", top_n=10)

# You can run it with mode="word2vec" or "hybrid" as well
