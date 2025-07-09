import sqlite3
import ir_datasets
from joblib import load
from sklearn.preprocessing import normalize

import text_preprocessor
import re
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary

class Cord19Retrieval:
    def __init__(self, db_path="DBs/cord19_trec_covid_data.db", tfidf_matrix_path="joblibs/cord19_trec_covid/cord19_trec_covid_data_tfidf_matrix.joblib", vectorizer_path="joblibs/cord19_trec_covid/cord19_trec_covid_data_tfidf_vectorizer.joblib", inverted_index_path="joblibs/cord19_trec_covid/cord19_trec_covid_data_inverted_index.joblib"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.dataset = ir_datasets.load("cord19/trec-covid")
        self.vectorizer = load(vectorizer_path)
        self.tfidf_matrix = load(tfidf_matrix_path)
        self.word2vec_model = load("joblibs/cord19_trec_covid/cord19_trec_covid_data_word2vec_model.joblib")
        self.doc_embeddings = load("joblibs/cord19_trec_covid/cord19_trec_covid_data_doc_embeddings.joblib")
        self.lda_model = LdaModel.load("joblibs/cord19_trec_covid/lda_model_cord19_trec_covid.model")
        self.lda_dict = Dictionary.load("joblibs/cord19_trec_covid/lda_dictionary_cord19_trec_covid.dict")
        self.doc_topics = load("joblibs/cord19_trec_covid/cord19_trec_covid_data_doc_topics.joblib")
        self.doc_ids, self.doc_texts = self._load_documents()
        self.inverted_index = load(inverted_index_path)
        self.text_preprocessor = text_preprocessor.TextPreprocessor()
        self._print_test_queries()
        self.hybrid_doc_vectors = load("joblibs/cord19_trec_covid/cord19_trec_covid_data_hybrid_doc_vectors.joblib")

    def clean_text(self, text):
        # يزيل رموز التحكم غير القابلة للطباعة
        return re.sub(r'[\x00-\x1F\x7F]', '', text)

    def _print_test_queries(self, file_path="cord19_test_queries.txt"):
        limit = 10
        with open(file_path, "w", encoding="utf-8") as f:
            for index, query in enumerate(self.dataset.queries_iter()):
                if index >= limit:
                    break
                query_id = self.clean_text(query.query_id)
                query_text = self.clean_text(f"{query.title} {query.description}")
                f.write(f"{query_id}\n{query_text}\n{'=' * 40}\n")

    def _load_documents(self):
        self.cursor.execute("SELECT doc_id, title, doi, date, abstract FROM documents")
        rows = self.cursor.fetchall()
        doc_ids, doc_texts = [], []
        for doc_id, title, doi, date, abstract in rows:
            full_text = f"{title or ''} {abstract or ''}"
            doc_ids.append(doc_id)
            doc_texts.append(full_text)
        return doc_ids, doc_texts

    def _get_query_topic(self, query: str) -> int:
        tokens = self.text_preprocessor.custom_tokenizer(query)
        bow = self.lda_dict.doc2bow(tokens)
        topic_probs = self.lda_model.get_document_topics(bow)
        if not topic_probs:
            return -1
        return max(topic_probs, key=lambda x: x[1])[0]

    def _embed_query_word2vec(self, query_tokens):
        word_vectors = [self.word2vec_model.wv[word] for word in query_tokens if word in self.word2vec_model.wv]
        if not word_vectors:
            return np.zeros(self.word2vec_model.vector_size)  # Return zero vector if no words found
        return np.mean(word_vectors, axis=0)

    def search(self, query: str, top_n=10):
        from sklearn.metrics.pairwise import cosine_similarity

        processed_query = self.text_preprocessor.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        query_terms = set(processed_query.split())

        candidate_doc_ids = set()
        for term in query_terms:
            candidate_doc_ids.update(self.inverted_index.get(term, set()))
        candidate_indices = [i for i, doc_id in enumerate(self.doc_ids) if doc_id in candidate_doc_ids]
        candidate_vectors = self.tfidf_matrix[candidate_indices]

        similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        top_local_indices = similarities.argsort()[::-1][:top_n]

        retrieved_docs = []
        for i in top_local_indices:
            if similarities[i] > 0:
                doc_id = self.doc_ids[candidate_indices[i]]
                self.cursor.execute(
                    "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
                )
                row = self.cursor.fetchone()
                if row:
                    doc_id, title, doi, date, abstract = row
                    retrieved_docs.append({
                        "doc_id": doc_id,
                        "title": title or "",
                        "summary": doi or "",
                        "description": abstract or "",
                        "similarity": similarities[i]
                    })

        return retrieved_docs

    def evaluate(self, query_id: str, retrieved_doc_ids: list) -> dict:
        relevant_doc_ids = {qrel.doc_id for qrel in self.dataset.qrels_iter()
                            if qrel.query_id == query_id and qrel.relevance > 0}

        matched = [doc_id for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids]

        # Recall
        recall = len(matched) / len(relevant_doc_ids) if relevant_doc_ids else 0

        # Average Precision
        avg_precision = 0.0
        num_relevant = 0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                num_relevant += 1
                avg_precision += num_relevant / (i + 1)
        avg_precision = avg_precision / len(relevant_doc_ids) if relevant_doc_ids else 0

        # MRR
        reciprocal_rank = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                reciprocal_rank = 1 / (i + 1)
                break

        return {
            "matched_count": len(matched),
            "precision": len(matched) / len(retrieved_doc_ids) if retrieved_doc_ids else 0,
            "recall": recall,
            "average_precision": avg_precision,
            "reciprocal_rank": reciprocal_rank,
            "matched_ids": matched
        }

    def search_word2vec(self, query: str, top_n=10):
        from sklearn.metrics.pairwise import cosine_similarity

        query_tokens = self.text_preprocessor.custom_tokenizer(query)
        query_vector = self._embed_query_word2vec(query_tokens)

        if np.all(query_vector == 0):
            print("⚠️ Query terms not found in Word2Vec vocabulary.")
            return []

        similarities = cosine_similarity([query_vector], self.doc_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_n]

        retrieved_docs = []
        for i in top_indices:
            if similarities[i] > 0:
                doc_id = self.doc_ids[i]
                self.cursor.execute(
                    "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
                )
                row = self.cursor.fetchone()
                if row:
                    doc_id, title, doi, date, abstract = row
                    retrieved_docs.append({
                        "doc_id": doc_id,
                        "title": title or "",
                        "summary": doi or "",
                        "description": abstract or "",
                        "similarity": similarities[i]
                    })

        return retrieved_docs

    def search_hybrid(self, query: str, top_n=10, use_topic_filter=False):
        from sklearn.metrics.pairwise import cosine_similarity

        # Preprocess query
        processed_query = self.text_preprocessor.preprocess_text(query)
        query_tokens = self.text_preprocessor.custom_tokenizer(query)

        if use_topic_filter:
            query_topic = self._get_query_topic(query)
            candidate_indices = [
                i for i, topic in enumerate(self.doc_topics) if topic == query_topic
            ]
            if not candidate_indices:
                return []
        else:
            candidate_indices = list(range(len(self.doc_ids)))

        if not candidate_indices:
            return []

        tfidf_query_vector = self.vectorizer.transform([processed_query]).toarray()
        w2v_query_vector = self._embed_query_word2vec(query_tokens).reshape(1, -1)

        # Combine vectors
        hybrid_query_vector = np.hstack([tfidf_query_vector, w2v_query_vector])
        # hybrid_query_vector = normalize(hybrid_query_vector)

        # Similarity with all hybrid document vectors
        similarities = cosine_similarity(hybrid_query_vector, self.hybrid_doc_vectors[candidate_indices])[0]
        top_indices = similarities.argsort()[::-1][:top_n]

        retrieved_docs = []
        for i in top_indices:
            if similarities[i] > 0:
                doc_id = self.doc_ids[candidate_indices[i]]
                self.cursor.execute(
                    "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
                )
                row = self.cursor.fetchone()
                if row:
                    doc_id, title, doi, date, abstract = row
                    retrieved_docs.append({
                        "doc_id": doc_id,
                        "title": title or "",
                        "summary": doi or "",
                        "description": abstract or "",
                        "similarity": similarities[i]
                    })

        return retrieved_docs
