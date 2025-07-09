import numpy as np
from joblib import load, dump
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix

# Load precomputed TF-IDF sparse matrix
tfidf_matrix = load("../joblibs/trec_ct_2021/trec_ct_2021_data_tfidf_matrix.joblib")

# Load dense Word2Vec document embeddings
w2v_embeddings = load("../joblibs/trec_ct_2021/trec_ct_2021_data_doc_embeddings.joblib")

# Normalize
tfidf_norm = normalize(tfidf_matrix)
w2v_norm = normalize(w2v_embeddings)

# Weight (optional)
tfidf_weight = 0.7
w2v_weight = 0.3

from scipy.sparse import csr_matrix, hstack
w2v_sparse = csr_matrix(w2v_norm * w2v_weight)
tfidf_weighted = tfidf_norm * tfidf_weight

hybrid_doc_vectors = hstack([tfidf_weighted, w2v_sparse])

# Save the hybrid document vector matrix to disk using joblib
dump(hybrid_doc_vectors, "../joblibs/trec_ct_2021/trec_ct_2021_data_hybrid_doc_vectors.joblib")

print("Hybrid document vectors generated and saved successfully.")
