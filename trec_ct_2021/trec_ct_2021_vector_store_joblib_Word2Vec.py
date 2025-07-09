import os
import sqlite3
import ir_datasets
import numpy as np
from gensim.models import Word2Vec
from joblib import dump
from text_preprocessor import TextPreprocessor
from tqdm import tqdm

# Initialize text preprocessor and dataset
text_preprocessor = TextPreprocessor()
dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

# Connect to SQLite database
conn = sqlite3.connect("../DBs/trec_ct_2021_data.db")
cursor = conn.cursor()

# Load documents from the database
print("Loading documents from the database...")
cursor.execute("SELECT doc_id, title, summary, description FROM documents")
rows = cursor.fetchall()

doc_ids = []
doc_texts = []

for row in rows:
    doc_id, title, summary, description = row
    # Combine all available fields into one text
    full_text = f"{title or ''} {summary or ''} {description or ''}"
    doc_ids.append(doc_id)
    doc_texts.append(full_text)

# Tokenize documents using the custom preprocessor
print("Tokenizing documents...")
tokenized_docs = [text_preprocessor.custom_tokenizer(text) for text in doc_texts]

# Train Word2Vec model on tokenized documents
print("Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=400,      # Higher-dimensional embeddings
    window=7,             # Context window size
    min_count=2,          # Ignore infrequent words
    workers=os.cpu_count(),  # Use all CPU cores
    sg=1
)
w2v_model.train(tokenized_docs, total_examples=len(tokenized_docs), epochs=15)

# Create a vector representation for each document
def document_vector(tokens, model):
    """Generate the average Word2Vec vector for a list of tokens."""
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Generate document embeddings
print("Generating document embeddings...")
doc_embeddings = np.array([
    document_vector(tokens, w2v_model) for tokens in tqdm(tokenized_docs)
])

# Save the trained model and the document embeddings
print("Saving Word2Vec model and document embeddings...")
dump(w2v_model, '../joblibs/trec_ct_2021/trec_ct_2021_data_word2vec_model.joblib')
dump(doc_embeddings, '../joblibs/trec_ct_2021/trec_ct_2021_data_doc_embeddings.joblib')

print("Done. Word2Vec model and document embeddings have been saved.")
