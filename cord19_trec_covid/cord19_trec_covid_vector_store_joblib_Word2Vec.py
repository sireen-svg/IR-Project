import os
import sqlite3
import ir_datasets
import numpy as np
from gensim.models import Word2Vec
from joblib import dump, load
from text_preprocessor import TextPreprocessor
from tqdm import tqdm

# Initialize text preprocessor and dataset
text_preprocessor = TextPreprocessor()
dataset = ir_datasets.load("cord19/trec-covid")

# Connect to SQLite database
conn = sqlite3.connect("../DBs/cord19_trec_covid_data.db")
cursor = conn.cursor()

# Load documents from the database
print("Loading documents from the database...")
cursor.execute("SELECT doc_id, title, doi, date, abstract FROM documents")
rows = cursor.fetchall()

doc_ids = []
doc_texts = []

for row in rows:
    doc_id, title, doi, date, abstract = row
    full_text = f"{title or ''} {abstract or ''}"
    doc_ids.append(doc_id)
    doc_texts.append(full_text)

# Tokenize documents using the custom preprocessor
print("Tokenize documents")
# Try loading pre-tokenized documents
tokenized_docs_path = "../joblibs/cord19_trec_covid/cord19_trec_covid_tokenized_docs.joblib"
if os.path.exists(tokenized_docs_path):
    print("Loading tokenized documents from disk...")
    tokenized_docs = load(tokenized_docs_path)
else:
    print("Tokenizing documents...")
    tokenized_docs = [text_preprocessor.custom_tokenizer(text) for text in tqdm(doc_texts)]
    dump(tokenized_docs, tokenized_docs_path)
    print(f"Saved tokenized documents to {tokenized_docs_path}")

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
dump(w2v_model, '../joblibs/cord19_trec_covid/cord19_trec_covid_data_word2vec_model.joblib')
dump(doc_embeddings, '../joblibs/cord19_trec_covid/cord19_trec_covid_data_doc_embeddings.joblib')

print("Done. Word2Vec model and document embeddings have been saved.")
