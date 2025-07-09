import sqlite3
import ir_datasets
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from text_preprocessor import TextPreprocessor

# Initialize text preprocessing tools
text_preprocessor = TextPreprocessor()

# Load TREC Clinical Trials 2021 dataset (queries and relevance judgments)
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

# Combine title, summary, and description into a single text for each document
for row in rows:
    doc_id, title, doi, date, abstract = row
    full_text = f"{title or ''} {abstract or ''}"
    doc_ids.append(doc_id)
    doc_texts.append(full_text)

print(f"Successfully loaded {len(doc_texts)} documents.")

# Create an inverted index
from collections import defaultdict
inverted_index = defaultdict(set)
for doc_id, text in zip(doc_ids, doc_texts):
    terms = set(text_preprocessor.custom_tokenizer(text))
    for term in terms:
        inverted_index[term].add(doc_id)
dump(inverted_index, '../joblibs/cord19_trec_covid/cord19_trec_covid_data_inverted_index.joblib')

# Apply TF-IDF transformation to the documents
print("Generating TF-IDF matrix for documents...")

vectorizer = TfidfVectorizer(
    tokenizer=text_preprocessor.custom_tokenizer,  # use custom tokenizer
    preprocessor=None,                             # no additional preprocessor
    lowercase=False                                # do not lowercase automatically
)

tfidf_matrix = vectorizer.fit_transform(doc_texts)

# Save TF-IDF matrix and vectorizer to disk
dump(tfidf_matrix, '../joblibs/cord19_trec_covid/cord19_trec_covid_data_tfidf_matrix.joblib')
dump(vectorizer, '../joblibs/cord19_trec_covid/cord19_trec_covid_data_tfidf_vectorizer.joblib')

print("TF-IDF matrix and vectorizer saved successfully.")
