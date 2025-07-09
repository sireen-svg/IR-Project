import os
import sqlite3
from joblib import dump, load
from gensim import corpora
from gensim.models import LdaModel
from tqdm import tqdm

from text_preprocessor import TextPreprocessor

# Initialize preprocessor
text_preprocessor = TextPreprocessor()

# Load documents from SQLite
conn = sqlite3.connect("../DBs/trec_ct_2021_data.db")
cursor = conn.cursor()
cursor.execute("SELECT doc_id, title, summary, description FROM documents")
rows = cursor.fetchall()

doc_ids = []
doc_texts = []

for row in rows:
    doc_id, title, summary, description = row
    full_text = f"{title or ''} {summary or ''} {description or ''}"
    doc_ids.append(doc_id)
    doc_texts.append(full_text)

print("Tokenize documents")
# Try loading pre-tokenized documents
tokenized_docs_path = "../joblibs/trec_ct_2021/trec_ct_2021_tokenized_docs.joblib"
if os.path.exists(tokenized_docs_path):
    print("Loading tokenized documents from disk...")
    tokenized_docs = load(tokenized_docs_path)
else:
    print("Tokenizing documents...")
    tokenized_docs = [text_preprocessor.custom_tokenizer(text) for text in tqdm(doc_texts)]
    dump(tokenized_docs, tokenized_docs_path)
    print(f"Saved tokenized documents to {tokenized_docs_path}")

print("Create dictionary and corpus for LDA")
# Create dictionary and corpus for LDA
dictionary = corpora.Dictionary(tokenized_docs)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

print("Train LDA model")
# Train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=20,
    passes=10,
    chunksize=2000,
    random_state=42
)

print("Assign dominant topic to each document")
# Assign dominant topic to each document
doc_topics = []
for doc_bow in tqdm(corpus, desc="Assigning topics"):
    topic_probs = lda_model.get_document_topics(doc_bow)
    if topic_probs:
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
    else:
        dominant_topic = -1
    doc_topics.append(dominant_topic)

# Save everything
dump(doc_ids, '../joblibs/trec_ct_2021/trec_ct_2021_data_doc_ids.joblib')
dump(doc_texts, '../joblibs/trec_ct_2021/trec_ct_2021_data_raw_texts.joblib')
dump(doc_topics, '../joblibs/trec_ct_2021/trec_ct_2021_data_doc_topics.joblib')
lda_model.save("../joblibs/trec_ct_2021/lda_model_trec_ct_2021.model")
dictionary.save("../joblibs/trec_ct_2021/lda_dictionary_trec_ct_2021.dict")

print("Topic detection files saved successfully.")
