import sqlite3
import ir_datasets
import csv
# إنشاء قاعدة البيانات
conn = sqlite3.connect("../DBs/cord19_trec_covid_data.db")
cursor = conn.cursor()

# إنشاء جدول الوثائق
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    doi TEXT,
    date TEXT,
    abstract TEXT
)
""")
# إنشاء جدول الاستعلامات
cursor.execute("""
CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    narrative TEXT
)
""")

# إنشاء جدول qrels
cursor.execute("""
CREATE TABLE IF NOT EXISTS qrels (
    query_id TEXT,
    doc_id TEXT,
    relevance INTEGER,
    PRIMARY KEY (query_id, doc_id)
)
""")

conn.commit()

dataset = ir_datasets.load("cord19/trec-covid")

for doc in dataset.docs_iter():
    cursor.execute("""
        INSERT OR IGNORE INTO documents (doc_id, title, doi, date, abstract)
        VALUES (?, ?, ?, ?, ?)
    """, (doc.doc_id, doc.title, doc.doi, doc.date, doc.abstract))
conn.commit()

for query in dataset.queries_iter():
    cursor.execute("""
        INSERT OR IGNORE INTO queries (query_id, title, description, narrative)
        VALUES (?, ?, ?, ?)
    """, (query.query_id, query.title, query.description, query.narrative))
conn.commit()

for qrel in dataset.qrels_iter():
    cursor.execute("""
        INSERT OR IGNORE INTO qrels (query_id, doc_id, relevance)
        VALUES (?, ?, ?)
    """, (qrel.query_id, qrel.doc_id, qrel.relevance))
conn.commit()
