import sqlite3
import ir_datasets

# إنشاء قاعدة البيانات
conn = sqlite3.connect("../DBs/trec_ct_2021_data.db")
cursor = conn.cursor()

# إنشاء جدول الوثائق
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    summary TEXT,
    description TEXT
)
""")
# إنشاء جدول الاستعلامات
cursor.execute("""
CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    text TEXT
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

dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

for doc in dataset.docs_iter():
    cursor.execute("""
        INSERT OR IGNORE INTO documents (doc_id, title, summary, description)
        VALUES (?, ?, ?, ?)
    """, (doc.doc_id, doc.title, doc.summary, doc.detailed_description))
conn.commit()

for query in dataset.queries_iter():
    cursor.execute("""
        INSERT OR IGNORE INTO queries (query_id, text)
        VALUES (?, ?)
    """, (query.query_id, query.text))
conn.commit()

for qrel in dataset.qrels_iter():
    cursor.execute("""
        INSERT OR IGNORE INTO qrels (query_id, doc_id, relevance)
        VALUES (?, ?, ?)
    """, (qrel.query_id, qrel.doc_id, qrel.relevance))
conn.commit()
