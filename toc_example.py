######
######  임베딩 코드
######


import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text, update, MetaData, Table
from sqlalchemy.orm import sessionmaker


# Step 2: SQL Database connection and data retrieval
db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
engine = create_engine(db_url)

# Create a session to handle transactions
Session = sessionmaker(bind=engine)
session = Session()

# Step 3: Reflect the table schema using MetaData
metadata = MetaData()
lca_toc_table = Table('lca_toc', metadata, autoload_with=engine)

# Define the SQL query to get the specific row where toc_id = 28
query_sql = "SELECT toc_id, body FROM lca_toc WHERE toc_id = 28"

# Fetch data from the database
with engine.connect() as connection:
    result = connection.execute(text(query_sql))
    sql_data = [dict(zip(result.keys(), row)) for row in result]

# Step 4: Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Step 5: Compute embeddings for toc_id = 28 and store them in the database
for row in sql_data:
    toc_id = row['toc_id']
    body_text = row['body']

    # Skip if the body_text is None or empty
    if not body_text:
        continue

    # Compute the embedding for the body text
    embedding = model.encode(body_text)

    # Convert the embedding to a list of Python floats
    embedding_array = [float(x) for x in embedding]  # Convert numpy float32 to Python float

    # Step 6: Update the lca_toc_embeddings column in the database with the computed embedding
    update_stmt = update(lca_toc_table).where(lca_toc_table.c.toc_id == toc_id).values(
        lca_toc_embeddings=embedding_array)

    # Execute the update query
    session.execute(update_stmt)
    session.commit()

# Close the session after completing all updates
session.close()
