import faiss
import numpy as np
import ast
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

# Step 1: Initialize the LLM (Ollama model)
llm = Ollama(model="llama3.1:latest")

# Step 2: Initialize the sentence transformer model for embeddings
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Step 3: SQL Database connection and data retrieval
db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
engine = create_engine(db_url)

# Fetch embeddings and data from the database
query_sql = "SELECT toc_id, section_title, subsection_title, body, lca_toc_embeddings FROM lca_toc WHERE lca_toc_embeddings IS NOT NULL"
with engine.connect() as connection:
    result = connection.execute(text(query_sql))
    sql_data = [dict(zip(result.keys(), row)) for row in result]

# Step 4: Parse the embeddings from string to list
embeddings = []
toc_ids = []
docs = []
for row in sql_data:
    toc_id = row['toc_id']
    section_title = row['section_title']
    subsection_title = row['subsection_title']
    body = row['body']
    embedding_str = row['lca_toc_embeddings']

    # Convert string to list of floats
    try:
        embedding = ast.literal_eval(embedding_str.replace('{', '[').replace('}', ']'))
        embeddings.append(embedding)
        toc_ids.append(toc_id)
        # Create Document objects for storage
        doc_content = f"{section_title} {subsection_title}: {body}"
        docs.append(Document(page_content=doc_content, metadata={"toc_id": toc_id}))
    except ValueError as e:
        print(f"Error parsing embedding for toc_id {toc_id}: {e}")
        continue

# Convert embeddings to numpy array
embeddings = np.array(embeddings, dtype=np.float32)

# Step 5: Build FAISS index
embedding_dimension = embeddings.shape[1]  # Dimension of your embeddings
index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance for similarity search
index.add(embeddings)  # Add embeddings to the index

# Step 6: Prepare the FAISS object in LangChain
# Use InMemoryDocstore to store the documents
docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})

# Create a mapping between FAISS index and docstore
index_to_docstore_id = {i: str(i) for i in range(len(docs))}

# Create a HuggingFaceEmbeddings object for embedding_function
embedding_function = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# Initialize FAISS vectorstore in LangChain with embedding_function
faiss_vectorstore = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_function
)

# Step 7: Create a retriever
retriever = faiss_vectorstore.as_retriever()

# Step 8: Define the LangChain RetrievalQA chain
from langchain.chains import RetrievalQA

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Step 9: Run the query
query_text = "민감도 분석에 대해서 알려줘"  # Example query
response = retrieval_qa.run(query_text)

print(response)
