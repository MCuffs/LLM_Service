import faiss
import numpy as np
import ast
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from langchain.chains import RetrievalQA
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "newnew"
os.environ["LANGCHAIN_API_KEY"] = "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"


# Step 1: Initialize the LLM (Ollama model)
llm = Ollama(model="llama3.1:latest")

# Step 2: Initialize the sentence transformer model for embeddings
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Streamlit title
st.title("LCA Document Retrieval with LangChain")

# Step 3: SQL Database connection and data retrieval
db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
engine = create_engine(db_url)

# Step 4: Fetch embeddings and data from the database
query_sql = "SELECT toc_id, section_title, subsection_title, body, lca_toc_embeddings FROM lca_toc WHERE lca_toc_embeddings IS NOT NULL"
with engine.connect() as connection:
    result = connection.execute(text(query_sql))
    sql_data = [dict(zip(result.keys(), row)) for row in result]

# Step 5: Parse the embeddings from string to list
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
        st.error(f"Error parsing embedding for toc_id {toc_id}: {e}")
        continue

# Convert embeddings to numpy array
embeddings = np.array(embeddings, dtype=np.float32)

# Step 6: Build FAISS index
embedding_dimension = embeddings.shape[1]  # Dimension of your embeddings
index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance for similarity search
index.add(embeddings)  # Add embeddings to the index

# Step 7: Prepare the FAISS object in LangChain
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

# Step 8: Create a retriever
retriever = faiss_vectorstore.as_retriever()

# Step 9: Define the LangChain RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Step 10: Streamlit input for query
query_text = st.text_input("Enter your query:", value=" ")

# Define your custom prompt
prompt_template = """
너는 이제부터 전과정평가(LCA) 레포트 작성 전문가야. 다음 규칙을 따라서 레포트를 작성해줘
1. 레포트 요약이 아닌 레포트 전체를 작성해야한다.
2. 무조건 한국어로 말해야한다.
3. 주어진 정보를 사용해야한다.

Question: {query}

Answer:
"""

# Run the query when the user hits the 'Run' button
if st.button('Run Query'):
    with st.spinner('Running query...'):
        # Step 11: Customize the input with the prompt
        prompt = prompt_template.format(query=query_text)

        # Step 12: Run the query and display the response
        response = retrieval_qa.run(prompt)
        st.success("Query completed!")
        st.write(response)
