import json
import faiss
import numpy as np
from IPython.testing.plugin.test_refs import doctest_ivars
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from sqlalchemy import create_engine, text
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings




# Step 1: Initialize Ollama model (used in RAG)
llm = Ollama(model="llama3.1:latest")

# Step 2: SQL Database connection and data retrieval
db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
engine = create_engine(db_url)

input_value = "Brake Disc"  # Example product input
industry = 'Automotive'


# SQL 쿼리 작성
query_sql = f"""
SELECT product_name, company_name
FROM products
INNER JOIN company ON company.product_id = products.product_id
WHERE products.product_name = '{input_value}';
"""

with engine.connect() as connection:
    result = connection.execute(text(query_sql))
    sql_data = [dict(zip(result.keys(), row)) for row in result]

print(sql_data)

# Step 3: Load the LCA dataset for FAISS retrieval
with open('/Users/jeongminsu/Downloads/combined_lca_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Step 4: Extract the relevant text from the dataset for FAISS
texts = [entry['text'] for entry in dataset]  # Extract 'text' holding the LCA purpose data
industries = [entry['Industry'] for entry in dataset]  # Extract 'Industry' field

# Step 3: Convert texts and industries into Document objects, storing 'Industry' in the metadata
documents = [Document(page_content=text, metadata={"Industry": industry}) for text, industry in zip(texts, industries)]

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Use HuggingFaceEmbeddings for a lighter model
hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
# Step 5: Create FAISS index using documents and the embedding model
try:
    faiss_index = FAISS.from_documents(documents, hf_embeddings)
except Exception as e:
    print(f"FAISS index creation error: {e}")
    faiss_index = None  # Handle failure case if needed

if faiss_index is not None:
    # Step 6: Set up the retriever
    retriever = faiss_index.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 10}  # 결과 개수 줄이기
    )

    # Step 7: Function to format the retrieved documents
    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    # Step 8: Get relevant documents based on 'Industry' metadata
    query = "Automotive"  # 메타데이터 'Industry' 필드에서 'Automotive' 검색
    docs = retriever.invoke(query)

else:
    print("Failed to create FAISS index.")


# Prompt
template = '''

my_data : {sql_data} , example_sentence : {context}, Industry : {ind_data}

Question: {question}

'''

query = 'Industry에 맞는 전과정 평가 목적과 의도된 용도 및 사용자를 작성할건데, product_name과 company_name을 대입 해서 작성하고 리포트 형태로 반환해줘'

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# Chain
chain = prompt | llm | StrOutputParser()

# Run
response = chain.invoke({'sql_data': (sql_data),'context': (format_docs(docs)), 'question':query, 'ind_data': (industry) })
print(response)