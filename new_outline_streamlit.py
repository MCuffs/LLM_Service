import json
import streamlit as st
from sqlalchemy import create_engine, text
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# User input fields for product and industry
input_value = st.text_input("Enter product name (e.g., Brake Disc)", "")
industry = st.text_input("Enter industry (e.g., Automotive)", "")

# 버튼을 눌러야 SQL 쿼리와 다음 단계를 진행
if st.button('Proceed with LCA Report Generation'):
    # 입력이 비어 있는지 확인
    if not input_value or not industry:
        st.error("Please enter both product name and industry.")
    else:
        # Step 1: Initialize Ollama model (used in RAG)
        llm = Ollama(model="llama3.1:latest")

        # Step 2: SQL Database connection and data retrieval
        db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
        engine = create_engine(db_url)

        # SQL query execution (based on user input)
        query_sql = f"""
        SELECT product_name, company_name
        FROM products
        INNER JOIN company ON company.product_id = products.product_id
        WHERE products.product_name = '{input_value}';
        """

        with engine.connect() as connection:
            result = connection.execute(text(query_sql))
            sql_data = [dict(zip(result.keys(), row)) for row in result]

        st.write("SQL Data:")
        st.write(sql_data)

        # Step 3: Load the LCA dataset for FAISS retrieval
        with open('/Users/jeongminsu/Downloads/lca_dataset_custom.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Step 4: Extract the relevant text from the dataset for FAISS
        texts = [entry['text'] for entry in dataset]  # Extract 'text' holding the LCA purpose data
        industries = [entry['Industry'] for entry in dataset]  # Extract 'Industry' field

        # Step 5: Convert texts and industries into Document objects, storing 'Industry' in the metadata
        documents = [Document(page_content=text, metadata={"Industry": industry}) for text, industry in zip(texts, industries)]

        model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

        # Step 6: Use HuggingFaceEmbeddings for a lighter model
        hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Step 7: Create FAISS index using documents and the embedding model
        try:
            faiss_index = FAISS.from_documents(documents, hf_embeddings)
        except Exception as e:
            st.error(f"FAISS index creation error: {e}")
            faiss_index = None  # Handle failure case if needed

        if faiss_index is not None:
            # Step 8: Set up the retriever
            retriever = faiss_index.as_retriever(
                search_type='mmr',
                search_kwargs={'k': 3, 'fetch_k': 10}  # 결과 개수 줄이기
            )

            # Step 9: Get relevant documents based on 'Industry' metadata
            query = industry  # 메타데이터 'Industry' 필드에서 'Automotive' 검색
            docs = retriever.invoke(query)

            # Step 10: Format retrieved documents
            def format_docs(docs):
                return '\n\n'.join([d.page_content for d in docs])

            # Step 11: Define the prompt
            template = '''
            my_data : {sql_data} , example_sentence : {context}, Industry : {ind_data}

            Question: {question}
            '''

            question = '내가 준 정보를 이용해서 전과정평가 목적을 작성해줘, 제목은 "1.1 전과정평가 목적" 이야'

            prompt = ChatPromptTemplate.from_template(template)

            # Chain
            chain = prompt | llm | StrOutputParser()

            # Run the chain
            response = chain.invoke({
                'sql_data': sql_data,
                'context': format_docs(docs),
                'question': question,
                'ind_data': industry
            })

            # Display the response
            st.write(response)

        else:
            st.error("Failed to create FAISS index.")
