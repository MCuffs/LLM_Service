import json
import streamlit as st
import faiss
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# Streamlit UI 설정
st.set_page_config(page_title="전과정평가 리포트 생성기", layout="centered", initial_sidebar_state="collapsed")

# 사이드바에 정보 추가
st.sidebar.title("전과정평가 리포트 생성기")
st.sidebar.markdown("""
**사용 방법:**
1. **제품명**과 **산업**을 입력하세요.
2. **리포트 생성하기** 버튼을 클릭하면 리포트가 생성됩니다.
""")

# 제목 및 설명
st.title("전과정평가 리포트 생성기 🌱")
st.write("제품 및 산업 정보를 기반으로 전과정평가 (LCA) 리포트를 생성합니다.")
st.markdown("---")

# 제품명과 산업 입력 필드 (설명 및 툴팁 추가)
input_value = st.text_input(
    "제품명을 입력하세요 (예: 브레이크 디스크)",
    placeholder="여기에 제품명을 입력하세요",
    help="생성할 전과정평가 리포트의 제품명을 입력하세요."
)

industry = st.selectbox(
    "산업을 선택하세요",
    ["자동차", "전자", "에너지", "포장", "섬유", "패션", "화학", "건설", "식음료"],
    help="해당 제품이 속한 산업을 선택하세요."
)

# 리포트 생성 버튼
if st.button('📄 리포트 생성하기'):
    # 입력 확인
    if not input_value or not industry:
        st.error("제품명과 산업을 모두 입력해 주세요.")
    else:
        # 로딩 애니메이션 표시
        with st.spinner("리포트를 생성 중입니다..."):
            # Step 1: Ollama 모델 초기화
            llm = Ollama(model="llama3.1:latest")

            # Step 2: SQL 데이터베이스 연결 및 데이터 가져오기
            db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
            engine = create_engine(db_url)

            # 사용자 입력에 따른 SQL 쿼리 실행
            query_sql = f"""
            SELECT product_name, company_name
            FROM products
            INNER JOIN company ON company.product_id = products.product_id
            WHERE products.product_name = '{input_value}';
            """

            with engine.connect() as connection:
                result = connection.execute(text(query_sql))
                sql_data = [dict(zip(result.keys(), row)) for row in result]

            if not sql_data:
                st.error(f"'{input_value}'에 대한 데이터를 찾을 수 없습니다. 다른 제품명을 입력해 주세요.")
            else:
                # SQL 데이터를 텍스트로 변환
                company_name = sql_data[0]["company_name"]
                product_name = sql_data[0]["product_name"]

                st.success(f"데이터베이스에서 조회된 데이터: {company_name} - {product_name}")

                # Step 3: FAISS 검색을 위한 데이터셋 불러오기
                with open('/Users/jeongminsu/Downloads/combined_lca_dataset.json', 'r', encoding='utf-8') as f:
                    dataset = json.load(f)

                # Step 4: 데이터셋에서 텍스트 및 산업 추출
                texts = [entry['text'] for entry in dataset]
                industries = [entry['Industry'] for entry in dataset]

                # Step 5: 텍스트와 산업을 Document 객체로 변환, 메타데이터에 'Industry' 저장
                documents = [Document(page_content=text, metadata={"Industry": industry}) for text, industry in zip(texts, industries)]

                model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

                # Step 6: HuggingFaceEmbeddings 사용
                hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

                # Step 7: FAISS 인덱스 생성
                try:
                    faiss_index = FAISS.from_documents(documents, hf_embeddings)
                except Exception as e:
                    st.error(f"FAISS 인덱스 생성 오류: {e}")
                    faiss_index = None

                if faiss_index is not None:
                    # Step 8: 검색 설정
                    retriever = faiss_index.as_retriever(
                        search_type='mmr',
                        search_kwargs={'k': 3, 'fetch_k': 10}
                    )

                    # Step 9: 'Industry' 메타데이터 기반 관련 문서 검색
                    query = industry
                    docs = retriever.invoke(query)

                    # Step 10: 검색된 문서 형식화
                    def format_docs(docs):
                        return '\n\n'.join([d.page_content for d in docs])

                    # Step 11: 프롬프트 정의
                    template = '''
                    my_data : {sql_data} , example_sentence : {context}, Industry : {ind_data}

                    Question: {question}
                    '''

                    # 템플릿에 맞게 데이터 전달
                    prompt = ChatPromptTemplate.from_template(template)

                    # 질문 내용
                    question = '이 제품에 대한 1.전과정평가 목적과 2.의도된 용도를 주어진 정보에 맞게 괄호에 대입해서 최종본만 작성해주세요.'

                    # Chain 실행
                    chain = prompt | llm | StrOutputParser()

                    # Chain 결과 실행
                    response = chain.invoke({
                        'sql_data': f"Company: {company_name}, Product: {product_name}",
                        'context': format_docs(docs),
                        'question': question,
                        'ind_data': industry
                    })

                    # 리포트 출력
                    st.markdown("### 생성된 전과정평가 리포트")
                    st.success("여기 당신의 전과정평가 리포트가 생성되었습니다:")
                    st.markdown(f"**제품명**: {product_name}")
                    st.markdown(f"**회사명**: {company_name}")
                    st.markdown(f"**산업**: {industry}")
                    st.markdown("---")
                    st.write(response)
                else:
                    st.error("FAISS 인덱스 생성에 실패했습니다.")
