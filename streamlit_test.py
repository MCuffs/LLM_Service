import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.schema import StrOutputParser
from langchain_community.llms import Ollama
from sqlalchemy import create_engine, text
from langchain_core.prompts import ChatPromptTemplate
import re

# Streamlit 제목 및 설명
st.title("LLM SQL Query Generator and LCA Report Creator")
st.write("This app generates SQL queries using an LLM and creates an LCA report based on database results.")

# 사용자 입력 받기
input_value = st.text_input("Enter the product name:", "Brake Disc")

# Ollama 모델 로드
llm = Ollama(model="ollama-bllossom:latest")

# SQL 데이터베이스 연결
db = SQLDatabase.from_uri("postgresql://sa:1@192.168.0.20:11032/Version.1")

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(
    """
    {query}가 제품명인데, 이 제품의 제품명(product_name), 회사명(company_name), 영향평가방법(impact_methodology)을 볼 수 있는 쿼리문을 작성해줘
    """
)

# 템플릿을 문자열로 변환
formatted_prompt = prompt.format(query=input_value)

# LLM과 SQL 데이터베이스 체인 생성
chain = create_sql_query_chain(llm, db, k=10) | StrOutputParser()

# SQL 쿼리 생성 및 실행
if st.button("Generate SQL Query"):
    try:
        answer = chain.invoke({"question": formatted_prompt})

        # 정규 표현식을 이용해 쿼리 추출
        result = re.search(r"(SELECT.*?;)", answer, re.DOTALL)
        if result:
            aa = result.group(1)
            st.session_state['sql_query'] = aa  # SQL 쿼리를 session_state에 저장
            st.write("Generated SQL Query:", aa)
        else:
            st.error("SQL query could not be extracted.")
    except Exception as e:
        st.error(f"Error during SQL generation: {e}")

# SQL 쿼리 실행 및 결과 출력
if st.button("Run SQL Query") and 'sql_query' in st.session_state:
    db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
    engine = create_engine(db_url)

    try:
        with engine.connect() as connection:
            sql_query = st.session_state['sql_query']  # session_state에서 쿼리를 가져옴
            result = connection.execute(text(sql_query))
            columns = result.keys()

            st.write("Query Result:")

            # 각 열 및 행을 출력
            rows = []
            for row in result:
                a = dict(zip(columns, row))
                rows.append(a)

            st.session_state['rows'] = rows  # rows를 session_state에 저장
            st.write(rows)
    except Exception as e:
        st.error(f"Error during SQL execution: {e}")

# LCA 보고서 생성을 위한 프롬프트
if st.button("Generate Final LCA Report"):
    if 'rows' in st.session_state:  # rows가 session_state에 저장되어 있는지 확인
        try:
            row_data = st.session_state['rows'][0]  # 첫 번째 결과 사용

            # 프롬프트 템플릿
            final_prompt = ChatPromptTemplate.from_template(
                """
                {b}

                이 정보를 이용해서 제품명과 회사명과 영향평가방법을 추론하고, 아래 문장의 괄호에 대입해서 최종 수정된 문서를 작성해서 보여줘

                1. 개요
                
                1.1 전과정평가 목적
                
                전과정평가는 [회사명]에서 생산하는 [제품명]의 전과정 (원료물질 취득, 가공, 운송, 제품의 생산 및 유통)에 걸친 에너지, 자원 및 온실가스 배출량을 조사하고 평가하여 고객과의 커뮤니케이션을 위한 탄소발자국 데이터를 확보하기 위한 것이다.

                1.2 의도된 용도 및 사용자
                
                [제품명] 전과정평가의 용도 및 사용자는 다음과 같다.
                - 고객([고객사명])의 요청에 따라 [제품명] 제품의 탄소발자국 커뮤니케이션 자료로 활용할 수 있다.
                - [제품명] 제품의 탄소발자국 저감을 위한 기초 자료로 활용할 수 있다. - 독립적인 검증기관의 검증에 사용할 수 있다.

                1.3 전과정평가 수행 방안
                
                고객과 커뮤니케이션을 위하여 전과정 단계를 세분화하여 결과를 도출하며, 전과정 단계는
                다음과 같이 구분하였다.
                - 원료물질 취득 및 생산 (Raw material production and supply)
                - 원료물질 운송 (Raw material transportation)
                - [제품명] 생산 (Manufacturing)
                · 전기 사용 (Manufacturing energy – electricity)
                · 기타 (Other manufacturing ancillary materials, waste processing, others)
                영향평가 방법은 국제적으로 인정받고 있는 IPCC 2013 방법론을 적용한다. 해당 방법론을 적용하기 위해서는 이를 지원할 수 있는 데이터베이스의 적용이 필요하며, Ecoinvent 3.9.1 데이터베이스를 적용하였다.

                1.4 적용 표준
                
                [제품명] 전과정평가는 [영향평가방법]에 따라 수행되었다.
                """
            )

            # LLM 모델 로드 및 처리
            llm2 = Ollama(model="llama3.1:latest")

            # LLM에 프롬프트 및 SQL 결과를 넘겨 최종 문서 생성
            chain = final_prompt | llm2 | StrOutputParser()
            final_answer = chain.invoke({"b": row_data})

            # Streamlit을 사용하여 최종 문서 출력
            st.write(final_answer)
        except Exception as e:
            st.error(f"Error during final document generation: {e}")
    else:
        st.error("No rows data available. Please run the SQL query first.")
