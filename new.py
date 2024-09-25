from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.schema import StrOutputParser
from langchain_community.llms import Ollama
from sqlalchemy import create_engine, text
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_teddynote.messages import stream_response
import re




input_value = "Car Battery"  # 제품명을 입력

# Ollama 모델 로드
llm = Ollama(model="ollama-bllossom:latest")

# SQL 데이터베이스 연결
db = SQLDatabase.from_uri("postgresql://sa:1@192.168.0.20:11032/Version.1")

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(
"""
{query}의 제품명, 회사명, impact_methodology에 대한 쿼리문을 작성해줘
"""
)

# 템플릿을 문자열로 변환
formatted_prompt = prompt.format(query=input_value)

# LLM과 SQL 데이터베이스 체인 생성
chain = create_sql_query_chain(llm, db, k=10) | StrOutputParser()

# 'invoke' 메서드를 사용하여 실행 (입력은 딕셔너리 형태로 전달)
answer = chain.invoke({"question": formatted_prompt})

result = re.search(r"(SELECT.*?;)", answer, re.DOTALL)

aa = result.group(1)


print(aa)

# Print the result (SQL query)



db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
engine = create_engine(db_url)

# SQL 쿼리
sql_query = aa

# 데이터베이스에 쿼리 입력 및 실행
with engine.connect() as connection:
    result = connection.execute(text(sql_query))

    # Get the column names
    columns = result.keys()

    # Print the column names
    #print("Columns:", columns)

    # Print each row with its respective column name
    for row in result:
        # Combine the column names with the row data
        a = dict(zip(columns, row))
        print(a)