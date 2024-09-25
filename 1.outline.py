from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.schema import StrOutputParser
from langchain_community.llms import Ollama
from sqlalchemy import create_engine, text
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_teddynote.messages import stream_response
import re

def clean_query(query):
    # HTML-like 태그 및 불필요한 텍스트 제거
    cleaned_query = re.sub(r'```sql', '', query)
    cleaned_query = re.sub(r'```', '', cleaned_query)
    # "SELECT" 또는 SQL 명령어 이후로 시작되는 구문만 추출
    match = re.search(r"(SELECT).*?(;)", cleaned_query, re.IGNORECASE)
    if match:
        cleaned_query = match.group(0)  # 매칭된 SQL 쿼리만 반환
    else:
        raise ValueError("올바른 SQL 쿼리를 추출하지 못했습니다.")

    return cleaned_query


input_value = "Brake Disc"  # 제품명을 입력






# Ollama 모델 로드
llm = Ollama(model="ollama-bllossom:latest")

# SQL 데이터베이스 연결
db = SQLDatabase.from_uri("postgresql://sa:1@192.168.0.20:11032/Version.1")

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(
    """
    {query}가 제품명인데, 오직 이 제품에 대한 제품명, 회사(company)에 대한 모든 정보, 전화번호 , 영향평가(impact_assessment)만 볼 수 있는 쿼리문을 작성해줘
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

prompt = ChatPromptTemplate.from_template(
"""
{b}

이 정보를 이용해서 제품, 회사, 담당자 정보를 분류하고, 결과를 기반해서 아래 모든 문장 괄호의 단어를 대체해서 결과를 반환해줘

1. 개요
1.1 전과정평가 목적
전과정평가는 [회사명]에서 생산하는 [제품]의 전과정 (원료물질 취득, 가공, 운송, 제품의 생산 및 유통)에 걸친 에너지, 자원 및 온실가스 배출량을 조사하고 평가하여 고객과의 커뮤니케이션을 위한 탄소발자국 데이터를 확보하기 위한 것이다.

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
영향평가 방법은 국제적으로 인정받고 있는 IPCC 2013 방법론을 적용한다. 해당 방법론을 적용하기 위해서는 이를 지원할 수 있는 데이터베이스의 적용이 필요하며, Ecoinvent 3.9.1 데 이터베이스를 적용하였다.

1.4 적용 표준
[제품명] 전과정평가는 [영향평가방법]에 따라 수행되었다.

1.5 연락 정보

회사 : [회사명]
대표자 : [대표자명]
소재지 : [위치]
이메일 : [이메일]

""")

# LLM 모델 로드 및 처리
llm2 = Ollama(model="llama3.1:latest")

chain = prompt | llm2 | StrOutputParser()

answer = chain.stream({"b":a})

stream_response(answer)






#llm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
# 결과 출력
#answer = llm.generate([formatted_prompt])

# LLMResult에서 생성된 텍스트에 접근
#generated_text = answer.generations[0][0].text

# 결과 출력
#print(generated_text)

