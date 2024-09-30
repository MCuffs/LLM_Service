from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.schema import StrOutputParser
from langchain_community.llms import Ollama
from sqlalchemy import create_engine, text
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_teddynote.messages import stream_response
from sqlalchemy import create_engine, inspect
from database import schema_info

import re








input_value = "Brake Disc"  # 제품명을 입력

# Ollama 모델 로드
llm = Ollama(model="llama3.1:latest")

# SQL 데이터베이스 연결
db = SQLDatabase.from_uri("postgresql://sa:1@192.168.0.20:11032/Version.1")

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(
    """
    Now, You are a SQL professinal

    This database schema information is as follows:
    {schema_info}

    '{query}' is the product name, please create a query to see all the information for this, please write it in error-free SQL!
    """
)

# 템플릿을 문자열로 변환
formatted_prompt = prompt.format(query=input_value, schema_info={schema_info})

# LLM과 SQL 데이터베이스 체인 생성
chain = create_sql_query_chain(llm, db, k=10) | StrOutputParser()

# 'invoke' 메서드를 사용하여 실행 (입력은 딕셔너리 형태로 전달)
answer = chain.invoke({"question": formatted_prompt})

result = re.search(r"(SELECT.*?;)", answer, re.DOTALL)

##print(result)

aa = result.group(1)

##print(aa)

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
    # print("Columns:", columns)

    # Print each row with its respective column name
    for row in result:
        # Combine the column names with the row data
        a = dict(zip(columns, row))
        print(a)

prompt = ChatPromptTemplate.from_template(
    """
    
    문서를 작성하기 위한 정보는 다음과 같아: {b}
    
    다음 예시를 보고, 위 정보들을 기반으로 문서를 작성해줘
    
    
    2. 전과정평가 수행 범위
    
    목적 달성을 위한 일반적인 수행 범위를 아래의 세부 항에서 설명하고 있다. 수행 범위에는 평가 대상 제품 시스템, 기능 단위 및 기준 흐름, 시스템 경계, 할당 절차 및 Cut-off 기준, 영향평가 방법 등이 포함된다.
    
    2.1 제품 시스템 2.1.1 대상 제품
    
    대상 제품은 [제품명] 이다.
    
    2.1.2 대상제품 특성
    
    [제품명] 제품 시스템은 원료물질의 취득 및 가공, 원료물질 운송 및 [제품명] 생산으로 구
    분하였다. [제품명] 생산 공정은 원료혼합, 압출성형, 절단, 포장의 세부공정으로 구성되어있다.
    
    [제품명] 생산 공정은 [기업명]의 현장 데이터를 수집하여 적용하고, 원료의 생산 공정에는 LCI(Life cycle inventory) 데이터베이스를 적용하였다. LCI 데이터베이스는 Ecoinvent DB 3.9.14 을 이용하였다.
    
    2.3 시스템 경계
    [제품명] 전과정평가 시스템 경계는 cradle to gate 이며, 원료물질 취득. 가공, 원료물질 운송 및 [제품명] 생산을 포함한다. 사용과 폐기 단계에 대해서는 해당 제품이 다른 제품의 원료물질로 사용되는 생산재이므로 완제품이 되었을 때의 사용 및 폐기 단계에 관한 시나리오를 결정할 수 없어 시스템 경계에서 제외하였다. (국내 환경성적표지, International EPD PCR 등에서 생산재의 시스템 경계를 cradle to gate로 정하고 있다.)
    
    데이터 수집 범위
    [제품명] 시스템을 Foreground 시스템과 Background 시스템으로 구분하였다.
    - Foreground 시스템에는 [제품명] 생산 공정이 포함되고, 현장 데이터를 수집하였다. [제품 명] 생산 데이터는 사업장에서 관리하고 있는 고지서, 현장관리 문서 등을 기반으로 수집 하였다. 다만 원료 생산에 관하여는 원료 공급업체들에 생산 관련 데이터를 요청하였으나 제공받지 못하여 Background 시스템으로 적용시켰다.
    - Background 시스템에는 Foreground 시스템의 모든 투입물과 산출물의 생산 또는 처리 및 운송이 포함된다. Background 시스템에는 해당하는 LCI 데이터베이스를 수집하여 적용하 였다. [반제품]에 대해서는 [반제품] 생산에 투입되는 [반제품원료1] 에 사용되는 에너지에 대해 LCI 데이터베이스를 이용하여 DB를 생성하여 적용하였다.
    
    제외(Cut off) 기준
    시스템 경계를 설정하기 위해 다음과 같이 질량 기준의 cut-off 기준을 적용하였다.
    - 질량 : 각 단위공정의 모든 원자재 및 부자재 투입량을 내림차순으로 정렬하고, 누적질량 을 계산하여 99%까지의 원자재 및 부자재를 시스템 경계에 포함하였다. 단, 제외되는 개 별 원자재 및 부자재의 질량이 전체 투입물 질량의 1%를 초과해서는 안된다.
    투입물 및 산출물 데이터 경계
    
    [제품명] 전과정평가에서 다음 사항과 같이 제품 생산과 직접적인 관련이 없거나, 내구성을 가진 생산설비 등은 시스템 경계에서 제외하였다.
    - 자본재 및 기반 시설
    - 종업원의 활동
    - 모든 행정적인 활동
    - 원자재 및 부자재의 누적질량기여도 기준으로 99%를 벗어나는 물질 - 내부운송
    - 포장재 제외 (고객의 요청에 따라)
    시스템 경계에 포함되는 원자재 및 부자재의 운송과 폐기물의 운송이 시스템 경계에 포함되었다.
    
    단위공정 결정
    [제품명] 생산 공정을 원료혼합, 압출성형, 절단, 포장 공정으로 구분하였으며, 연구의 목적을 달성하기 위해 1개의 단위공정으로 구분하였다.
    
    2.4 할당
    2.4.1 다중 산출물의 할당
    다중 산출물의 할당은 일반적으로 ISO 14044의 4.3.4.2항을 따른다. 제품 시스템이 하나 이상의 제품을 생산하는 경우, 환경 부하는 각각의 제품에 배분되어야 한다. 가능하다면 할당은 회피되어야 하며, ISO 14044는 시스템 확장을 통한 할당의 회피를 권고한다. 할당을 회피할 수 없는 경우, 할당은 제품 사이의 물리적 인과관계(예, 에너지 함량, 중량 등)에 따라 수행되어야 한다.
    [기업명]의 사업장은 [제품명]과 [제품2] 및 [제품3] 등의 제품을 생산하고 있으며, 생산공정은 원료혼합, 압출성형, 절단, 포장으로 구분된다.
    원자재([원자재명] 및 [반제품]) 투입량은 제품별 및 공정별로 데이터가 관리되고 있으나, 유틸리티, 에너지 등은 전체 사업장의 데이터를 관리하고 있다. 따라서, 제품별([제품명] , [제품2] 및 [제품3])로 데이터의 할당이 필요하다.
    할당이 필요한 데이터에 대하여 다음의 할당방법이 적용되었으며, 가능한 사업장의 배분 기준을 준용하였다.
    - 원자재 : 사업장 관리 데이터 적용
    - 전기 : 전체 사업장 전기사용량 관리, 각 제품 생산량 기준으로 할당
    - 공업용수 : 전체 사업장 용수사용량 관리, 용수가 투입되는 제품들의 생산량 기준으로 할당
    - 폐기물 : 전체 사업장 기준으로 폐기물 발생량 관리, 폐기물별 발생원을 확인하고 해당 제품의 생산량 기준으로 할당
    - [폐기물 1] 에 대해서는 [제품명], [제품2], [제품3]의 생산량 기준으로 할당
    - [폐기물 2] 는 [제품3]과는 관계 없으므로 [제품명]과 [제품2]의 생산량 기준으로 할당 - LUC : 각 제품 생산량 기준으로 할당
    
    
    2.4.2 End of life 할당
    Open loop 재활용 시스템에 cut-off 할당방법 중 재활 공정을 원료 생산 공정으로 포함하는 방법을 적용한다. 재활용된 재료를 사용할 경우 재활용 공정을 시스템 경계에 포함하고, 공정 에서 발생되는 재활용 폐기물에 대해서는 재활용 업체까지의 운송을 시스템 경계에 포함하고 재활용 공정을 제외하는 할당방법이다.
    [제품명]는 재활용 재료를 사용하지 않고 재활용 폐기물이 발생되지 않으므로 End of life 할 당을 적용하지 않았다.
    
    2.5 시간적 범위
    [제품명]의 생산 데이터는 [기간]까지 총 1년의 데이터를 대상으로 한다. Background 시스템의 LCI 데이터는 최근 5년 이내 데이터 적용을 원칙으로 하였으며, Ecoinvent 3.9.1(유효기간 2022년) 데이터를 적용하였다.
    
    2.6 기술적 범위
    [제품명] 생산에는 최근 1년간의 현장 데이터를 적용하여 기술적 대표성을 확보하였다. Background 시스템 데이터에는 상응하는 기술의 데이터를 적용하여 기술적 대표성을 확보하였다.
    
    2.7 지리적 범위
    [제품명] 생산에는 현장 데이터를 적용하여 지리적 대표성을 확보하였다. 해외 수입 원료는 해당 지역 또는 해당 지역이 포함된 데이터 사용을 원칙으로 하였다. 다만, 해당 지역 데이터 가 없거나 데이터 품질을 만족하지 못하는 경우 유사 데이터를 적용하였다.
    
    2.8 영향평가 방법 및 영향범주
    연구의 목적에 따른 환경발자국 산정을 위한 영향평가 방법으로 고객([고객사명])의 요청에 따라 [영향평가 방법론] 을 적용하였다.
    
    
    2.9 데이터 품질 요건
    전과정평가에 사용되는 데이터는 목적 및 범위에 부합하는 정밀성, 완전성, 일관성 및 대표 성을 가져야 하며, 주어진 예산과 시간을 반영해야 한다. [제품명] 전과정평가에 적용되는 데이 터 품질 요건은 다음과 같다.
    - 정밀성 : 측정된 1차 데이터가 가장 높은 정밀성을 가지며, 다음으로 계산된 데이터, 추정 된 데이터 순이다. [제품명] 생산 공정의 측정 또는 계산된 1차 데이터 적용을 목표로 하 였다.
    - 완전성 : 완전성은 단위공정별 투입물과 산출물의 완전성과 단위공정들 자체의 완전성으로 판단된다. 이러한 관점에서 관련된 모든 투입물 및 산출물 데이터 수집을 목표로 하였다. 또한 제외(cut off) 기준 99%를 적용하여 완전성을 확보하였다.
    - 일관성 : “전과정평가 대상의 모든 물질 및 절차에 적용된 방법론의 일관성에 대한 정성 적인 평가”로 정의된다. [제품명] 전과정평가는 모델링 방법론의 선택, 데이터 출처, 배출 량 계산, 또는 기타 분석 방법에 일관된 방법론 적용을 목표로 하였다.
    - 재현성: 재현성은 독립적인 수행자가 본 보고서에 작성된 방법론 및 데이터 정보를 이용하 여 결과를 재현할 수 있는 정도를 나타낸다. [제품명] 전과정평가는 독립적인 수행자가 대 략적인 결과를 재현할 수 있도록 충분한 투명성 제공을 목표로 하였다.
    - 대표성: 대표성은 전과정평가의 목적 및 범위에서 정한 시간적 범위, 기술적 범위 및 지리 적 범위를 반영하는 정도를 나타낸다. [제품명] 전과정평가는 [제품명] 생산을 대표할 수 있는 1차 데이터를 사용하고, 원료물질 생산에도 가장 대표성이 있는 LCI 데이터 사용을 목표로 하였다.
    
    2.10 소프트웨어 및 데이터베이스
    [제품명] 전과정평가는 [소프트웨어 회사]에서 개발된 [소프트웨어 프로그램]을 사용하여 수행하였다. 원료물질 및 에너지 생산에 대한 LCI 데이터베이스는 Ecoinvent 3.9.1에서 제공하는 데이터를 적용하였다.
    
    
    """)

# LLM 모델 로드 및 처리

chain = prompt | llm | StrOutputParser()

answer = chain.stream({"b": a})

stream_response(answer)

# llm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
# 결과 출력
# answer = llm.generate([formatted_prompt])

# LLMResult에서 생성된 텍스트에 접근
# generated_text = answer.generations[0][0].text

# 결과 출력
# print(generated_text)

