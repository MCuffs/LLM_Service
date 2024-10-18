import os
import psycopg2
import re
import uuid
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM  # 변경된 import
import logging
import string
from datetime import datetime
import markdown  # Markdown 변환 라이브러리

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

# 환경 변수 로드
load_dotenv()

# 임시 저장소 (실제 운영 환경에서는 데이터베이스나 캐시 사용 권장)
reports = {}

# PostgreSQL 데이터베이스 연결 함수 (필요 시)
def get_all_lca_sections():
    try:
        # PostgreSQL 연결 설정
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        cursor = conn.cursor()
        
        # 모든 섹션 가져오기 (순서가 중요한 경우 ORDER BY 절 추가)
        query = """
        SELECT subsection_title, body FROM lca_toc
        WHERE toc_id > 28
        ORDER BY toc_id ASC;  -- toc_id 또는 다른 정렬 기준에 따라 수정
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # 연결 종료
        cursor.close()
        conn.close()

        if results:
            return results  # [(subsection_title, body), ...]
        else:
            return []
    except Exception as e:
        logging.error(f"Database connection or query failed: {e}")
        return []

# 새로운 함수: 수치 데이터 가져오기
def get_numerical_data():
    try:
        # PostgreSQL 연결 설정
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        cursor = conn.cursor()
        
        # 주어진 SQL 쿼리 실행
        query = """
        SELECT * FROM manual_project mp
        INNER JOIN manual_process mp2 ON mp2.manual_project_id = mp.id
        INNER JOIN manual_impact mi ON mi.manual_project_id = mp.impact_mtd_info_id
        INNER JOIN manual_lca ml ON ml.manual_project_id = mp.id
        INNER JOIN manual_product_calc mpc ON mpc.manual_project_id = mp.id
        INNER JOIN manual_unit_process mup ON mup.manual_project_id = mp.id
        LIMIT 1;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        # 연결 종료
        cursor.close()
        conn.close()

        if results:
            # 딕셔너리 형태로 변환
            data = [dict(zip(columns, row)) for row in results]
            return data  # [{'column1': value1, 'column2': value2, ...}]
        else:
            return []
    except Exception as e:
        logging.error(f"Database connection or query failed: {e}")
        return []

# 새로운 함수: 수치 데이터 포맷팅
def format_db_data(db_data):
    if not db_data:
        return "데이터가 없습니다."

    # 첫 번째 행만 사용 (LIMIT 1)
    data = db_data[0]

    # 테이블 형식으로 변환 (Markdown), None 값은 "데이터가 없습니다"로 대체
    formatted_data = "| 필드 | 값 |\n|------|-----|\n"
    for key, value in data.items():
        if value is None:
            value = "데이터가 없습니다"
        formatted_data += f"| {key} | {value} |\n"

    return formatted_data

# Ollama LLM 설정 (변경된 클래스 사용)
try:
    logging.info("Initializing Ollama LLM...")
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3.1")
    )
    logging.info("Ollama LLM initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Ollama LLM: {e}")
    llm = None

# 메인 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 보고서 라우트
@app.route('/report/<report_id>')
def report(report_id):
    report = reports.get(report_id)
    if not report:
        return "보고서를 찾을 수 없습니다.", 404
    # Markdown을 HTML로 변환
    report_html = markdown.markdown(report['content'], extensions=['tables', 'fenced_code'])
    
    # 서버 측에서 섹션 추출
    sections = re.findall(r'<h2>(.*?)</h2>', report_html)
    
    return render_template('report.html', report_content=report_html,
                           sections=sections,
                           current_date=datetime.now().strftime("%Y-%m-%d"),
                           product_name=report['product_name'])

# 채팅 요청 처리 엔드포인트
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message')
        logging.info(f"Received user input: {user_input}")
        
        if not user_input:
            return jsonify({'message': '입력된 메시지가 없습니다.'}), 400

        # 특정 질문 패턴 정의 (수정된 정규 표현식)
        specific_question_pattern = r'(.+?)라는 제품의 전과정평가 보고서를 작성해줘~?'

        match = re.match(specific_question_pattern, user_input.strip())
        if match:
            product_name = match.group(1)
            logging.info(f"Extracted product name: {product_name}")

            # DB에서 모든 섹션의 제목과 내용을 가져옴
            sections = get_all_lca_sections()
            logging.info(f"Retrieved {len(sections)} sections from the database.")

            # DB에서 수치 데이터 가져오기
            numerical_data = get_numerical_data()
            logging.info(f"Retrieved numerical data: {numerical_data}")

            if sections:
                # 각 섹션의 내용을 조합하여 프롬프트 생성
                assembled_sections = ""
                if numerical_data:
                    db_vars = numerical_data[0]
                else:
                    db_vars = {}
                    logging.warning("No numerical data found. Proceeding without DB variables.")

                for title, body in sections:
                    try:
                        # 플레이스홀더 추출
                        formatter = string.Formatter()
                        placeholders = [field_name for _, field_name, _, _ in formatter.parse(body) if field_name]

                        # 필요한 변수 제공, None값은 "데이터가 없습니다"로 대체
                        variables = db_vars.copy()
                        variables['PRODUCT_NAME'] = product_name  # 제품명 추가

                        # 변수가 없으면 "데이터가 없습니다"로 대체
                        for placeholder in placeholders:
                            if placeholder not in variables or variables[placeholder] is None:
                                variables[placeholder] = "데이터가 없습니다"

                        # 템플릿에 변수 채우기
                        filled_body = body.format(**variables)
                        assembled_sections += f"\n\n## {title}\n{filled_body}"
                    except KeyError as e:
                        logging.error(f"Missing variable {e} in section '{title}'. Skipping this section.")
                        continue  # 섹션 스킵
                    except Exception as e:
                        logging.error(f"Unexpected error formatting section '{title}': {e}")
                        continue  # 섹션 스킵

                if not assembled_sections.strip():
                    return jsonify({'message': '모든 섹션에서 필요한 변수가 누락되어 보고서를 생성할 수 없습니다.'}), 400

                # 수치 데이터를 포맷팅
                formatted_db_data = format_db_data(numerical_data)

                # 최종 프롬프트 생성 (Markdown 형식으로 응답하도록 지시)
                full_prompt = f"""
DB 데이터: {formatted_db_data}

이 데이터를 참고하고

프롬프트 템플릿과 예시응답을 기반으로, 다음의 섹션들을 포함하는 전문적이고 구조화된 Markdown 형식의 전과정평가 보고서를 작성해 주세요. 각 섹션은 적절한 제목과 본문을 포함해야 하며, 표, 리스트, 강조 등을 사용하여 가독성을 높여주세요.

{assembled_sections}

위의 내용을 참고하여 최종 보고서를 Markdown 형식으로 생성해 주세요. 표가 꼭 포함되어야 합니다.
"""
                logging.info(f"Full prompt: {full_prompt}")

                if llm is None:
                    return jsonify({'message': 'LLM이 로드되지 않았습니다.'}), 500

                try:
                    # Ollama LLM에 최종 프롬프트 전달 (파라미터 설정)
                    bot_response = llm(full_prompt, max_tokens=3000, temperature=1, top_p=0.8)
                    logging.info(f"LLM response: {bot_response}")

                    # 고유한 보고서 ID 생성
                    report_id = str(uuid.uuid4())
                    reports[report_id] = {
                        'content': bot_response,
                        'product_name': product_name
                    }

                    return jsonify({'type': 'report', 'report_id': report_id})
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    return jsonify({'message': f"응답 생성 중 오류가 발생했습니다: {str(e)}"}), 500
            else:
                return jsonify({'message': '데이터베이스에서 섹션을 찾을 수 없습니다.'}), 404
        else:
            # 일반적인 질문 처리 (Ollama 모델 사용)
            if llm is None:
                return jsonify({'message': 'LLM이 로드되지 않았습니다.'}), 500
            
            try:
                # 단순 프롬프트 생성 (간결하게 응답하도록 지시)
                full_prompt = f"사용자 입력: {user_input}\n\n답변: 간결하게 요점만 전달해 주세요."
                logging.info(f"Full prompt for general question: {full_prompt}")

                # Ollama LLM에 프롬프트 전달 (파라미터 설정)
                bot_response = llm(full_prompt, max_tokens=500, temperature=0.8, top_p=0.8)
                logging.info(f"LLM response for general question: {bot_response}")
                return jsonify({'type': 'general', 'message': bot_response})
            except Exception as e:
                logging.error(f"Error generating response for general question: {e}")
                return jsonify({'message': f"응답 생성 중 오류가 발생했습니다: {str(e)}"}), 500

    except Exception as e:
        logging.exception("Unexpected error:")
        return jsonify({'message': '서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.'}), 500

if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(debug=True)
