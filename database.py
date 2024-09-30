from sqlalchemy import create_engine, inspect

# 데이터베이스 연결
db_url = "postgresql://sa:1@192.168.0.20:11032/Version.1"
engine = create_engine(db_url)

# 인스펙터를 사용해 테이블과 열 정보를 가져옴
inspector = inspect(engine)

# 스키마 정보를 저장할 변수
schema_info = ""

# 모든 테이블 이름 출력 및 스키마 정보 저장
tables = inspector.get_table_names()
schema_info += "Tables and Columns in the Database:\n"

# 각 테이블의 컬럼 정보를 가져옴
for table_name in tables:
    schema_info += f"Table: {table_name}\n"
    columns = inspector.get_columns(table_name)
    for column in columns:
        schema_info += f"  {column['name']} - {column['type']}\n"

# 결과 출력
print(schema_info)
