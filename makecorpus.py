# 파일에서 1000개의 텍스트 데이터 읽기
file_path = "/Users/jeongminsu/Downloads/lca_dataset_1000_examples.txt"
with open(file_path, 'r') as f:
    texts = f.read().split('\n')

# 텍스트 데이터를 하나로 합치기
combined_text = "\n".join(texts)

# SQL 파일 생성
sql_file_path = "/Users/jeongminsu/Downloads/insert.sql"
toc_id = 28  # 넣고 싶은 toc_id 값
with open(sql_file_path, 'w') as f:
    f.write(f"INSERT INTO lca_toc (toc_id, body) VALUES ({toc_id}, $$ {combined_text} $$);")
