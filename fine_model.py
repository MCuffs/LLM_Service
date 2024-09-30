from sentence_transformers import SentenceTransformer
import json


file_path = '/Users/jeongminsu/Downloads/template_augmented.json'

# 파일을 열고 JSON 데이터를 로드
with open(file_path, 'r', encoding='utf-8') as f:
    try:
        data = json.load(f)  # json.load()는 파일 객체를 사용합니다.
        print("JSON 데이터가 성공적으로 로드되었습니다.")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")

# Load pre-trained model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract content and generate embeddings
documents = []
for entry in data['domain_knowledge']:
    content = entry.get('contents', {}).get('content', [])
    for section in content:
        embedding = model.encode(section)
        documents.append({
            "text": section,
            "embedding": embedding
        })