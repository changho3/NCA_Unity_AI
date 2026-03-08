import os
import numpy as np
from google import genai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# API 키 및 클라이언트 설정
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("오류: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 입력해주세요.")
    exit(1)

client = genai.Client(api_key=api_key)

# 코사인 유사도 계산 함수
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# 분류할 기준 카테고리와 대표 문장들
categories = {
    "직진 (Go Straight)": [
        "앞으로 가",
        "직진해",
        "계속 나아가",
        "앞으로 이동해",
        "직진"
    ],
    "후진 (Go Backward)": [
        "뒤로 가",
        "후진해",
        "뒤로 이동해",
        "후진"
    ],
    "좌회전 (Turn Left)": [
        "왼쪽으로 방향을 틀어주세요",
        "좌회전해",
        "왼쪽으로 가",
        "좌측으로 이동해"
    ],
    "우회전 (Turn Right)": [
        "오른쪽으로 방향을 틀어주세요",
        "우회전해",
        "오른쪽으로 가",
        "우측으로 이동해"
    ],
    "유턴 (U-Turn)": [
        "유턴해",
        "차 돌려",
        "반대 방향으로 돌아가",
        "유턴"
    ]
}

def get_embedding(text):
    """주어진 텍스트의 임베딩 벡터를 반환합니다."""
    # 최신 임베딩 모델 사용
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values

def main():
    print("기준 데이터 임베딩 생성 중...")
    category_embeddings = {}
    
    # 각 카테고리의 대표 문장들을 임베딩하여 평균 벡터(또는 각 벡터 배열) 저장
    for category_name, sentences in categories.items():
        embeddings = []
        for sentence in sentences:
            emb = get_embedding(sentence)
            embeddings.append(emb)
        # 여러 대표 문장의 임베딩 평균을 구하여 해당 카테고리의 기준 벡터로 사용
        avg_embedding = np.mean(embeddings, axis=0)
        category_embeddings[category_name] = avg_embedding
    
    print("\n[ 시스템 준비 완료 ]")
    print("종료하려면 '종료'를 입력하세요.\n")
    
    while True:
        user_input = input(">> 명령어를 입력하세요 (예: 15km 속도로 좌회전해줘): ")
        
        if user_input.strip() == "":
            continue
        if user_input.strip() == "종료":
            print("프로그램을 종료합니다.")
            break
            
        try:
            # 사용자 입력 텍스트 임베딩
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=user_input,
            )
            query_embedding = response.embeddings[0].values
            
            # 각 카테고리와의 코사인 유사도 계산
            similarities = {}
            for category_name, category_vec in category_embeddings.items():
                sim = cosine_similarity(query_embedding, category_vec)
                similarities[category_name] = sim
            
            # 유사도 기준 내림차순 정렬
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # 결과 출력
            print("\n[ 분류 결과 ]")
            best_match, best_score = sorted_similarities[0]
            print(f"✅ 가장 유사한 카테고리: {best_match} (유사도: {best_score:.4f})")
            
            print("--- 전체 유사도 ---")
            for cat, score in sorted_similarities:
                print(f" - {cat}: {score:.4f}")
            print("-" * 30 + "\n")
            
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
