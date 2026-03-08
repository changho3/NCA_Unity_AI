import os
import numpy as np
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# API 키 및 클라이언트 설정
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("오류: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 입력해주세요.")

client = genai.Client(api_key=api_key)

app = FastAPI(title="Command Classification API")

# 코사인 유사도 계산 함수
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b))

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

def get_embedding(text: str) -> list[float]:
    """주어진 텍스트의 임베딩 벡터를 반환합니다."""
    # 최신 임베딩 모델 사용
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values

# 전역 변수로 기준 벡터 캐싱
CATEGORY_EMBEDDINGS = {}

@app.on_event("startup")
def startup_event():
    print("서버 시작 중: 기준 데이터 임베딩 생성...")
    # 각 카테고리의 대표 문장들을 임베딩하여 평균 벡터(또는 각 벡터 배열) 저장
    for category_name, sentences in categories.items():
        embeddings = []
        for sentence in sentences:
            emb = get_embedding(sentence)
            embeddings.append(emb)
        # 여러 대표 문장의 임베딩 평균을 구하여 해당 카테고리의 기준 벡터로 사용
        avg_embedding = np.mean(embeddings, axis=0)
        CATEGORY_EMBEDDINGS[category_name] = avg_embedding
    print("시스템 준비 완료!")

class ClassificationRequest(BaseModel):
    text: str

class CategoryScore(BaseModel):
    category: str
    score: float

class ClassificationResponse(BaseModel):
    best_match: str
    best_score: float
    all_scores: list[CategoryScore]

@app.post("/classify", response_model=ClassificationResponse)
def classify_text(req: ClassificationRequest):
    user_input = req.text.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")
        
    try:
        # 사용자 입력 텍스트 임베딩
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_input,
        )
        query_embedding = response.embeddings[0].values
        
        # 각 카테고리와의 코사인 유사도 계산
        similarities = {}
        for category_name, category_vec in CATEGORY_EMBEDDINGS.items():
            sim = cosine_similarity(query_embedding, category_vec)
            similarities[category_name] = sim
        
        # 유사도 기준 내림차순 정렬
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        best_match, best_score = sorted_similarities[0]
        
        all_scores = [
            CategoryScore(category=cat, score=score)
            for cat, score in sorted_similarities
        ]
        
        return ClassificationResponse(
            best_match=best_match,
            best_score=best_score,
            all_scores=all_scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩/분류 중 오류 발생: {str(e)}")

# uvicorn server:app --reload
