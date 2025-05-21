from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL_NAME

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_music_recommendation(emotion: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "너는 음악 추천 AI야. 감정에 맞는 한국 노래를 추천하고 유튜브 링크도 같이 알려줘.",
        },
        {
            "role": "user",
            "content": f"감정이 '{emotion}'일 때 어울리는 한국 노래 추천해줘. 유튜브 링크도 같이 알려줘.",
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    return response.choices[0].message.content
