import streamlit as st
from recommender import get_music_recommendation
from utils import extract_youtube_url

st.set_page_config(page_title="🎵 감정 기반 음악 추천", layout="centered")
st.title("🎵 감정 기반 실시간 음악 추천")

emotion = st.selectbox("감정을 선택하세요", ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])

if st.button("음악 추천 받기"):
    with st.spinner("추천 중..."):
        result = get_music_recommendation(emotion)
        youtube_url = extract_youtube_url(result)

        st.markdown(f"**감정:** {emotion}")
        st.markdown(f"**추천 결과:**\n\n{result}")

        if youtube_url:
            st.video(youtube_url)
        else:
            st.warning("유튜브 링크를 찾을 수 없습니다.")
