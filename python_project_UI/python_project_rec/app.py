import streamlit as st
import cv2
import time
from collections import Counter
from deepface import DeepFace
from recommender import get_music_recommendation
from utils import extract_youtube_url

st.set_page_config(page_title="🎵 감정 기반 음악 추천", layout="centered")
st.title("🎵 감정 기반 실시간 음악 추천")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None

if st.button("오늘의 감정 확인하기"):
    info_placeholder = st.empty()
    countdown_placeholder = st.empty()
    wait_time = 10
    detected_emotions = []

    for elapsed in range(wait_time):
        remaining = wait_time - elapsed
        if remaining == wait_time:
            info_placeholder.info("잠시후 10초 동안 표정을 분석합니다. 잠시 기다려주세요.")
        else:
            info_placeholder.info("시작합니다!")
        countdown_placeholder.write(f"⏳ 남은 시간: {remaining}초")

        # 웹캠 프레임 읽기 및 감정 분석 (실제 환경에서는 아래 코드가 while문 안에 위치)
        if elapsed == 0:
            cap = cv2.VideoCapture(0)
            start_time = time.time()

        ret, frame = cap.read()
        if ret:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion'].capitalize()
                if emotion in emotions:
                    detected_emotions.append(emotion)
            except Exception:
                pass
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()
    countdown_placeholder.write("분석 완료!")
    info_placeholder.empty()

    if detected_emotions:
        most_common_emotion = Counter(detected_emotions).most_common(1)[0][0]
        st.session_state.detected_emotion = most_common_emotion
        st.success(f"오늘의 감정은 **{most_common_emotion}** 입니다!")
    else:
        st.warning("감정을 감지하지 못했습니다. 다시 시도해 주세요.")

emotion = st.selectbox(
    "감정을 선택하세요. 만약 AI기반 얼굴인식을 통해 내 표정의 감정을 알고 싶다면 ""오늘의 감정 확인하기"" 버튼을 눌려 주세요! ",
    emotions,
    index=emotions.index(st.session_state.detected_emotion) if st.session_state.detected_emotion else 0
)

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
