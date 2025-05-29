import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# 모델 및 데이터 파일 경로 설정
face_cascade_path = 'haarcascade_frontalface_default.xml'
predictor_path = 'shape_predictor_68_face_landmarks.dat'
emotion_model_path = 'emotion_model.hdf5'

# 얼굴 감지 CascadeClassifier 객체 생성
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# dlib 얼굴 특징점 예측기 객체 생성
predictor = dlib.shape_predictor(predictor_path)

# 감정 인식 모델 로드
emotion_model = load_model(emotion_model_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 비디오 캡처 객체 생성 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = frame[y:y + h, x:x + w]

        # dlib을 사용하여 얼굴 특징점 예측
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, dlib_rect)
        face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # 얼굴 특징점을 사용하여 감정 인식 모델에 입력할 이미지 전처리
        face_img = cv2.resize(face_roi_gray, (48, 48))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        # 감정 예측
        emotion_prediction = emotion_model.predict(face_img)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_index = np.argmax(emotion_prediction)
        predicted_emotion = emotion_labels[emotion_label_index]

        # 얼굴 영역과 감정 레이블을 프레임에 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{predicted_emotion} ({emotion_probability:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # (선택 사항) 얼굴 특징점 그리기
        # for (px, py) in face_landmarks:
        #     cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

    # 결과 프레임 보여주기
    cv2.imshow('Real-time Face Emotion Recognition', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()