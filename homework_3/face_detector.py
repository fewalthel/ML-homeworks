import cv2
import mediapipe as mp
import numpy as np

# Загрузка обученной модели распознавания лица
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_recognizer.yml')

# Загрузка классификатора лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Инициализация видеопотока
cap = cv2.VideoCapture(0)

# Ваши данные
name = "Svetlana"
surname = "Berezhnaya"

# Инициализация детекторов
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Не удалось захватить изображение.")
        break

    # Конвертация в RGB и серый (для OpenCV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция лиц (каскады Хаара)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Детекция рук (MediaPipe)
    hand_results = hands.process(image_rgb)

    # Детекция лица (MediaPipe Face Mesh)
    face_results = face_mesh.process(image_rgb)

    for (x, y, w, h) in faces:
        # Предсказание лица (LBPH)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))
        label, confidence = model.predict(face_resize)

        # Отображение рамки и имени
        name_text = "It's You" if label == 0 and confidence < 100 else "Unknown"
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, name_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Подсчет поднятых пальцев
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                fingers_up = 0
                landmarks = hand_landmarks.landmark

                # Проверка пальцев (8, 12, 16, 20 — кончики пальцев)
                for i, tip in enumerate([8, 12, 16, 20]):
                    if landmarks[tip].y < landmarks[tip - 2].y:
                        fingers_up += 1

                # Вывод имени при 1 пальце
                if fingers_up == 1:
                    cv2.putText(image, name, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Вывод фамилии при 2 пальцах
                elif fingers_up == 2:
                    cv2.putText(image, surname, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Анализ эмоции при 3 пальцах
                elif fingers_up == 3 and face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Координаты ключевых точек рта и бровей
                        lips_top = face_landmarks.landmark[13].y  # Верхняя губа
                        lips_bottom = face_landmarks.landmark[14].y  # Нижняя губа
                        mouth_open = lips_bottom - lips_top  # Открытость рта

                        left_eyebrow = face_landmarks.landmark[65].y  # Левая бровь
                        right_eyebrow = face_landmarks.landmark[295].y  # Правая бровь

                        # Определение эмоции
                        if mouth_open > 0.03:  # Рот открыт → happy
                            emotion = "Happy"
                        elif (left_eyebrow > face_landmarks.landmark[63].y + 0.01 and
                              right_eyebrow > face_landmarks.landmark[293].y + 0.01):  # Брови опущены → sad
                            emotion = "Sad"
                        else:
                            emotion = "Neutral"

                        # Вывод эмоции
                        cv2.putText(image, f"Emotion: {emotion}", (x, y + h + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Webcam', image)

    # Выход по ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()