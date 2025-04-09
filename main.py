import cv2
import mediapipe as mp

# Mediapipe'nin el tespiti modüllerini başlatıyoruz
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Video kaynağını açıyoruz
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntü işleme: BGR'den RGB'ye dönüştürme
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Eğer elde parmaklar algılanmışsa
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Her bir eldeki işaretçileri çizeceğiz
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Parmakların uçlarını tespit etme (örneğin baş parmak)
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Ekranda parmak uçlarını gösterme
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            middle_tip_x, middle_tip_y = int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])

            # Parmak uçlarını kırmızı noktalarla işaretleme
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (index_tip_x, index_tip_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (middle_tip_x, middle_tip_y), 5, (0, 0, 255), -1)

            # Parmaklar arası mesafe hesaplama örneği
            distance = ((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2) ** 0.5
            cv2.putText(frame, f'Distance: {int(distance)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Sonuçları gösteriyoruz
    cv2.imshow('Hand Tracking', frame)

    # 'q' tuşuna basıldığında çıkış yapar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
