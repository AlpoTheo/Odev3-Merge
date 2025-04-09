import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Mediapipe'nin el tespiti modüllerini başlatıyoruz
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Video kaynağını açıyoruz
cap = cv2.VideoCapture(0)

# Tkinter arayüzünü başlatıyoruz
root = tk.Tk()
root.title("Hand Tracking with AI")
root.geometry("800x600")

# Video penceresini eklemek için bir canvas oluşturuyoruz
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Çıkış butonu
def on_close():
    cap.release()
    root.quit()

root.protocol("WM_DELETE_WINDOW", on_close)

# Video akışını Tkinter canvas'ına aktaran fonksiyon
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

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

            # Parmak uçlarını kırmızı noktalarla işaretleme
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (index_tip_x, index_tip_y), 5, (0, 0, 255), -1)

    # Tkinter'a uyumlu hale getirmek için OpenCV formatını değiştiriyoruz
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img_tk = ImageTk.PhotoImage(image=img)

    # Canvas'ta görüntüyü güncelliyoruz
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk

    # Fonksiyonu tekrar çağırıyoruz
    root.after(10, update_frame)

# Başlangıçta video akışını başlatıyoruz
update_frame()

# Uygulama penceresini çalıştırıyoruz
root.mainloop()
