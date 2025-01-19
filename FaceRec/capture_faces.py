import cv2
import os

# Set up face detector and camera
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Create directory to store images if not exists
if not os.path.exists("images"):
    os.mkdir("images")

print("Enter your name:")
name = input()  # Get name from user input
name_folder = os.path.join("images", name)

if not os.path.exists(name_folder):
    os.mkdir(name_folder)

print("Capturing faces... Press 'q' to stop.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y + h, x:x + w]
        cv2.imwrite(f"{name_folder}/{name}_{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {count} images of {name}.")
