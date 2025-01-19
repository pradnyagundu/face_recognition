import cv2
import numpy as np

# Load the trained face recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Map labels to names (this should match your training data labels)
# Replace these with the actual names and their respective labels
names = {
    0: "Pradnya",  # Replace with the name associated with label 0
    1: "vivek",   # Replace with the name associated with label 1
    # Add more mappings if you have more people in your dataset
}

print("Face recognition in progress. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y + h, x:x + w]

        # Predict the label (name) and confidence
        label, confidence = recognizer.predict(face)

        # Get the name from the dictionary using the predicted label
        if label in names:
            name = names[label]
        else:
            name = "Unknown"

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the name on the frame
        cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
