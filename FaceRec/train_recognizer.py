import cv2
import os
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []
names = {}
label_counter = 0

# Traverse through the 'images' folder to get images and labels
for person_name in os.listdir("images"):
    person_folder = os.path.join("images", person_name)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Resize the image to a fixed size (e.g., 100x100)
                img_resized = cv2.resize(img, (100, 100))  # Resize to 100x100

                faces.append(img_resized)
                labels.append(label_counter)
                names[label_counter] = person_name
        label_counter += 1

# Convert faces and labels to NumPy arrays
faces = np.array(faces)
labels = np.array(labels)

# Train the recognizer
recognizer.train(faces, labels)

# Save the trained model
recognizer.save("face_recognizer.yml")
print("Training completed and model saved.")
