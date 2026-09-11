import cv2
import os

# Use local haarcascade file
haar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')
datasets = 'dataset'
(width, height) = (130, 100)

sub_data = input("Enter person's name: ").strip()
path = os.path.join(datasets, sub_data)
os.makedirs(path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Error: Failed to load cascade classifier. Check OpenCV installation.")
    exit()

webcam = cv2.VideoCapture(0)  # Use 0 for built-in camera, 1 for external
count = 1
print(f"Collecting 30 samples for {sub_data}...")
while count < 31:
    ret, im = webcam.read()
    if not ret:
        continue
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f"{path}/{count}.png", face_resize)
        count += 1
    cv2.imshow('Collecting Faces', im)
    if cv2.waitKey(10) == 27:  # ESC to exit early
        break

webcam.release()
cv2.destroyAllWindows()
print(f"Done! Collected {count-1} samples for {sub_data}")