import cv2
import pickle
import datetime
import os
import time
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# ---------------------------
# MongoDB Configuration
# ---------------------------
MONGO_CONFIG = {
    'host': 'localhost',
    'port': 27017,
    'database': 'attendance_system',
    'collection': 'attendance'
}

# Initialize MongoDB connection
def get_mongo_collection():
    """Get MongoDB collection instance."""
    client = MongoClient(MONGO_CONFIG['host'], MONGO_CONFIG['port'])
    db = client[MONGO_CONFIG['database']]
    collection = db[MONGO_CONFIG['collection']]
    # Create unique index on name and date to prevent duplicates
    collection.create_index([('name', 1), ('date', 1)], unique=True)
    return collection, client

def mark_attendance_db(name, date, time_str):
    """Insert attendance record. Returns True if inserted, False if duplicate or error."""
    client = None
    try:
        collection, client = get_mongo_collection()
        document = {
            'name': name,
            'date': date,
            'time': time_str,
            'timestamp': datetime.datetime.now()
        }
        collection.insert_one(document)
        return True
    except DuplicateKeyError:
        return False
    except Exception as err:
        print(f"MongoDB error: {err}")
        return False
    finally:
        if client:
            client.close()

# ---------------------------
# Face Recognition Setup
# ---------------------------
# Use local haarcascade file
haar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')
width, height = 130, 100
confidence_threshold = 800

# Load trained model
if not os.path.exists("trainer.yml") or not os.path.exists("names.pkl"):
    print("Error: Missing model files. Run train_model.py first.")
    exit()

model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

with open("names.pkl", "rb") as f:
    names = pickle.load(f)
print("Loaded names:", names)

# Load face detector using Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Error: Failed to load Haar Cascade classifier.")
    exit()

# Initialize webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Cooldown dictionary to avoid printing every frame
last_marked = {}   # name -> timestamp (seconds)
cooldown_seconds = 5   # minimum seconds between prints for the same person

print("Haar Cascade + LBPH Attendance System (MongoDB) Ready. Press ESC to quit.")
unknown_counter = 0

while True:
    ret, im = webcam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        # Use LBPH for face recognition
        label, confidence = model.predict(face_resize)

        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if confidence < confidence_threshold:
            name = names[label]
            text = f"{name} ({int(confidence)})"
            cv2.putText(im, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            today = datetime.date.today().isoformat()
            now = datetime.datetime.now().strftime("%H:%M:%S")
            current_time = time.time()

            # Only attempt DB insert and print if cooldown has passed
            if name not in last_marked or (current_time - last_marked[name]) > cooldown_seconds:
                if mark_attendance_db(name, today, now):
                    print(f"✅ {name} marked at {now}")
                else:
                    print(f"ℹ️ {name} already marked today")
                last_marked[name] = current_time

            unknown_counter = 0
        else:
            cv2.putText(im, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            unknown_counter += 1
            if unknown_counter > 100:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"unknown_{ts}.jpg", im)
                print(f"⚠️ Unknown person saved as unknown_{ts}.jpg")
                unknown_counter = 0

    cv2.imshow('Haar Cascade + LBPH Attendance (MongoDB)', im)
    if cv2.waitKey(10) == 27:   # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
print("System exited.")