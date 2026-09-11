import cv2
import numpy as np
import os
import pickle

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'

print("Training...")

images = []
labels = []
names = {}  # id -> name
current_id = 0

# Walk through dataset directory
for (root, dirs, files) in os.walk(datasets):
    for dir_name in dirs:
        names[current_id] = dir_name
        subject_path = os.path.join(datasets, dir_name)
        
        for file_name in os.listdir(subject_path):
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                path = os.path.join(subject_path, file_name)
                label = current_id
                im = cv2.imread(path, 0)  # read as grayscale
                images.append(im)
                labels.append(label)
        current_id += 1

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

print(f"Loaded {len(images)} images for {len(names)} person(s).")

# Train LBPH face recognizer
# model = cv2.face.FisherFaceRecognizer_create()  # uncomment to use Fisherface
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Save the model
model.save("trainer.yml")
print("Model saved as trainer.yml")

# Save the names mapping
with open("names.pkl", "wb") as f:
    pickle.dump(names, f)
print("Names mapping saved as names.pkl")