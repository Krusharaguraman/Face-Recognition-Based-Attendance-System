# Face Recognition Based Attendance System

A real-time attendance system built with OpenCV, Haar Cascade for face detection, and LBPH (Local Binary Patterns Histograms) for face recognition. Attendance records are stored in MongoDB.

## Features

- **Real-time Face Detection** using Haar Cascade classifier
- **Face Recognition** using LBPH algorithm
- **MongoDB Integration** for storing attendance records
- **Duplicate Prevention** - one attendance per person per day
- **Unknown Face Capture** - saves images of unrecognized faces

## Tech Stack

- Python 3.7+
- OpenCV (with contrib modules)
- Haar Cascade Classifier
- LBPH Face Recognizer
- MongoDB
- PyMongo

## Project Structure

```
Attendance/
├── .gitignore                              # Git ignore file
├── .vscode/                                # VS Code settings
├── dataset/                                # Training face images
│   └── <person_name>/
│       ├── 1.png
│       ├── 2.png
│       └── ...
├── haarcascade_frontalface_default.xml     # Haar Cascade classifier
├── trainer.yml                             # Trained LBPH model
├── names.pkl                               # Name-to-ID mapping
├── collect_data.py                         # Collect face samples
├── train_model.py                          # Train LBPH model
├── attendance_system.py                    # Run attendance system
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Prerequisites

- Python 3.7 or higher
- MongoDB installed and running on localhost:27017
- Webcam

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Krusharaguraman/Face-Recognition-Based-Attendance-System.git
cd Face-Recognition-Based-Attendance-System
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Collect Face Data

Run the data collection script to register faces:

```bash
python collect_data.py
```

- Enter the person's name when prompted
- 30 face samples will be captured automatically
- Repeat for each person you want to register

### Step 2: Train the Model

Train the LBPH face recognizer:

```bash
python train_model.py
```

This generates:
- `trainer.yml` - trained model
- `names.pkl` - name-to-ID mapping

### Step 3: Run Attendance System

Start the real-time attendance system:

```bash
python attendance_system.py
```

- Camera opens and detects faces
- Recognized faces are matched against trained model
- Attendance is marked in MongoDB
- **Press `ESC` to exit**

## MongoDB Configuration

Default configuration in `attendance_system.py`:

```python
MONGO_CONFIG = {
    'host': 'localhost',
    'port': 27017,
    'database': 'attendance_system',
    'collection': 'attendance'
}
```

### Document Structure

Each attendance record is stored as:

```json
{
    "_id": ObjectId("..."),
    "name": "John Doe",
    "date": "2024-01-15",
    "time": "09:30:00",
    "timestamp": ISODate("2024-01-15T09:30:00.000Z")
}
```

### Query Examples

```python
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['attendance_system']
collection = db['attendance']

# Get all records
all_records = collection.find()

# Get records for a specific person
person_records = collection.find({'name': 'John Doe'})

# Get records for a specific date
date_records = collection.find({'date': '2024-01-15'})
```

### Using MongoDB Shell

```bash
mongosh
```

```javascript
use attendance_system

// View all attendance records
db.attendance.find().pretty()

// View records for a specific person
db.attendance.find({ name: "John Doe" }).pretty()

// Count total records
db.attendance.countDocuments()
```

## How It Works

### Haar Cascade Face Detection

Haar Cascade is a machine learning-based approach where a cascade function is trained from a lot of positive and negative images. It is used to detect objects in other images.

Parameters used:
- **Scale Factor**: `1.3` - How much the image size is reduced at each image scale
- **Min Neighbors**: `5` - How many neighbors each candidate rectangle should have to retain it

### LBPH Face Recognition

LBPH (Local Binary Patterns Histograms) is a face recognition algorithm that:
1. Extracts local binary patterns from face images
2. Builds histograms of these patterns
3. Compares histograms to recognize faces

**Confidence Threshold**: `800` (lower value = better match)

## Configuration

### Adjusting Confidence Threshold

In `attendance_system.py`:
```python
confidence_threshold = 800  # Lower = stricter, Higher = more lenient
```

### Adjusting Face Detection

In `collect_data.py` and `attendance_system.py`:
```python
faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
# scaleFactor: 1.1-1.5 (higher = faster but may miss faces)
# minNeighbors: 3-6 (higher = fewer false positives)
```

## Requirements

```
opencv-contrib-python>=4.5.0.52
numpy>=1.21.0
pymongo>=4.0.0
Pillow>=8.0.0
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Failed to load cascade classifier` | Ensure `haarcascade_frontalface_default.xml` is in project folder |
| `Missing model files` | Run `train_model.py` after collecting data |
| `Could not open webcam` | Check camera connection and permissions |
| `pymongo not found` | Run `pip install pymongo` |
| `MongoDB connection error` | Ensure MongoDB is running on localhost:27017 |
| Poor recognition accuracy | Collect more samples with varied lighting and angles |

## Use Cases

- **Classroom Attendance**: Automatically mark student attendance
- **Employee Attendance**: Track employee check-ins
- **Event Management**: Verify attendees at events
- **Security Access**: Restricted area access control

## License

This project is open source and available under the MIT License.

## Author

**Krusharaguraman**
- GitHub: [@Krusharaguraman](https://github.com/Krusharaguraman)
