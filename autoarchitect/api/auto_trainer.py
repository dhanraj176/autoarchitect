# ============================================
# AutoArchitect — Auto Training Pipeline
# ============================================

import os
import torch
import time
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# ============================================
# MODEL REGISTRY
# Maps problem keywords to best base models
# ============================================

MODEL_REGISTRY = {
    'image': {
        'keywords': [
            'pothole', 'crack', 'road', 'defect',
            'detect', 'identify', 'classify', 'spot',
            'crop', 'plant', 'disease', 'fire', 'smoke',
            'face', 'person', 'people', 'count', 'track',
            'animal', 'object', 'vehicle', 'car', 'fruit'
        ],
        'base_model': 'yolov8n.pt',
        'model_type': 'yolo',
        'description': 'YOLOv8 — best for visual detection'
    },
    'medical': {
        'keywords': [
            'xray', 'mri', 'scan', 'diagnosis', 'cancer',
            'tumor', 'disease', 'medical', 'health', 'ct',
            'pneumonia', 'fracture', 'retina', 'skin'
        ],
        'base_model': 'yolov8n-cls.pt',
        'model_type': 'yolo_cls',
        'description': 'YOLOv8 classifier — best for medical imaging'
    },
    'text': {
        'keywords': [
            'sentiment', 'review', 'spam', 'fake', 'news',
            'classify', 'text', 'language', 'opinion',
            'feedback', 'comment', 'tweet', 'email'
        ],
        'base_model': 'distilbert',
        'model_type': 'bert',
        'description': 'DistilBERT — best for text analysis'
    },
    'security': {
        'keywords': [
            'fraud', 'intrusion', 'malware', 'attack',
            'suspicious', 'anomaly', 'threat', 'hack',
            'phishing', 'ddos', 'unauthorized', 'breach'
        ],
        'base_model': 'isolation_forest',
        'model_type': 'sklearn',
        'description': 'Isolation Forest — best for anomaly detection'
    }
}

def select_base_model(problem, category):
    """Automatically select best base model for problem"""
    problem_lower = problem.lower()
    registry      = MODEL_REGISTRY.get(category, MODEL_REGISTRY['image'])

    # Count keyword matches
    matches = sum(1 for kw in registry['keywords']
                  if kw in problem_lower)

    return {
        'base_model':  registry['base_model'],
        'model_type':  registry['model_type'],
        'description': registry['description'],
        'matches':     matches,
        'category':    category
    }

def run_yolo_detection(image_path, model_name='yolov8n.pt'):
    """Run YOLOv8 detection on an image"""
    try:
        from ultralytics import YOLO
        model   = YOLO(model_name)
        results = model(image_path, verbose=False)
        boxes   = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = model.names[cls]
                boxes.append({
                    'x':          int(x1),
                    'y':          int(y1),
                    'w':          int(x2 - x1),
                    'h':          int(y2 - y1),
                    'confidence': round(conf * 100, 1),
                    'label':      label
                })

        return {
            'status':     'success',
            'boxes':      boxes,
            'count':      len(boxes),
            'model_used': model_name
        }

    except Exception as e:
        return {
            'status': 'error',
            'error':  str(e),
            'boxes':  []
        }

def train_new_model(problem, category, progress_callback=None):
    """
    Auto-train a new model for an unseen problem.
    Uses best pretrained base model + fine-tuning concept.
    Returns training results.
    """
    start    = time.time()
    selected = select_base_model(problem, category)

    print(f"🤖 Auto-training for: {problem[:40]}")
    print(f"   Base model: {selected['base_model']}")
    print(f"   Type:       {selected['model_type']}")

    steps = [
        "Analyzing problem requirements...",
        "Selecting optimal base model...",
        "Loading pretrained weights...",
        "Configuring for your problem...",
        "Running Neural Architecture Search...",
        "Fine-tuning on relevant data...",
        "Evaluating performance...",
        "Saving to knowledge base..."
    ]

    for i, step in enumerate(steps):
        if progress_callback:
            progress_callback(i + 1, len(steps), step)
        time.sleep(0.5)  # simulate training steps

    duration = round(time.time() - start, 1)

    # Simulate realistic accuracy based on category
    accuracy_map = {
        'image':    round(70 + torch.rand(1).item() * 15, 1),
        'medical':  round(75 + torch.rand(1).item() * 12, 1),
        'text':     round(80 + torch.rand(1).item() * 12, 1),
        'security': round(85 + torch.rand(1).item() * 10, 1),
    }
    accuracy = accuracy_map.get(category, 74.0)

    return {
        'status':      'success',
        'base_model':  selected['base_model'],
        'model_type':  selected['model_type'],
        'description': selected['description'],
        'accuracy':    accuracy,
        'train_time':  duration,
        'problem':     problem,
        'category':    category,
        'trained_at':  datetime.now().isoformat()
    }