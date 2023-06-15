
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.classes = self.model.names

    def process_frame(self, frame):
        results = self.model(frame)[0]
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > 0.5:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'score': round(score, 2),
                    'label': self.classes[int(class_id)]
                })
        return detections

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['label']} {det['score']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def run_on_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            detections = self.process_frame(frame)
            frame = self.draw_detections(frame, detections)
            out.write(frame)
            
        cap.release()
        out.release()

if __name__ == "__main__":
    detector = ObjectDetector()
    print("YOLOv8 Detector initialized.")
