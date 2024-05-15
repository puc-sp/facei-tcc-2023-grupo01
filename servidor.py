import streamlit as st
import cv2
import numpy as np
from pytube import YouTube
import torch
from torchvision.transforms import transforms
from PIL import Image
from io import BytesIO
import requests
import torch
import torchvision.models as models

import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from yolov5.models.experimental import attempt_load

def load_model(model_path):
    # Load the YOLO model checkpoint
    model = attempt_load(model_path)
    return model


# Function to retrieve video frames from local file
def retrieve_frames_from_video(video_path, frame_interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval == 0:
            frames.append(frame)
    cap.release()
    return frames

import cv2

def retrieve_frames_from_video(video_path, frame_interval, target_size=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)  # Resize frame to target size
        frame_count += 1
        if frame_count % frame_interval == 0:
            frames.append(frame)
    cap.release()
    return frames


# Function to download YouTube video and retrieve frames
def retrieve_frames_from_youtube(link, frame_interval):
    try:
        yt = YouTube(link)
        yt_stream = yt.streams.get_highest_resolution()
        video_url = yt_stream.url
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            video_bytes = BytesIO(response.content)
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes.getvalue())
            return retrieve_frames_from_video(video_path, frame_interval)
        else:
            st.error("Failed to download the YouTube video.")
            return []
    except Exception as e:
        st.error(f"An error occurred while downloading the YouTube video: {str(e)}")
        return []

# Perform object detection using your model
def detect_objects(model, frames):
    detections = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for frame in frames:
        # Preprocess the frame
        img = Image.fromarray(frame)
        img = transform(img).unsqueeze(0)
        # Perform inference
        with torch.no_grad():
            output = model.forward(img)  # Call model's forward method
        # Post-process the output if needed
        # For now, just appending the original frame without any detections
        detections.append(frame)
    return detections

def detect_objects(model, frames):
    detections = []
    for frame in frames:
        # Perform inference
        results = model(frame)
        # Draw bounding boxes on the frame
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = [int(i) for i in xyxy]
            class_name = model.names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Append the frame with bounding boxes to detections
        detections.append(frame)
    return detections
import torch

def detect_objects(model, frames):
    detections = []
    for frame in frames:
        # Convert frame to PyTorch tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        
        # Perform inference
        results = model(frame_tensor)
        
        # Process detection results
        for result in results.pred:
            # Extract bounding box coordinates, confidence scores, and class indices
            boxes = result[:, :4]  # Extract bounding box coordinates
            confidences = result[:, 4]  # Extract confidence scores
            class_indices = result[:, 5]  # Extract class indices

            # Draw bounding boxes on the frame
            for box, confidence, class_index in zip(boxes, confidences, class_indices):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(class_index)]
                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Append the frame with bounding boxes to detections
        detections.append(frame)
    
    return detections
def detect_objects(model, frames):
    detections = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for frame in frames:
        # Preprocess the frame
        img = Image.fromarray(frame)
        img = transform(img).unsqueeze(0)
        # Perform inference
        with torch.no_grad():
            results = model(img)  # Call model's forward method
        # Print or inspect the results object
        print(results)
        print(dir(results))
        print(type(results))
        # Post-process the output if needed
        # For now, just appending the original frame without any detections
        detections.append(frame)
    return detections





def main():
    st.title("Video Object Detection")
    video_url = st.text_input("Enter YouTube Video URL")
    if st.button("Process"):
        if video_url:
            # Load your model
            model = load_model("model.pt")
            st.write("Model loaded successfully")
            # Retrieve frames from YouTube URL
            frames = retrieve_frames_from_youtube(video_url, frame_interval=15)
            st.write(f"Number of frames retrieved: {len(frames)}")
            # Perform object detection
            detections = detect_objects(model, frames)
            st.write(f"Number of detections found: {len(detections)}")
            # Display results
            for i, detection in enumerate(detections):
                st.image(detection, caption=f"Detection {i+1}")
        else:
            st.warning("Please enter a YouTube video URL")

if __name__ == "__main__":
    main()
