import streamlit as st
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image

VIDEO_PATH = 'temp_video.mp4'

model = YOLO("model.pt")

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

# Initialize dictionaries to store class-wise detection counts
class_counts = {}
class_sum = {}
class_max = {}
num_frames_processed = 0

def process_frame(frame: np.ndarray, index) -> np.ndarray:
    global class_counts, class_sum, class_max, num_frames_processed

    # Process every 10th frame
    if index % 100 != 0:
        return frame

    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)

    num_frames_processed += 1

    # Print detections for the first frame only
    if index == 0:
        for i in range(len(detections.data['class_name'])):
            class_name = detections.data['class_name'][i]
            confidence = detections.confidence[i]
            bbox = detections.xyxy[i]

    # Update class-wise detection counts
    for i in range(len(detections.data['class_name'])):
        class_name = detections.data['class_name'][i]
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_sum[class_name] = 0
            class_max[class_name] = 0
        class_counts[class_name] += 1
        class_sum[class_name] += detections.confidence[i]
        class_max[class_name] = max(class_max[class_name], class_counts[class_name])

    # Annotate the frame with the detections and display it
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}" for i in range(len(detections.data['class_name']))]
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Convert the frame to PIL format for displaying in Streamlit
    annotated_frame_pil = Image.fromarray(annotated_frame)

    caption = f"Frame {index} - \n"
    class_counts_frame = {class_name: 0 for class_name in class_counts}
    for i in range(len(detections.data['class_name'])):
        class_name = detections.data['class_name'][i]
        class_counts_frame[class_name] += 1

    for class_name, count in class_counts_frame.items():
        caption += f"Class {class_name}: {count} detections\n"

    st.image(annotated_frame_pil, caption=caption, use_column_width=True)

    return frame


sv.process_video(source_path=VIDEO_PATH, target_path="result.mp4", callback=process_frame)

# Display class-wise statistics
for class_name in class_counts:
    st.write(f"Class: {class_name}, Detected {class_counts[class_name]} times")
    st.write(f"Average detections per frame: {class_sum[class_name] / num_frames_processed}")
    st.write(f"Maximum detections in a frame: {class_max[class_name]}")
