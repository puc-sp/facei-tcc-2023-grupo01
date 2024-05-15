import streamlit as st
import numpy as np
from pytube import YouTube
import ssl
from PIL import Image
from io import BytesIO
import requests
import supervision as sv
from ultralytics import YOLO
ssl._create_default_https_context = ssl._create_unverified_context

st.sidebar.title("Otimizando o desenvolvimento infantil com tecnologia: reconhecimento de objetos em vídeos do YouTube")
st.sidebar.subheader("Alunas: Ana Beatriz O. de Macedo e Bruna Bellini Faria")
st.sidebar.markdown("<p style='text-align: justify;'>Este estudo apresenta uma plataforma de detecção resultante de uma metodologia de otimização no reconhecimento de objetos para crianças em estágios iniciais de desenvolvimento cognitivo, especialmente em ambientes digitais ricos em conteúdo de streaming, como vídeos do YouTube. O trabalho destaca a importância da diversidade de estímulos visuais no crescimento cognitivo infantil e propõe a implementação de uma rede neural específica, denominada You Only Look Once, para identificar eficientemente uma ampla gama de objetos em vídeos.</p>", unsafe_allow_html=True)
# Function to download YouTube video and retrieve frames
def retrieve_frames_from_youtube(link, frame_interval):
    try:
        yt = YouTube(link)
        yt_stream = yt.streams.get_highest_resolution()
        video_url = yt_stream.url
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            video_bytes = BytesIO(response.content)
            video_path = "temp_video2.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes.getvalue())
            return video_path
        else:
            st.error("Failed to download the YouTube video.")
            return [], ""
    except Exception as e:
        st.error(f"An error occurred while downloading the YouTube video: {str(e)}")
        return [], ""

VIDEO_PATH = ''

st.subheader("YouTube Video Object Detection")

# Input field for YouTube video URL
video_url = st.text_input("Insira a URL de um vídeo do YouTube:")
if st.button("Processar"):
    if video_url:
        VIDEO_PATH = retrieve_frames_from_youtube(video_url, 10)

model = YOLO("model.pt")

if VIDEO_PATH:
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

    # Initialize dictionaries to store class-wise detection counts
    class_counts = {}
    class_sum = {}
    class_max = {}
    num_frames_processed = 0

    def process_frame(frame: np.ndarray, index) -> np.ndarray:
        global class_counts, class_sum, class_max, num_frames_processed

        # Process every 10th frame
        if index % 70 != 0:
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
        # Update class-wise detection counts and maximum detections in a frame
        for i in range(len(detections.data['class_name'])):
            class_name = detections.data['class_name'][i]
            if class_name not in class_counts:
                class_counts[class_name] = 0
                class_sum[class_name] = 0
                class_max[class_name] = 0
            class_counts[class_name] += 1
            class_sum[class_name] += detections.confidence[i]
            class_max[class_name] = max(class_max[class_name], len(detections.data['class_name']))

        # Annotate the frame with the detections and display it
        box_annotator = sv.BoxAnnotator(thickness=3, text_thickness=3, text_scale=2)
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
        st.write(f"Average detections per frame: {round(class_sum[class_name] / num_frames_processed)}")
        st.write(f"Maximum detections in a frame: {class_max[class_name]}")
