import streamlit as st
import numpy as np
from pytube import YouTube
import ssl
from PIL import Image
from io import BytesIO
import requests
import supervision as sv
from ultralytics import YOLO
import random
import matplotlib.pyplot as plt
import time
ssl._create_default_https_context = ssl._create_unverified_context

st.sidebar.title("Otimizando o desenvolvimento infantil com tecnologia: reconhecimento de objetos em vídeos do YouTube")
st.sidebar.subheader("Alunas: Ana Beatriz O. de Macedo e Bruna Bellini Faria")
st.sidebar.markdown("<p style='text-align: justify;'>Este estudo apresenta uma plataforma de detecção resultante de uma metodologia de otimização no reconhecimento de objetos para crianças em estágios iniciais de desenvolvimento cognitivo, especialmente em ambientes digitais ricos em conteúdo de streaming, como vídeos do YouTube. O trabalho destaca a importância da diversidade de estímulos visuais no crescimento cognitivo infantil e propõe a implementação de uma rede neural específica, denominada You Only Look Once, para identificar eficientemente uma ampla gama de objetos em vídeos.</p>", unsafe_allow_html=True)
# Function to download YouTube video
def retrieve_video_from_youtube(link):
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
            return video_path
        else:
            st.error("Failed to download the YouTube video.")
            return None
    except Exception as e:
        st.error(f"An error occurred while downloading the YouTube video: {str(e)}")
        return None

model = YOLO("model.pt")

# Initialize dictionaries to store class-wise detection counts
class_counts = {}
class_sum = {}
class_max = {}
num_frames_processed = 0
annotated_frames = []

def process_frame(frame: np.ndarray, index) -> np.ndarray:
    global class_counts, class_sum, class_max, num_frames_processed, annotated_frames

    # Process every 10th frame
    if index % 100 != 0:
        return frame

    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)

    num_frames_processed += 1

    # Update class-wise detection counts
    class_counts_frame = {}
    for i in range(len(detections.data['class_name'])):
        class_name = detections.data['class_name'][i]
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_sum[class_name] = 0
            class_max[class_name] = 0
        class_counts[class_name] += 1
        class_sum[class_name] += detections.confidence[i]
        class_counts_frame[class_name] = class_counts_frame.get(class_name, 0) + 1

    for class_name, count in class_counts_frame.items():
        class_max[class_name] = max(class_max[class_name], count)

    # Annotate the frame with the detections and store it
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}" for i in range(len(detections.data['class_name']))]
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    annotated_frames.append(annotated_frame)

    return frame

st.title("YouTube Video Object Detection")

#carregamento = "_Importando o vídeo e fazendo detecções..._"


#def stream_data():
#    for word in carregamento.split(" "):
#        yield word + " "
#        time.sleep(0.02)

# User input for YouTube video URL
video_url = st.text_input("Insira a URL de um vídeo do YouTube:")
frame_interval = 10  # Adjust as needed
if st.button("Processar vídeo"):
    #st.write_stream(stream_data)
    st.caption('_Importando o vídeo e fazendo detecções..._')
    video_path = retrieve_video_from_youtube(video_url)
    if video_path:
        sv.process_video(source_path=video_path, target_path="result.mp4", callback=process_frame)

        # Select 9 random frames to display
        random_frames = random.sample(annotated_frames, min(9, len(annotated_frames)))

        # Plot 9 random frames in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for ax, frame in zip(axes.flatten(), random_frames):
            ax.imshow(frame)
            ax.axis('off')

        st.pyplot(fig)

        # Display class-wise statistics
        data = {
            "Class": [],
            "Total Detections": [],
            "Average Detections per Frame": [],
            "Maximum Detections in a Single Frame": []
        }

        for class_name in class_counts:
            data["Class"].append(class_name)
            data["Total Detections"].append(class_counts[class_name])
            data["Average Detections per Frame"].append(round(class_sum[class_name] / num_frames_processed, 2))
            data["Maximum Detections in a Single Frame"].append(class_max[class_name])

        st.dataframe(data)
        for class_name in class_counts:
            st.write(f"Class: {class_name}, Detected {class_counts[class_name]} times")
            st.write(f"Average detections per frame: {round(class_sum[class_name] / num_frames_processed)}")
            st.write(f"Maximum detections in a single frame: {class_max[class_name]}")
