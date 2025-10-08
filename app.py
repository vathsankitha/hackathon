import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import io

# Function to load the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to predict depth
@st.cache_resource
def load_model():
    model_name = "Intel/dpt-large"  # Alternative: "LiheYoung/depth-anything-large-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# Streamlit app
st.title("Image Depth Estimation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    st.write("")
    st.write("Predicting depth...")

    # Prepare inputs and predict
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original sizestre
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Visualize depth map
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    fig_depth, ax_depth = plt.subplots(figsize=(10, 6))
    ax_depth.imshow(formatted, cmap="inferno")
    ax_depth.set_title("Depth Map")
    ax_depth.axis('off')
    st.pyplot(fig_depth)

    # Generate 3D mesh
    st.write("Generating 3D mesh...")
    x, y = np.meshgrid(np.arange(output.shape[1]), np.arange(output.shape[0]))
    fig_3d = go.Figure(data=[go.Surface(z=output, x=x, y=y, colorscale='Viridis')])
    fig_3d.update_layout(title='3D Mesh from Depth Map', autosize=False,
                      width=700, height=700,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig_3d)