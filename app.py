import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io  # ESSENTIAL: Used for handling file uploads correctly

# --- Configuration ---
MODEL_NAME = "Intel/dpt-large"  # High-quality depth estimation model

# --- Utility Functions ---


@st.cache_resource
def load_model():
    """Loads the pre-trained DPT model and processor."""
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Updated to accept z_exaggeration_factor


def predict_depth(image, processor, model, z_exaggeration_factor):
    """Processes image and performs depth prediction with Z-axis exaggeration."""

    # 1. Prepare inputs
    inputs = processor(images=image, return_tensors="pt")

    # 2. Predict depth
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # 3. Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # 4. Convert to numpy and normalize for 3D plotting
    output_raw = prediction.squeeze().cpu().numpy()

    # Invert and scale depth for Z-axis visualization
    depth_min = output_raw.min()
    depth_max = output_raw.max()
    normalized_depth = (output_raw - depth_min) / (depth_max - depth_min)

    # Scaling factor controls the base scale (was 20).
    # The new factor is passed by the user via the sidebar.
    base_scaling_factor = 5  # Use a smaller base, let the multiplier do the work

    # Apply inversion and user-defined exaggeration
    Z_mesh = (1 - normalized_depth) * \
        base_scaling_factor * z_exaggeration_factor

    return output_raw, Z_mesh

# Function to generate a truly textured 3D mesh using go.Mesh3d
# Updated to accept z_exaggeration_factor for accurate aspect ratio calculation


def generate_textured_3d_mesh(Z_data, original_image, z_exaggeration_factor):
    """
    Generates a Plotly 3D mesh plot (go.Mesh3d) textured with the original image colors.
    """

    H, W = Z_data.shape

    # 1. Create vertices (x, y, z coordinates for each pixel)
    x_coords = np.arange(W)
    y_coords = np.arange(H)

    X_flat = x_coords[np.newaxis, :].repeat(H, axis=0).flatten()
    Y_flat = y_coords[:, np.newaxis].repeat(W, axis=1).flatten()
    Z_flat = Z_data.flatten()

    # 2. Get vertex colors from the original image
    img_array = np.array(original_image.convert("RGB"))
    vertex_colors = img_array.reshape(-1, 3)

    # 3. Define faces (triangles)
    faces_i = []
    faces_j = []
    faces_k = []

    for r in range(H - 1):
        for c in range(W - 1):
            v00 = r * W + c
            v10 = (r + 1) * W + c
            v01 = r * W + (c + 1)
            v11 = (r + 1) * W + (c + 1)

            # Triangle 1
            faces_i.append(v00)
            faces_j.append(v10)
            faces_k.append(v01)

            # Triangle 2
            faces_i.append(v10)
            faces_j.append(v11)
            faces_k.append(v01)

    # Create the Mesh3d object
    fig = go.Figure(data=[
        go.Mesh3d(
            x=X_flat,
            y=Y_flat,
            z=Z_flat,
            i=faces_i,
            j=faces_j,
            k=faces_k,
            vertexcolor=vertex_colors,  # Assign the RGB colors to each vertex
            flatshading=False,
            # Enhanced lighting for better volume perception
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.8,
                          roughness=0.5, fresnel=0.01),
            # Light source from a corner
            lightposition=dict(x=1000, y=1000, z=1000),
            showscale=False
        )
    ])

    # 4. Final Layout Enhancements

    # Calculate a sensible Z-aspect ratio based on the total Z-range and the Z-exaggeration
    Z_range = Z_data.max() - Z_data.min()
    X_range = W
    Y_range = H

    # Aspect Z calculation: normalize Z-range relative to X-range, then apply a visual multiplier (e.g., 0.5)
    # The Z-exaggeration is implicitly handled by the Z_data itself.
    z_aspect = Z_range / X_range if X_range > 0 else 1.0

    fig.update_layout(
        title=f'3D Digital Twin (Exaggeration: {z_exaggeration_factor:.1f}x)',
        scene=dict(
            xaxis_title='Image Width (X)',
            yaxis_title='Image Height (Y)',
            zaxis_title=f'Relative Depth (Z) x{z_exaggeration_factor:.1f}',
            # Set aspect ratio dynamically to emphasize the Z-exaggeration
            aspectratio=dict(x=1, y=Y_range/X_range, z=z_aspect * 1.5),
            aspectmode='manual',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.75)
            )
        ),
        height=800,
        width=800
    )
    return fig


# --- Streamlit App (Main Execution) ---

st.set_page_config(layout="wide")

st.title("Vision Depth🔍")
st.markdown("---")

processor, model = load_model()

if processor is None or model is None:
    st.stop()

# --- Sidebar for User Control (The new interactive feature) ---
st.sidebar.header("3D Model Controls")

# The Z-Exaggeration Multiplier is the key to adding volume!
z_exaggeration_factor = st.sidebar.slider(
    "Z-Depth Exaggeration Multiplier",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="Increase this value to give the 3D model more visual volume and depth."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Z-Factor:** `{z_exaggeration_factor:.1f}x`")
# -------------------------------------------------------------------


st.header("1. Upload Image")
uploaded_file = st.file_uploader(
    "Choose an image (e.g., a room, landscape, or object)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    try:
        # --- Image Loading FIX ---
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

    except Exception as e:
        st.error(
            f"Error loading image: {e}. Please ensure the file is a valid JPG, JPEG, or PNG.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    st.markdown("---")
    st.header("2. Processing & Results")

    # --- Depth Estimation ---
    with st.spinner("Step 2a: Inferring Depth (Deep Learning Model)..."):
        try:
            # Pass the exaggeration factor to the prediction function
            output_raw, Z_mesh = predict_depth(
                image, processor, model, z_exaggeration_factor)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # --- Display Depth Map ---
    with col2:
        st.subheader("Relative Depth Map")
        formatted_vis = (output_raw * 255 / np.max(output_raw)).astype("uint8")

        fig_depth, ax_depth = plt.subplots(figsize=(10, 6))
        ax_depth.imshow(formatted_vis, cmap="magma")
        ax_depth.set_title("Brighter = Closer (Relative Depth)")
        ax_depth.axis('off')
        st.pyplot(fig_depth)

    st.markdown("---")

    # --- Generate and Display 3D Mesh ---
    st.header("3. Interactive 3D Digital Twin")

    with st.spinner("Step 2b: Generating Interactive Textured 3D Mesh..."):
        try:
            # Pass the exaggeration factor to the mesh generation function
            fig_3d = generate_textured_3d_mesh(
                Z_mesh, image, z_exaggeration_factor)
        except Exception as e:
            st.error(f"3D mesh generation failed: {e}")
            st.stop()

    st.plotly_chart(fig_3d, use_container_width=True)

    st.success(
        "🎉 Processing Complete! Use the **sidebar slider** to adjust the volume of the 3D model.")


