# Streamlit frontend
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

API_URL = "http://localhost:8000/predict"

st.title("YOLO Object Detection")
st.write("Upload an image to perform object detection using YOLO.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Send the image to the API
    with st.spinner("Performing object detection..."):
        response = requests.post(API_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})

    if response.status_code == 200:
        detections = response.json().get("detections", [])
        st.success("Object detection completed!")

        # Draw bounding boxes and labels on the image
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=20)  # Ensure arial.ttf is accessible
        except IOError:
            font = ImageFont.load_default()

        for det in detections:
            xmin, ymin, xmax, ymax = det["xmin"], det["ymin"], det["xmax"], det["ymax"]
            label = det["name"]
            confidence = det["confidence"]

            # Draw bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

            # Add label and confidence
            draw.text((xmin, ymin), f"{label} ({confidence:.4f})", fill="red", font=font)

        # Display the image with bounding boxes
        st.image(image, caption="Detected Objects", use_container_width=True)
    else:
        st.error("Error during prediction. Please check the backend.")
        st.json(response.json())  # Show error details
