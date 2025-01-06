import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

API_URL = "http://api-service.default.svc.cluster.local:8000/process"

st.title("YOLO Object Detection")
st.write("Upload an image to perform object detection using YOLO.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    with st.spinner("Performing object detection..."):
        response = requests.post(API_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})

    if response.status_code == 200:
        detections = response.json().get("detections", [])
        st.success("Object detection completed!")

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=30)
        except IOError:
            font = ImageFont.load_default()

        for det in detections:
            xmin, ymin, xmax, ymax = det["xmin"], det["ymin"], det["xmax"], det["ymax"]
            label = det["name"]
            confidence = det["confidence"]

            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)

            text = f"{label} ({confidence:.2f})"
            bbox = draw.textbbox((0, 0), text, font=font)  
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = xmin
            text_y = ymin - text_height - 5 
            if text_y < 0: 
                text_y = ymin + 5

            margin = 3
            draw.rectangle(
                [text_x - margin, text_y - margin, text_x + text_width + margin, text_y + text_height + margin],
                fill="red"
            )

            draw.text((text_x, text_y), text, fill="white", font=font)

        st.image(image, caption="Detected Objects", use_container_width=True)
    else:
        st.error("Error during prediction. Please check the backend.")
        st.json(response.json()) 
