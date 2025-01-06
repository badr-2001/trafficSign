import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

# API endpoint pour le middleware
API_URL = "http://api-service.default.svc.cluster.local:8000/process"

st.title("YOLO Object Detection")
st.write("Upload an image to perform object detection using YOLO.")

# Chargement du fichier image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convertir l'image en bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Envoyer l'image à l'API
    with st.spinner("Performing object detection..."):
        response = requests.post(API_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})

    if response.status_code == 200:
        detections = response.json().get("detections", [])
        st.success("Object detection completed!")

        # Dessiner des boîtes de délimitation et des étiquettes sur l'image
        draw = ImageDraw.Draw(image)
        try:
            # Charger une police avec une taille plus grande
            font = ImageFont.truetype("arial.ttf", size=30)
        except IOError:
            # Utiliser la police par défaut si "arial.ttf" n'est pas trouvée
            font = ImageFont.load_default()

        for det in detections:
            xmin, ymin, xmax, ymax = det["xmin"], det["ymin"], det["xmax"], det["ymax"]
            label = det["name"]
            confidence = det["confidence"]

            # Dessiner la boîte avec des bordures plus épaisses
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)

            # Préparer le texte
            text = f"{label} ({confidence:.2f})"
            bbox = draw.textbbox((0, 0), text, font=font)  # Utiliser textbbox
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculer la position du texte (juste au-dessus du rectangle)
            text_x = xmin
            text_y = ymin - text_height - 5  # Ajuster l'espace entre le texte et la boîte
            if text_y < 0:  # Si le texte sort de l'image en haut, placez-le en bas
                text_y = ymin + 5

            # Dessiner un arrière-plan pour le texte
            margin = 3
            draw.rectangle(
                [text_x - margin, text_y - margin, text_x + text_width + margin, text_y + text_height + margin],
                fill="red"
            )

            # Ajouter le texte en blanc
            draw.text((text_x, text_y), text, fill="white", font=font)

        # Afficher l'image avec les prédictions
        st.image(image, caption="Detected Objects", use_container_width=True)
    else:
        st.error("Error during prediction. Please check the backend.")
        st.json(response.json())  # Afficher les détails de l'erreur
