import config.mlflow
import streamlit as st
import traceback
from src.mlflow import load_champion_model
from PIL import Image


# ==================================================
# 1. Configuration Streamlit
# ==================================================

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NeuroVision AI")
st.subheader("Détection des tumeurs cérébrales par IRM")

st.write(
    "Téléchargez une image IRM pour détecter "
    "et localiser les éventuelles tumeurs cérébrales."
)


# ==================================================
# 2. Chargement du modèle Champion
# ==================================================

@st.cache_resource
def load_modele():
    return load_champion_model()


# ==================================================
# 3. Chargement du modèle
# ==================================================

try:

    model = load_modele()

    st.success(
        "Modèle chargé avec succès."
    )

except Exception as error:

    st.error("Erreur lors du chargement du modèle.")

    st.exception(error)

    st.stop()


# ==================================================
# 4. Upload de l'image
# ==================================================

uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["jpg", "jpeg", "png", "bmp"]
)


# ==================================================
# 5. Prédiction
# ==================================================

if uploaded_file is not None:

    try:

        # Convertir UploadedFile en image PIL
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Image IRM sélectionnée",
            use_container_width=True
        )

    except Exception as error:

        st.error("Impossible de lire l'image.")

        st.exception(error)

        st.stop()

    if st.button(
        "Analyser l'image",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Détection en cours..."):

                # Effectuer la prédiction
                results = model.predict(
                    source=image,
                    conf=0.25,
                    save=False
                )

                result = results[0]

            # Image annotée avec les bounding boxes
            annotated_image = result.plot()

            st.subheader("Résultat de la détection")

            st.image(
                annotated_image,
                caption="Image avec les détections",
                channels="BGR",
                use_container_width=True
            )

            # Afficher les détections
            st.subheader("Tumeurs détectées")

            if len(result.boxes) > 0:

                for index, box in enumerate(result.boxes, start=1):

                    class_id = int(box.cls[0])

                    confidence = float(box.conf[0])

                    class_name = result.names[class_id]

                    st.write(f"### Détection {index}")

                    st.write(
                        f"**Classe :** {class_name}"
                    )

                    st.metric(
                        label="Confiance",
                        value=f"{confidence:.2%}"
                    )

                    st.divider()

            else:

                st.info("Aucune tumeur détectée.")

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant la prédiction."
            )

            st.exception(error)

            print(traceback.format_exc())


# ==================================================
# 6. Avertissement médical
# ==================================================

st.divider()

st.caption(
    "Outil expérimental à but éducatif. "
    "Les prédictions ne constituent pas un diagnostic médical."
)
