import streamlit as st
import config.mlflow
import traceback
from src.mlflow import load_model


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
# 2. Chargement du modèle Champion depuis MLflow
# ==================================================

@st.cache_resource
def load_modele():
    return load_model()


# ==================================================
# 3. Chargement du modèle
# ==================================================

try:

    model = load_modele()

    st.success(
        "Modèle Champion chargé avec succès depuis MLflow."
    )

except Exception as error:

    st.error("Erreur lors du chargement du modèle Champion.")

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

        st.image(
            uploaded_file,
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
                    source=uploaded_file.getvalue(),
                    conf=0.25,
                    save=False
                )

                result = results[0]

            # Afficher l'image avec les bounding boxes
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

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    confidence = float(box.conf[0])

                    class_name = result.names[class_id]

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
