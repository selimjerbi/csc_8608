import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="TP1 – SAM Segmentation", layout="wide")

st.title("TP1 – Mini-dataset viewer")
st.write("Sélectionnez une image dans `TP1/data/images/` et affichez-la.")

img_dir = Path("TP1/data/images")

if not img_dir.exists():
    st.error(f"Dossier introuvable : {img_dir.resolve()}")
else:
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    if len(images) == 0:
        st.warning("Aucune image trouvée dans TP1/data/images/")
    else:
        st.success(f"{len(images)} image(s) détectée(s)")

        selected = st.selectbox(
            "Choisissez une image",
            images,
            format_func=lambda p: p.name
        )

        img = Image.open(selected).convert("RGB")
        st.image(img, caption=selected.name, use_container_width=True)
