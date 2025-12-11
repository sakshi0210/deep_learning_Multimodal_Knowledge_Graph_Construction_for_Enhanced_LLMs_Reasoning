import os
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from model_code import build_kg_for_image, answer_question

st.set_page_config(page_title="Multimodal KG + RAG", layout="wide")

st.title("Multimodal Knowledge Graph Builder & RAG Q&A")
st.write("YOLOv8 + SE-C2f + BLIP + spaCy + Zephyr-7B (KG + caption + detections)")

# -----------------------------------------------------
# Image folder
# -----------------------------------------------------
IMAGE_FOLDER = "C:/Users/User/Downloads/dl/val2017"

if not os.path.exists(IMAGE_FOLDER):
    st.error(f"Folder not found: {IMAGE_FOLDER}")
    st.stop()

images = [f for f in os.listdir(IMAGE_FOLDER)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not images:
    st.error("No images found in folder.")
    st.stop()

st.sidebar.header("Image Selection")
selected = st.sidebar.selectbox("Choose an image:", images)
path = os.path.join(IMAGE_FOLDER, selected)

st.sidebar.image(path, caption=selected)

# -----------------------------------------------------
# Run pipeline
# -----------------------------------------------------
st.subheader("Running Multimodal Pipeline...")
G, visual_entities, metrics, qa_context = build_kg_for_image(path)

col1, col2 = st.columns([1, 1])

# LEFT COLUMN: Image + detections + metrics
with col1:
    st.subheader("Input Image")
    st.image(Image.open(path), use_container_width=True)

    st.subheader("🟦 Detected Objects")
    if visual_entities:
        for label, conf in visual_entities:
            st.write(f"- **{label}** ({conf:.3f})")
    else:
        st.write("⚠ No detections")

    st.subheader("📊 Metrics")
    st.write(f"**Mean YOLO Confidence:** {metrics['mean_confidence']:.3f}")
    st.write(f"**Mean Label–Caption Alignment:** {metrics['mean_alignment_score']:.3f}")
    st.write(f"**Caption Length:** {metrics['caption_length']} words")

    st.subheader("📝 BLIP Caption")
    st.write(metrics["caption"])

# RIGHT COLUMN: KG + Q&A
with col2:
    st.subheader("🔗 Knowledge Graph")

    if len(G.nodes) == 0:
        st.write("KG is empty.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)

        # Color nodes by type
        colors = []
        for n, data in G.nodes(data=True):
            t = data.get("type", "")
            if t == "visual":
                colors.append("#8ecae6")
            elif t == "text":
                colors.append("#bde0fe")
            elif t == "caption":
                colors.append("#ffb703")
            else:
                colors.append("#cccccc")

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=colors,
            node_size=2500,
            font_size=9,
            font_weight="bold",
            ax=ax,
        )
        st.pyplot(fig)

    st.subheader("❓ Ask a Question About the Image & Graph")
    question = st.text_input("Your question:")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Reasoning with KG + caption + detections..."):
                answer = answer_question(question, qa_context)
            st.markdown("**Answer:**")
            st.write(answer)
