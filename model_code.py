# model_code.py
"""
Multimodal KG + Spatial Reasoning + RAG Q&A
YOLOv8 + SE-C2f Novelty + BLIP + spaCy + Zephyr-7B

Note: We removed Transformers CLIP to avoid meta-tensor issues on CPU
and instead use a simple string-based alignment between YOLO labels and caption.
"""

import os
import torch
import numpy as np
import networkx as nx
from PIL import Image

import spacy
from ultralytics import YOLO
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)
from sklearn.metrics.pairwise import cosine_similarity  # still imported if you extend later

# ---------------------------------------------------------
# Disable torch._dynamo (avoid meta tensor shenanigans)
# ---------------------------------------------------------
try:
    import torch._dynamo
    torch._dynamo.disable()
except Exception:
    pass

# ---------------------------------------------------------
# 1. Inject SE Attention into FIRST C2f (novelty-safe)
# ---------------------------------------------------------
from ultralytics.nn.modules.block import C2f
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

_original_forward = C2f.forward
_SE_PATCH_APPLIED = False

def se_forward(self, x):
    """Apply SE only to the first-ever C2f module created."""
    global _SE_PATCH_APPLIED
    y = _original_forward(self, x)

    if not _SE_PATCH_APPLIED:
        self.se = SEBlock(y.shape[1])
        y = self.se(y)
        _SE_PATCH_APPLIED = True

    return y

C2f.forward = se_forward

# ---------------------------------------------------------
# 1b. Disable Ultralytics fuse() globally (avoid meta tensors)
# ---------------------------------------------------------
from ultralytics.nn.tasks import DetectionModel

def _no_fuse(self, verbose=False):
    """Disable fuse() to avoid meta tensor crash."""
    return self

DetectionModel.fuse = _no_fuse


# ---------------------------------------------------------
# 2. Spatial Relation Utility
# ---------------------------------------------------------
def spatial_relation(boxA, boxB):
    (xa1, ya1, xa2, ya2) = boxA
    (xb1, yb1, xb2, yb2) = boxB

    ax = (xa1 + xa2) / 2
    ay = (ya1 + ya2) / 2
    bx = (xb1 + xb2) / 2
    by = (yb1 + yb2) / 2

    relations = []

    if ax < bx - 40:
        relations.append("left_of")
    if ax > bx + 40:
        relations.append("right_of")
    if ay < by - 40:
        relations.append("above")
    if ay > by + 40:
        relations.append("below")

    # Nearness
    dist = np.sqrt((ax - bx)**2 + (ay - by)**2)
    if dist < 180:
        relations.append("near")

    return relations


# ---------------------------------------------------------
# 3. Simple label–caption alignment score (no CLIP)
# ---------------------------------------------------------
def label_caption_alignment(label: str, caption: str) -> float:
    """
    Very simple semantic alignment:
    - lowercase
    - split into tokens
    - overlap between label tokens and caption tokens
    Returns a score in [0, 1].
    """
    label_tokens = set(label.lower().replace("_", " ").split())
    cap_tokens = set(caption.lower().replace(",", " ").replace(".", " ").split())
    if not label_tokens:
        return 0.0
    overlap = label_tokens.intersection(cap_tokens)
    return len(overlap) / len(label_tokens)


# ---------------------------------------------------------
# 4. KG Builder
# ---------------------------------------------------------
def build_kg_for_image(image_path):
    """
    Build a multimodal KG with:
      - visual nodes
      - text nodes
      - spatial relations
      - semantic relations (string-based)
    Also returns a RAG-ready context for Q&A.
    """

    # ---------------- spaCy ----------------
    nlp = spacy.load("en_core_web_sm")

    # ---------------- BLIP ----------------
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # ---------------- YOLOv8 (SE-patched) ----------------
    yolo_model = YOLO("yolov8s.pt")  # using SE-modified backbone via C2f.forward

    results = yolo_model(image_path)
    boxes = results[0].boxes

    visual_entities = []
    yolo_boxes = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls = int(b.cls.cpu())
            conf = float(b.conf.cpu())
            label = yolo_model.names[cls]

            xyxy = b.xyxy[0].cpu().numpy()
            yolo_boxes.append((label, xyxy))

            visual_entities.append((label, conf))

    # ---------------- BLIP Caption ----------------
    raw_image = Image.open(image_path).convert("RGB")
    blip_inputs = blip_processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**blip_inputs, max_length=30)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # ---------------- spaCy Text Entities ----------------
    doc = nlp(caption)
    text_entities = [ent.text for ent in doc.ents] if doc.ents else []

    # ---------------- Alignment Scores (string-based) ----
    align_scores = []
    for (label, _c) in visual_entities:
        score = label_caption_alignment(label, caption)
        align_scores.append(score)

    # ---------------- Build KG ----------------
    G = nx.DiGraph()

    # Visual nodes
    for label, conf in visual_entities:
        G.add_node(label, type="visual", confidence=conf)

    # Text nodes
    for ent in text_entities:
        G.add_node(ent, type="text")

    # Caption node
    caption_node = f"caption: {caption}"
    G.add_node(caption_node, type="caption")

    # Visual → caption
    for label, _c in visual_entities:
        G.add_edge(label, caption_node, relation="mentioned_in")

    # Text → caption
    for ent in text_entities:
        G.add_edge(ent, caption_node, relation="entity_in_caption")

    # Spatial relations
    for i in range(len(yolo_boxes)):
        objA, boxA = yolo_boxes[i]
        for j in range(i + 1, len(yolo_boxes)):
            objB, boxB = yolo_boxes[j]
            for rel in spatial_relation(boxA, boxB):
                G.add_edge(objA, objB, relation=rel)

    # Caption semantic relations (simple heuristic)
    lower_cap = caption.lower()
    if "room" in lower_cap:
        for lbl, _c in visual_entities:
            if any(k in lbl.lower() for k in ["tv", "television", "chair", "table", "fireplace", "plant"]):
                G.add_edge(lbl, "room", relation="part_of")

    # Alignment relations
    for (label, _c), score in zip(visual_entities, align_scores):
        if score > 0.0:
            G.add_edge(label, caption_node, relation="lexical_aligned")

    # Relation summary
    relation_summary = "\n".join(
        f"{u} --{d['relation']}--> {v}" for u, v, d in G.edges(data=True)
    )

    # Metrics
    metrics = {
        "num_detections": len(visual_entities),
        "mean_confidence": float(np.mean([c for _, c in visual_entities])) if visual_entities else 0.0,
        "mean_alignment_score": float(np.mean(align_scores)) if align_scores else 0.0,
        "caption_length": len(caption.split()),
        "caption": caption,
    }

    # RAG context (Option B: KG + caption + detections)
    qa_context = f"""
IMAGE CAPTION:
{caption}

DETECTED OBJECTS (label, confidence):
{visual_entities}

TEXT ENTITIES FROM CAPTION:
{text_entities}

GRAPH RELATIONSHIPS:
{relation_summary}
"""

    return G, visual_entities, metrics, qa_context


# ---------------------------------------------------------
# 5. Zephyr-based RAG Question Answering
# ---------------------------------------------------------
def answer_question(question: str, context: str) -> str:
    """
    RAG Q&A using Zephyr-7B via chat_completion (universal fallback).
    """
    from huggingface_hub import InferenceClient

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return "⚠ HuggingFace token missing. Set HUGGINGFACEHUB_API_TOKEN."

    # Zephyr works in chat mode
    client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=token)

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system",
                 "content": (
                    "You are an assistant that answers ONLY using the provided "
                    "knowledge graph and caption context. If uncertain, reply: "
                    "'I cannot determine from the given image.'"
                 )},{
            "role": "user",
            "content": f"""
Use ONLY the following context to answer the question.
Do NOT include any structured tags (no QUESTION:, ANSWER:, INST:, ASS:, XML, JSON).
Respond in plain English only.

Context:
{context}

Question:
{question}

Give a short answer in 1–3 natural sentences.
"""
        }
    ],
            max_tokens=200,
            temperature=0.3,
            top_p=0.9,
        )

        # Extract generated answer
        if "choices" in response:
            return response["choices"][0]["message"]["content"].strip()

        return str(response)

    except Exception as e:
        return f"Error calling HF API: {e}"
