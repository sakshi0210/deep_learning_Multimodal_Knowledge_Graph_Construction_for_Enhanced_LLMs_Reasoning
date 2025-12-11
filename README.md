# Multimodal Knowledge Graph Builder with SE-Enhanced YOLOv8 and RAG-based Question Answering

A deep learning project that constructs a **Multimodal Knowledge Graph (MMKG)** from images by integrating object detection, image captioning, named entity recognition, and retrieval-augmented generation for explainable question answering.

---

## Project Overview

This system combines:
- **SE-enhanced YOLOv8** for object detection with channel-wise attention
- **BLIP** for image captioning
- **spaCy** for named entity recognition (NER)
- **NetworkX** for knowledge graph construction
- **Zephyr-7B** (via Hugging Face) for RAG-based Q&A
- **Streamlit** for interactive visualization and querying

The pipeline transforms raw images into structured multimodal knowledge graphs, enabling grounded, interpretable question answering about visual content.

---

## Key Features

- **Novel SE-C2f Enhancement**: Squeeze-and-Excitation attention integrated into the first C2f layer of YOLOv8 for improved early feature extraction
- **Multimodal KG Construction**: Fuses visual detections, spatial relations, and textual entities into a unified graph
- **RAG-based QA**: Uses KG context with Zephyr-7B to answer natural language questions with controlled hallucination
- **Interactive Interface**: Streamlit-based UI for image upload, KG visualization, and Q&A
- **Explainable Reasoning**: Transparent graph-based reasoning connecting vision and language

---

## Architecture
https://github.com/user-attachments/assets/e90af0cd-3fdb-4c29-9ff4-6c702ca78804



---

## Repository Structure

```
├── app.py                 # Streamlit application
├── model_code.py          # Core pipeline implementation
├── DL_report.pdf          # Detailed project report
├── DL_PRESENTATIONFINAL_(3).pptx  # Presentation slides
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mmkg-builder.git
cd mmkg-builder
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics transformers spacy streamlit networkx matplotlib pillow
pip install huggingface-hub scikit-learn
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Set Up Hugging Face Token
```bash
export HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"
```
### 6. We are using COCO128 for testing 
---

## Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Application Interface
The interface provides:
- **Image Selection**: Choose from COCO dataset or upload custom images
- **Object Detection**: Displays detected objects with confidence scores
- **Knowledge Graph Visualization**: Interactive graph with color-coded nodes
- **Q&A Module**: Ask natural language questions about the image
- **Metrics Dashboard**: Shows performance metrics and caption information

### Example Workflow
1. Upload an image or select from dataset
2. View detected objects and generated caption
3. Explore the multimodal knowledge graph
4. Ask questions like:
   - "What objects are in the image?"
   - "Where is the person relative to the car?"
   - "Describe the scene."
https://github.com/user-attachments/assets/8e8c1873-ec39-43e5-b32c-b1976b664de4


---

## Methodology Details

### SE-C2f Enhancement
- Squeeze-and-Excitation block added to first C2f layer only
- Channel-wise attention improves feature saliency
- Maintains architectural stability while enhancing early feature processing

### Knowledge Graph Construction
- **Visual Nodes**: Detected objects with bounding boxes and confidence
- **Text Nodes**: Named entities extracted from captions
- **Spatial Edges**: Geometric relationships (left_of, right_of, above, below, near)
- **Semantic Edges**: Entity-caption connections
- **Alignment Edges**: Lexical similarity between visual and text nodes

### RAG Pipeline
- KG triples used as retrieval context
- Zephyr-7B generates answers grounded in visual context
- Prompt engineering based on query type
- Hallucination control through confidence thresholds

---

---

## Novel Contributions

1. **Selective SE Integration**: Novel application of SE attention to only the first C2f layer in YOLOv8
2. **Multimodal KG Pipeline**: End-to-end system combining vision and language for structured reasoning
3. **KG-Guided RAG**: Using constructed graphs as context for hallucination-controlled Q&A
4. **Interactive Exploration**: Streamlit interface for real-time visualization and querying

---

## Limitations

- Lexical alignment instead of semantic embeddings
- Single-layer SE application (not full backbone)
- API dependency for Zephyr-7B
- Static images only (no video support)
- Fixed spatial relation thresholds

---

## Future Work

1. **Semantic Alignment**: Integrate CLIP or LLaVA embeddings
2. **Architectural Improvements**: Apply SE across all C2f blocks
3. **Video Processing**: Support for temporal reasoning
4. **Local LLM**: Offline deployment with Llama.cpp or Ollama
5. **Ontology Expansion**: Integrate WordNet/ConceptNet

---

## References

1. Liu et al. (2025) - "Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction."
2. Radford et al. (2021) - CLIP: Contrastive Language-Image Pre-training
3. Li et al. (2022) - BLIP: Bootstrapping Language-Image Pre-training
4. Jocher et al. (2023) - YOLOv8: Real-time object detection
5. Hu et al. (2018) - Squeeze-and-Excitation Networks
6. Lewis et al. (2020) - Retrieval-Augmented Generation

---

## Authors

- **Sakshi Vispute** (252IT032) - sakshijitendravispute.252it032@nitk.edu.in
- **Samriddhi Sharma** (252IT024) - samriddhisharma.252it024@nitk.edu.in

**Guide**: Dr. Dinesh Naik - dineshnaik@nitk.edu.in  
**Department of Information Technology**  
**National Institute of Technology Karnataka, Surathkal**

---

## License

This project was developed as part of the Deep Learning course (IT702) at NITK Surathkal.  
All rights reserved by the authors.

---

## Acknowledgments

We thank the Department of Information Technology, NITK Surathkal for computational resources and support. Special thanks to Dr. Dinesh Naik for guidance. We also acknowledge the open-source communities behind YOLOv8, BLIP, spaCy, Hugging Face, and COCO dataset.

---
