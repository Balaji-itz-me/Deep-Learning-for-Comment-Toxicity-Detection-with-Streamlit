import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import json
import pandas as pd
from sklearn.metrics import classification_report
from typing import List

# ----------------------------
# Load label list
with open("label_list.json", "r") as f:
    LABELS = json.load(f)

NUM_LABELS = len(LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")

# ----------------------------
# Define model class (same as training)
class BertMultilabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertMultilabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        return self.classifier(self.dropout(pooled))

# ----------------------------
# Load model
model = BertMultilabelClassifier(num_labels=NUM_LABELS)
model.load_state_dict(torch.load("bert_multilabel_best.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----------------------------
# Prediction function
def predict(text: str) -> dict:
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {label: float(np.round(p, 3)) for label, p in zip(LABELS, probs)}

# ----------------------------
# Streamlit App UI
st.set_page_config(page_title="Toxicity Detector", layout="centered")
st.title("🧠 Toxic Comment Classifier (BERT)")
st.markdown("Enter a comment below or upload a CSV file to detect toxicity.")

# Text input prediction
user_input = st.text_area("💬 Enter a comment here")

if st.button("Predict Toxicity"):
    if user_input.strip():
        result = predict(user_input)
        st.markdown("### 🔍 Prediction:")
        for label, score in result.items():
            emoji = "✅" if score > 0.5 else "❌"
            st.write(f"**{label}**: {emoji} {score}")
    else:
        st.warning("Please enter a comment.")

# Bulk CSV Prediction
st.markdown("---")
st.subheader("📁 Upload CSV for Bulk Predictions")
uploaded_file = st.file_uploader("Upload a CSV with a 'comment_text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'comment_text' not in df.columns:
        st.error("CSV must contain a 'comment_text' column.")
    else:
        with st.spinner("🔍 Predicting toxicity for each comment..."):
            preds = [predict(comment) for comment in df["comment_text"]]
            pred_df = pd.DataFrame(preds)
            result_df = pd.concat([df, pred_df], axis=1)

        st.success("✅ Predictions complete!")
        st.markdown("### 🔄 Preview of results:")
        st.dataframe(result_df.head())

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Prediction CSV",
            data=csv,
            file_name="toxic_predictions.csv",
            mime="text/csv"
        )


# ----------------------------
# Model Info
st.markdown("---")
st.markdown("ℹ️ **Model Info:**")
st.markdown("""
- Model: `BERT-base-uncased`
- Trained on: 6 toxicity labels (`toxic`, `severe_toxic`, etc.)
- ROC-AUC scores > 0.98 on average
- Fine-tuned using multi-label BCE loss
""")
