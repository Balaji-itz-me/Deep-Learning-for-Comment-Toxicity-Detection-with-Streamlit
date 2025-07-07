import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import BertTokenizer, BertModel
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Toxicity Detection App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .toxic-prediction {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
    }
    .safe-prediction {
        background-color: #e6ffe6;
        border-left: 5px solid #44ff44;
    }
</style>
""", unsafe_allow_html=True)

# BERT Model Class
class BertMultiLabelClassifier(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-uncased'):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Cache model loading
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Load label list
        with open('label_list.json', 'r') as f:
            label_list = json.load(f)
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BertMultiLabelClassifier(n_classes=len(label_list))
        
        # Load state dict with compatibility handling
        state_dict = torch.load('bert_multilabel_best.pth', map_location=device)
        
        # Handle missing position_ids key for compatibility
        model_state = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state:
                # Check if shapes match
                if model_state[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    st.warning(f"Shape mismatch for {key}: expected {model_state[key].shape}, got {value.shape}")
            else:
                st.warning(f"Unexpected key in state_dict: {key}")
        
        # Add missing keys with default values
        for key in model_state:
            if key not in filtered_state_dict:
                filtered_state_dict[key] = model_state[key]
                st.info(f"Using default value for missing key: {key}")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model.to(device)
        
        return model, tokenizer, label_list, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check if all required files are present:")
        st.error("- bert_multilabel_best.pth")
        st.error("- label_list.json") 
        st.error("- bert_tokenizer/ directory with tokenizer files")
        return None, None, None, None

# Prediction function
def predict_toxicity(text, model, tokenizer, label_list, device, threshold=0.5):
    """Predict toxicity for a given text"""
    try:
        # Tokenize text
        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Create results
        results = {}
        predictions = {}
        
        for i, label in enumerate(label_list):
            prob = float(probabilities[i])
            results[label] = prob
            predictions[label] = prob > threshold
        
        return results, predictions, True
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, False

# Bulk prediction function
def predict_bulk(df, text_column, model, tokenizer, label_list, device, threshold=0.5):
    """Predict toxicity for multiple texts"""
    results = []
    
    progress_bar = st.progress(0)
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        text = str(row[text_column])
        probs, preds, success = predict_toxicity(text, model, tokenizer, label_list, device, threshold)
        
        if success:
            result = {'text': text, 'row_index': idx}
            result.update(probs)
            result.update({f"{label}_predicted": preds[label] for label in label_list})
            results.append(result)
        
        progress_bar.progress((idx + 1) / total_rows)
    
    return pd.DataFrame(results)

# Visualization functions
def create_probability_chart(results, label_list):
    """Create a bar chart of toxicity probabilities"""
    fig = go.Figure()
    
    colors = ['#ff4444' if prob > 0.5 else '#44ff44' for prob in results.values()]
    
    fig.add_trace(go.Bar(
        x=list(results.keys()),
        y=list(results.values()),
        marker_color=colors,
        text=[f'{prob:.3f}' for prob in results.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Toxicity Probability by Category",
        xaxis_title="Toxicity Categories",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400
    )
    
    return fig

def create_bulk_analysis_charts(df, label_list):
    """Create analysis charts for bulk predictions"""
    charts = []
    
    # Overall toxicity distribution
    toxic_counts = {}
    for label in label_list:
        toxic_counts[label] = df[f"{label}_predicted"].sum()
    
    fig1 = px.bar(
        x=list(toxic_counts.keys()),
        y=list(toxic_counts.values()),
        title="Number of Toxic Predictions by Category",
        labels={'x': 'Toxicity Categories', 'y': 'Count'}
    )
    charts.append(fig1)
    
    # Probability distributions
    fig2 = make_subplots(
        rows=2, cols=3,
        subplot_titles=label_list,
        specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*3]
    )
    
    for i, label in enumerate(label_list):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        fig2.add_trace(
            go.Histogram(x=df[label], name=label, nbinsx=20),
            row=row, col=col
        )
    
    fig2.update_layout(
        title="Probability Distributions by Category",
        showlegend=False,
        height=600
    )
    charts.append(fig2)
    
    return charts

# Main app
def main():
    st.markdown('<h1 class="main-header">🛡️ Toxicity Detection App</h1>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, label_list, device = load_model_and_tokenizer()
    
    if model is None:
        st.error("Failed to load model. Please check if all required files are present.")
        return
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.1)
    
    # Sample test cases
    sample_cases = {
        "Non-toxic": "I love spending time with my family and friends.",
        "Mild toxicity": "This is really annoying and frustrating.",
        "Moderate toxicity": "You're being stupid and ignorant about this.",
        "High toxicity": "I hate you and wish you would just disappear."
    }
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📊 Bulk Analysis", "📈 Model Insights", "🧪 Test Cases"])
    
    with tab1:
        st.header("Single Text Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter text to analyze:",
                height=100,
                placeholder="Type your message here..."
            )
            
            if st.button("🔍 Analyze Text", type="primary"):
                if user_input.strip():
                    with st.spinner("Analyzing text..."):
                        results, predictions, success = predict_toxicity(
                            user_input, model, tokenizer, label_list, device, threshold
                        )
                    
                    if success:
                        # Overall toxicity assessment
                        is_toxic = any(predictions.values())
                        max_prob = max(results.values())
                        
                        if is_toxic:
                            st.markdown(
                                f'<div class="prediction-box toxic-prediction">'
                                f'<h3>⚠️ Potentially Toxic Content Detected</h3>'
                                f'<p><strong>Highest Probability:</strong> {max_prob:.3f}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="prediction-box safe-prediction">'
                                f'<h3>✅ Content Appears Safe</h3>'
                                f'<p><strong>Highest Probability:</strong> {max_prob:.3f}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Detailed results
                        st.subheader("Detailed Analysis")
                        
                        # Create two columns for metrics
                        metrics_cols = st.columns(3)
                        for i, (label, prob) in enumerate(results.items()):
                            with metrics_cols[i % 3]:
                                status = "🔴" if predictions[label] else "🟢"
                                st.metric(
                                    f"{status} {label}",
                                    f"{prob:.3f}",
                                    f"{'Detected' if predictions[label] else 'Safe'}"
                                )
                        
                        # Probability chart
                        st.plotly_chart(
                            create_probability_chart(results, label_list),
                            use_container_width=True
                        )
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.subheader("📊 Quick Stats")
            if 'results' in locals():
                avg_prob = np.mean(list(results.values()))
                max_category = max(results.keys(), key=lambda x: results[x])
                
                st.metric("Average Probability", f"{avg_prob:.3f}")
                st.metric("Highest Risk Category", max_category)
                st.metric("Risk Score", f"{results[max_category]:.3f}")
    
    with tab2:
        st.header("Bulk CSV Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="CSV file should contain a column with text data to analyze"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} rows")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Select text column
                text_column = st.selectbox(
                    "Select the column containing text to analyze:",
                    df.columns.tolist()
                )
                
                if st.button("🚀 Analyze All Texts", type="primary"):
                    with st.spinner("Processing bulk predictions..."):
                        results_df = predict_bulk(df, text_column, model, tokenizer, label_list, device, threshold)
                    
                    if not results_df.empty:
                        st.success("Analysis complete!")
                        
                        # Summary statistics
                        st.subheader("📊 Analysis Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_texts = len(results_df)
                            st.metric("Total Texts", total_texts)
                        
                        with col2:
                            toxic_texts = sum(results_df[[f"{label}_predicted" for label in label_list]].any(axis=1))
                            st.metric("Potentially Toxic", toxic_texts)
                        
                        with col3:
                            safe_texts = total_texts - toxic_texts
                            st.metric("Safe Texts", safe_texts)
                        
                        with col4:
                            toxicity_rate = (toxic_texts / total_texts) * 100
                            st.metric("Toxicity Rate", f"{toxicity_rate:.1f}%")
                        
                        # Visualizations
                        st.subheader("📈 Analysis Charts")
                        charts = create_bulk_analysis_charts(results_df, label_list)
                        
                        for chart in charts:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Detailed results
                        st.subheader("📋 Detailed Results")
                        
                        # Filter options
                        filter_option = st.selectbox(
                            "Filter results:",
                            ["All", "Toxic Only", "Safe Only"]
                        )
                        
                        if filter_option == "Toxic Only":
                            filtered_df = results_df[results_df[[f"{label}_predicted" for label in label_list]].any(axis=1)]
                        elif filter_option == "Safe Only":
                            filtered_df = results_df[~results_df[[f"{label}_predicted" for label in label_list]].any(axis=1)]
                        else:
                            filtered_df = results_df
                        
                        st.dataframe(filtered_df)
                        
                        # Download results
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"toxicity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("Model Performance & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Model Information")
            st.write(f"**Model Type:** BERT Multi-Label Classifier")
            st.write(f"**Number of Classes:** {len(label_list)}")
            st.write(f"**Classes:** {', '.join(label_list)}")
            st.write(f"**Device:** {device}")
            st.write(f"**Current Threshold:** {threshold}")
        
        with col2:
            st.subheader("📊 Class Distribution")
            # Create a simple bar chart showing class names
            fig = px.bar(
                x=label_list,
                y=[1] * len(label_list),
                title="Available Toxicity Categories"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("ℹ️ How It Works")
        st.write("""
        This application uses a fine-tuned BERT model for multi-label toxicity classification. The model:
        
        1. **Tokenizes** input text using BERT tokenizer
        2. **Encodes** text into numerical representations
        3. **Classifies** text across multiple toxicity categories simultaneously
        4. **Outputs** probabilities for each category
        5. **Applies** threshold to determine final predictions
        
        The model can detect multiple types of toxicity in a single text, making it more comprehensive than binary classifiers.
        """)
        
        st.subheader("🔧 Threshold Impact")
        st.write(f"""
        Current threshold: **{threshold}**
        
        - **Lower threshold** (e.g., 0.3): More sensitive, catches more potential toxicity but may have false positives
        - **Higher threshold** (e.g., 0.7): More conservative, only flags highly confident toxic content
        - **Recommended**: 0.5 for balanced performance
        """)
    
    with tab4:
        st.header("Test Cases & Examples")
        
        st.subheader("🧪 Pre-defined Test Cases")
        
        for case_name, case_text in sample_cases.items():
            with st.expander(f"{case_name}: {case_text[:50]}..."):
                st.write(f"**Full text:** {case_text}")
                
                if st.button(f"Test: {case_name}", key=f"test_{case_name}"):
                    with st.spinner("Analyzing..."):
                        results, predictions, success = predict_toxicity(
                            case_text, model, tokenizer, label_list, device, threshold
                        )
                    
                    if success:
                        # Show results
                        is_toxic = any(predictions.values())
                        
                        if is_toxic:
                            st.error("⚠️ Potentially toxic content detected")
                        else:
                            st.success("✅ Content appears safe")
                        
                        # Show detailed probabilities
                        for label, prob in results.items():
                            status = "🔴" if predictions[label] else "🟢"
                            st.write(f"{status} **{label}**: {prob:.3f}")
        
        st.subheader("💡 Tips for Testing")
        st.write("""
        **Good test cases to try:**
        - Neutral/positive content
        - Mild negative sentiment
        - Strong negative language
        - Hate speech or discriminatory content
        - Threats or violent language
        - Profanity and offensive language
        
        **Note:** This model is designed to help identify potentially harmful content, but human review is always recommended for final decisions.
        """)

if __name__ == "__main__":
    main()
