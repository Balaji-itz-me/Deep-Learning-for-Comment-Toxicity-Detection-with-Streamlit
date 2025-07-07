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

# Ultra-optimized bulk prediction with aggressive memory management
def predict_bulk_ultra_optimized(df, text_column, model, tokenizer, label_list, device, threshold=0.5, batch_size=8):
    """Ultra-optimized prediction with aggressive memory management"""
    results = []
    total_rows = len(df)
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    
    start_time = datetime.now()
    
    try:
        # Process in very small batches to avoid timeout
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            
            # Get batch texts
            batch_texts = []
            batch_indices = []
            
            for idx in range(batch_start, batch_end):
                try:
                    text = str(df.iloc[idx][text_column])
                    # Truncate very long texts to avoid memory issues
                    if len(text) > 500:
                        text = text[:500] + "..."
                    batch_texts.append(text)
                    batch_indices.append(idx)
                except:
                    continue
            
            if not batch_texts:
                continue
                
            try:
                # Tokenize with strict limits
                encoding = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=64,  # Reduced from 128
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Batch prediction
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask)
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                
                # Clear GPU memory immediately
                del input_ids, attention_mask, outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Process results
                for i, idx in enumerate(batch_indices):
                    text = batch_texts[i]
                    probs = probabilities[i]
                    
                    result = {'text': text, 'row_index': idx}
                    
                    # Add probabilities and predictions
                    for j, label in enumerate(label_list):
                        prob = float(probs[j])
                        result[label] = prob
                        result[f"{label}_predicted"] = prob > threshold
                    
                    results.append(result)
                
                # Clear memory
                del probabilities, encoding
                
            except Exception as e:
                st.warning(f"Skipped batch {batch_start}-{batch_end}: {str(e)}")
                continue
            
            # Update progress
            progress = (batch_end / total_rows)
            progress_bar.progress(progress)
            
            # Time estimation
            elapsed = (datetime.now() - start_time).total_seconds()
            if progress > 0:
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
                status_text.text(f"Processed {batch_end}/{total_rows} rows ({progress:.1%})")
                time_text.text(f"⏱️ Elapsed: {elapsed:.1f}s | Estimated remaining: {remaining:.1f}s")
            
            # Force garbage collection every 50 batches
            if batch_start % (batch_size * 50) == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        
    status_text.text("Processing complete!")
    time_text.text(f"Total time: {(datetime.now() - start_time).total_seconds():.1f}s")
    
    return pd.DataFrame(results)

# Emergency processing for very large files
def emergency_sample_processing(df, text_column, model, tokenizer, label_list, device, threshold=0.5, sample_size=5000):
    """Emergency processing with intelligent sampling"""
    
    st.warning(f"🚨 Emergency mode: Processing {sample_size} samples from {len(df)} rows")
    
    # Intelligent sampling strategy
    if len(df) > sample_size:
        # Take random sample + first/last rows for variety
        random_sample = df.sample(n=sample_size-200, random_state=42)
        first_rows = df.head(100)
        last_rows = df.tail(100)
        
        df_sample = pd.concat([first_rows, random_sample, last_rows]).drop_duplicates()
    else:
        df_sample = df
    
    st.info(f"📊 Processing {len(df_sample)} rows")
    
    return predict_bulk_ultra_optimized(df_sample, text_column, model, tokenizer, label_list, device, threshold, batch_size=4)

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
                # Show file info
                file_size = uploaded_file.size / (1024 * 1024)  # MB
                st.info(f"📁 File size: {file_size:.2f} MB")
                
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df):,} rows")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Select text column
                text_column = st.selectbox(
                    "Select the column containing text to analyze:",
                    df.columns.tolist()
                )
                
                # Processing strategy based on file size
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(df) > 20000:
                        st.error("🚨 Large file detected! Choose processing strategy:")
                        processing_strategy = st.selectbox(
                            "Processing Strategy:",
                            [
                                "Smart Sample (Recommended)",
                                "Ultra-Fast Sample (5K rows)",
                                "Conservative Full Processing",
                                "Emergency Mode (1K rows)"
                            ],
                            help="Smart sampling is recommended for large files"
                        )
                    else:
                        processing_strategy = "Standard Processing"
                        st.success("✅ File size is manageable for full processing")
                
                with col2:
                    if processing_strategy == "Smart Sample (Recommended)":
                        sample_size = st.slider("Sample size:", 1000, 20000, 10000, step=1000)
                        st.info(f"Will process {sample_size} intelligently selected rows")
                    elif processing_strategy == "Ultra-Fast Sample (5K rows)":
                        sample_size = 5000
                        st.info("Fast processing with 5K random sample")
                    elif processing_strategy == "Emergency Mode (1K rows)":
                        sample_size = 1000
                        st.info("Emergency processing with 1K sample")
                    else:
                        sample_size = len(df)
                        if len(df) > 50000:
                            st.warning("⚠️ This may take 20-30 minutes and could timeout")
                
                # Time and resource warnings
                if len(df) > 50000 and processing_strategy == "Conservative Full Processing":
                    st.error("❌ NOT RECOMMENDED: This will likely timeout on Streamlit Cloud")
                    st.write("Streamlit Cloud has 30-minute timeout limits")
                elif len(df) > 20000 and processing_strategy == "Conservative Full Processing":
                    st.warning("⚠️ May timeout. Consider using Smart Sample instead")
                
                # Batch size selection
                batch_size = st.selectbox(
                    "Batch size (smaller = more stable):",
                    [4, 8, 16],
                    index=1,
                    help="Smaller batches are more stable but slower"
                )
                
                if st.button("🚀 Analyze Texts", type="primary"):
                    start_time = datetime.now()
                    
                    # Choose processing method based on strategy
                    if processing_strategy == "Smart Sample (Recommended)":
                        st.info(f"🧠 Using Smart Sample: {sample_size} rows")
                        with st.spinner("Processing smart sample..."):
                            results_df = emergency_sample_processing(
                                df, text_column, model, tokenizer, label_list, 
                                device, threshold, sample_size
                            )
                    
                    elif processing_strategy == "Ultra-Fast Sample (5K rows)":
                        st.info("⚡ Using Ultra-Fast Sample: 5K rows")
                        df_sample = df.sample(n=5000, random_state=42)
                        with st.spinner("Processing ultra-fast sample..."):
                            results_df = predict_bulk_ultra_optimized(
                                df_sample, text_column, model, tokenizer, label_list, 
                                device, threshold, batch_size=4
                            )
                    
                    elif processing_strategy == "Emergency Mode (1K rows)":
                        st.info("🆘 Using Emergency Mode: 1K rows")
                        df_sample = df.sample(n=1000, random_state=42)
                        with st.spinner("Processing emergency sample..."):
                            results_df = predict_bulk_ultra_optimized(
                                df_sample, text_column, model, tokenizer, label_list, 
                                device, threshold, batch_size=4
                            )
                    
                    else:  # Conservative Full Processing
                        st.warning("🐌 Using Conservative Full Processing - This may timeout!")
                        with st.spinner("Processing all texts (this may take 20-30 minutes)..."):
                            results_df = predict_bulk_ultra_optimized(
                                df, text_column, model, tokenizer, label_list, 
                                device, threshold, batch_size=batch_size
                            )
                    
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    if not results_df.empty:
                        st.success(f"✅ Analysis complete! Processed {len(results_df):,} texts in {processing_time:.1f} seconds")
                        
                        # Show sampling info
                        if len(results_df) < len(df):
                            st.info(f"📊 Results based on {len(results_df):,} rows from {len(df):,} total rows ({(len(results_df)/len(df)*100):.1f}% sample)")
                        
                        # Performance metrics
                        texts_per_second = len(results_df) / processing_time
                        st.metric("Processing Speed", f"{texts_per_second:.1f} texts/second")
                        
                        # Summary statistics
                        st.subheader("📊 Analysis Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_texts = len(results_df)
                            st.metric("Total Texts", f"{total_texts:,}")
                        
                        with col2:
                            toxic_texts = sum(results_df[[f"{label}_predicted" for label in label_list]].any(axis=1))
                            st.metric("Potentially Toxic", f"{toxic_texts:,}")
                        
                        with col3:
                            safe_texts = total_texts - toxic_texts
                            st.metric("Safe Texts", f"{safe_texts:,}")
                        
                        with col4:
                            toxicity_rate = (toxic_texts / total_texts) * 100
                            st.metric("Toxicity Rate", f"{toxicity_rate:.1f}%")
                        
                        # Category breakdown
                        st.subheader("📈 Category Breakdown")
                        category_counts = {}
                        for label in label_list:
                            category_counts[label] = results_df[f"{label}_predicted"].sum()
                        
                        # Create metrics grid
                        metric_cols = st.columns(len(label_list))
                        for i, (label, count) in enumerate(category_counts.items()):
                            with metric_cols[i]:
                                percentage = (count / total_texts) * 100
                                st.metric(
                                    label.replace('_', ' ').title(),
                                    f"{count:,}",
                                    f"{percentage:.1f}%"
                                )
                        
                        # Visualizations
                        st.subheader("📊 Analysis Charts")
                        charts = create_bulk_analysis_charts(results_df, label_list)
                        
                        for chart in charts:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Most toxic texts
                        st.subheader("⚠️ Most Toxic Content")
                        
                        # Calculate overall toxicity score
                        results_df['overall_toxicity'] = results_df[label_list].max(axis=1)
                        most_toxic = results_df.nlargest(10, 'overall_toxicity')[
                            ['text', 'overall_toxicity'] + label_list
                        ]
                        
                        st.dataframe(most_toxic, use_container_width=True)
                        
                        # Detailed results with pagination
                        st.subheader("📋 Detailed Results")
                        
                        # Filter options
                        filter_option = st.selectbox(
                            "Filter results:",
                            ["All", "Toxic Only", "Safe Only", "High Confidence (>0.8)", "Low Confidence (<0.3)"]
                        )
                        
                        if filter_option == "Toxic Only":
                            filtered_df = results_df[results_df[[f"{label}_predicted" for label in label_list]].any(axis=1)]
                        elif filter_option == "Safe Only":
                            filtered_df = results_df[~results_df[[f"{label}_predicted" for label in label_list]].any(axis=1)]
                        elif filter_option == "High Confidence (>0.8)":
                            filtered_df = results_df[results_df[label_list].max(axis=1) > 0.8]
                        elif filter_option == "Low Confidence (<0.3)":
                            filtered_df = results_df[results_df[label_list].max(axis=1) < 0.3]
                        else:
                            filtered_df = results_df
                        
                        st.write(f"Showing {len(filtered_df):,} of {len(results_df):,} results")
                        
                        # Pagination for large results
                        if len(filtered_df) > 1000:
                            page_size = st.selectbox("Results per page:", [100, 500, 1000], index=1)
                            total_pages = (len(filtered_df) - 1) // page_size + 1
                            page = st.number_input("Page", 1, total_pages, 1)
                            
                            start_idx = (page - 1) * page_size
                            end_idx = min(start_idx + page_size, len(filtered_df))
                            display_df = filtered_df.iloc[start_idx:end_idx]
                            
                            st.write(f"Page {page} of {total_pages} (showing rows {start_idx+1}-{end_idx})")
                        else:
                            display_df = filtered_df
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download results
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"toxicity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("❌ No results generated. Please check your data and try again.")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error("Please ensure your CSV file is properly formatted and not corrupted.")
    
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
