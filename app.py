import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gdown

# Function to download the dataset if not already available
def download_data():
    file_id = '1E4W1RvNGgyawc6I4TxQk76n289FX9kCK'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, 'dataset social media.xlsx', quiet=False)

# Check if dataset exists, if not, download it
import os
if not os.path.exists('dataset social media.xlsx'):
    download_data()

# ===============================
# === INISIALISASI ANALYZER  ====
# ===============================
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()

# ===============================
# === SETUP & PERSIAPAN DATA ====
# ===============================

@st.cache_data
def load_data():
    df = pd.read_excel('dataset social media.xlsx', sheet_name='Working File')
    # Cleaning kolom utama
    for col in ['Platform', 'Post Type', 'Audience Gender', 'Age Group', 'Sentiment', 'Time Periods', 'Weekday Type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    # Drop kolom tidak relevan
    drop_cols = [
        'Post ID', 'Date', 'Time', 'Audience Location', 'Audience Continent',
        'Audience Interests', 'Campaign ID', 'Influencer ID'
    ]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    # Konversi timestamp dan fitur waktu
    df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'], errors='coerce')
    df = df.dropna(subset=['Post Timestamp'])
    df['Post Hour'] = df['Post Timestamp'].dt.hour
    df['Post Day Name'] = df['Post Timestamp'].dt.day_name()
    if 'Weekday Type' in df.columns:
        df = df.drop(columns=['Weekday Type'])
    return df

df = load_data()

# ===============================
# === FUNGSI UTAMA ANALISIS  ====
# ===============================

def analyze_sentiment(caption):
    score = vader_analyzer.polarity_scores(caption)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def apply_engagement_rate_formatting(df):
    df['Engagement Rate'] = (df['Engagement Rate'] / 1000).round(4)  # Membagi dengan 1000
    df['Engagement Rate'] = df['Engagement Rate'].clip(0.01, 1.00) * 100  # Membatasi antara 1% dan 100%
    df['Engagement Rate'] = df['Engagement Rate'].astype(str) + '%'
    return df

# ===============================
# === ANTARMUKA STREAMLIT APP ===
# ===============================

st.markdown(
    """
    <div style='text-align: center;'>
        <span style="font-size:3em;">📊</span><br>
        <span style="font-size:1.8em; font-weight: bold;">Social Media Caption & Posting Analytics</span><br>
        <span style="font-size:1.2em; color:gray;">Boost Your Engagement with Smart Caption Analysis and Optimal Posting Times</span><br><br>
        <!-- Logo Row -->
        <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; max-width: 100%; overflow: hidden;">
            <div>
                <img src="https://github.com/error404-sudo/NewCapstone/raw/main/X.png" width="100" />
                <p>X</p>
            </div>
            <div>
                <img src="https://github.com/error404-sudo/NewCapstone/raw/main/linkedin.png" width="100" />
                <p>LinkedIn</p>
            </div>
            <div>
                <img src="https://github.com/error404-sudo/NewCapstone/raw/main/instagram.png" width="100" />
                <p>Instagram</p>
            </div>
            <div>
                <img src="https://github.com/error404-sudo/NewCapstone/raw/main/facebook.png" width="100" />
                <p>Facebook</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form(key='input_form'):
    caption_input = st.text_area("Enter Your Caption:")
    post_type_input = st.selectbox("Select Post Type", sorted(df['Post Type'].unique()))
    audience_gender_input = st.selectbox("Select Audience Gender", sorted(df['Audience Gender'].unique()))
    age_group_input = st.selectbox("Select Age Group", sorted(df['Age Group'].unique()))
    platforms = list(sorted(df['Platform'].unique())) + ['All']
    platform_input = st.selectbox("Select Platform", platforms)
    submit_button = st.form_submit_button(label='Check Recommendations')

if submit_button:
    sentiment_result = analyze_sentiment(caption_input)
    st.success(f"Predicted Sentiment for Your Caption: **{sentiment_result}**")
    
    # Recommendation Pipeline
    reco_pipeline, warning_pipeline = hybrid_recommendation_pipeline_super_adaptive(
        post_type_input,
        audience_gender_input,
        age_group_input,
        sentiment_result,
        platform_input
    )
    
    # Display recommendation and other results
    if not reco_pipeline.empty:
        best_reco = reco_pipeline.iloc[0]
        reco_text = f"Post at **{int(best_reco['Post Hour']):02d}:00 WIB** on **{best_reco['Post Day Name']}** for maximum engagement."
        st.markdown(f"### ⏰ Posting Time Recommendation\n{reco_text}")
        with st.expander("View Top 5 Recommendations"):
            st.dataframe(reco_pipeline.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.error("No relevant recommendations based on your input.")

    if warning_pipeline:
        st.warning(warning_pipeline)
    
    # Content Strategy and Alternative Platforms
    strategy_reco = strategy_recommendation(post_type_input, audience_gender_input, age_group_input)
    if not strategy_reco.empty:
        st.markdown(f"### 🎯 Content Strategy\n{strategy_reco.iloc[0]}")
    alt_platform_reco = alternative_platform_suggestion(post_type_input, audience_gender_input, age_group_input, platform_input)
    if not alt_platform_reco.empty:
        st.markdown(f"### 🔄 Alternative Platform Suggestions\n{alt_platform_reco.iloc[0]}")

st.markdown(f"**Engagement Rate Model - RMSE:** {rmse:.3f} | **MAE:** {mae:.3f} | **R2:** {r2:.4f}")
