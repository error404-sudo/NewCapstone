import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# === INISIALISASI ANALYZER  ====
# ===============================

nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()

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
    df['Engagement Rate'] = df['Engagement Rate'].astype(str) + '%'  # Menambahkan tanda persen
    return df

def hybrid_recommendation_pipeline_super_adaptive(post_type, audience_gender, age_group, sentiment=None, platform_input=None):
    warning_text = ""
    filtered = df[ 
        (df['Post Type'] == post_type) & 
        (df['Audience Gender'] == audience_gender) & 
        (df['Age Group'] == age_group)
    ]
    if sentiment:
        filtered_sent = filtered[filtered['Sentiment'] == sentiment]
    else:
        filtered_sent = filtered

    if not platform_input or platform_input.lower() == 'all':
        group_cols = ['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']
        filtered_sent_platform = filtered_sent
    else:
        filtered_sent_platform = filtered_sent[filtered_sent['Platform'] == platform_input.title()]
        group_cols = ['Time Periods', 'Post Day Name', 'Post Hour']

    main_reco = (
        filtered_sent_platform.groupby(group_cols)
        .agg({'Engagement Rate': 'mean'})
        .sort_values('Engagement Rate', ascending=False)
        .reset_index()
    )
    main_reco = apply_engagement_rate_formatting(main_reco)  # Format Engagement Rate

    return main_reco.head(5), warning_text

# ===============================
# === ANTARMUKA STREAMLIT APP ===
# ===============================

st.markdown("""
<div style='text-align: center;'>
    <span style="font-size:3em;">📊</span><br>
    <span style="font-size:1.8em; font-weight: bold;">Social Media Caption & Posting Analytics</span><br>
    <span style="font-size:1.2em; color:gray;">Boost Your Engagement with Smart Caption Analysis and Optimal Posting Times</span>
</div>
""", unsafe_allow_html=True)

with st.form(key='input_form'):
    caption_input = st.text_area("Enter Your Caption:")
    post_type_input = st.selectbox("Select Post Type", sorted(df['Post Type'].unique()))
    audience_gender_input = st.selectbox("Select Audience Gender", sorted(df['Audience Gender'].unique()))
    age_group_input = st.selectbox("Select Age Group", sorted(df['Age Group'].unique()))
    platform_input = st.selectbox("Select Platform", sorted(df['Platform'].unique()))
    submit_button = st.form_submit_button(label='Check Recommendations')

if submit_button:
    sentiment_result = analyze_sentiment(caption_input)
    st.success(f"Predicted Sentiment for Your Caption: **{sentiment_result}**")

    reco_pipeline, warning_pipeline = hybrid_recommendation_pipeline_super_adaptive(
        post_type_input, 
        audience_gender_input, 
        age_group_input, 
        sentiment_result, 
        platform_input
    )
    
    if not reco_pipeline.empty:
        best_reco = reco_pipeline.iloc[0]
        reco_text = f"Post at **{int(best_reco['Post Hour']):02d}:00 WIB** on **{best_reco['Post Day Name']}** using platform **{best_reco['Platform']}**."
        st.markdown(f"### ⏰ Posting Time Recommendation\n{reco_text}")
        st.dataframe(reco_pipeline)
    else:
        st.error("No relevant recommendations based on your input.")
