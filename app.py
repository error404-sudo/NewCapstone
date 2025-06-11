import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- Cek dan Download Data dari Google Drive (hanya jika belum ada) ---
DATA_FILE = 'dataset social media.xlsx'
FILE_ID = '1E4W1RvNGgyawc6I4TxQk76n289FX9kCK'
URL = f'https://drive.google.com/uc?id={FILE_ID}'

def download_data_if_needed():
    if not os.path.exists(DATA_FILE):
        import gdown
        st.warning("Mendownload dataset dari Google Drive. Tunggu beberapa detik...")
        gdown.download(URL, DATA_FILE, quiet=False)

download_data_if_needed()

# --- Load Data ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset `{DATA_FILE}` tidak ditemukan.")
        return pd.DataFrame()
    df = pd.read_excel(DATA_FILE, sheet_name='Working File', engine='openpyxl')
    # Lanjutkan proses preprocessing seperti sebelumnya...
    # [....]
    return df

df = load_data()
if df.empty:
    st.stop()

# Lanjutkan kode aslimu seperti biasa...


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
    # Membagi Engagement Rate dengan 1000, membatasi antara 1% dan 100%, lalu mengonversi ke persen
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

    if main_reco.empty and platform_input and platform_input.lower() != 'all':
        warning_text += "Data terlalu sempit dengan filter platform.\n"
        group_cols = ['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']
        main_reco = (
            filtered_sent.groupby(group_cols)
            .agg({'Engagement Rate': 'mean'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
        main_reco = apply_engagement_rate_formatting(main_reco)  # Format Engagement Rate

    if main_reco.empty and sentiment:
        warning_text += "Data terlalu sempit dengan filter sentiment.\n"
        filtered_no_sent = df[
            (df['Post Type'] == post_type) &
            (df['Audience Gender'] == audience_gender) &
            (df['Age Group'] == age_group)
        ]
        if platform_input and platform_input.lower() != 'all':
            filtered_no_sent_platform = filtered_no_sent[filtered_no_sent['Platform'] == platform_input.title()]
            group_cols = ['Time Periods', 'Post Day Name', 'Post Hour']
            main_reco = (
                filtered_no_sent_platform.groupby(group_cols)
                .agg({'Engagement Rate': 'mean'})
                .sort_values('Engagement Rate', ascending=False)
                .reset_index()
            )
            main_reco = apply_engagement_rate_formatting(main_reco)  # Format Engagement Rate
        else:
            group_cols = ['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']
            main_reco = (
                filtered_no_sent.groupby(group_cols)
                .agg({'Engagement Rate': 'mean'})
                .sort_values('Engagement Rate', ascending=False)
                .reset_index()
            )
            main_reco = apply_engagement_rate_formatting(main_reco)  # Format Engagement Rate

    if main_reco.empty:
        warning_text += "Data sangat sempit, memberikan rekomendasi umum untuk post type saja.\n"
        main_reco = (
            df[df['Post Type'] == post_type]
            .groupby(['Platform', 'Time Periods', 'Post Day Name', 'Post Hour'])
            .agg({'Engagement Rate': 'mean'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
        main_reco = apply_engagement_rate_formatting(main_reco)  # Format Engagement Rate

    return main_reco.head(5), warning_text

def strategy_recommendation(post_type, audience_gender, age_group):
    filtered = df[
        (df['Post Type'] == post_type) &
        (df['Audience Gender'] == audience_gender) &
        (df['Age Group'] == age_group)
    ]
    strategy = (
        filtered.groupby('Sentiment')
        .agg({'Engagement Rate': 'mean'})
        .sort_values('Engagement Rate', ascending=False)
        .reset_index()
    )
    strategy = apply_engagement_rate_formatting(strategy)  # Format Engagement Rate

    if strategy.empty:
        strategy = (
            df[df['Post Type'] == post_type]
            .groupby('Sentiment')
            .agg({'Engagement Rate': 'mean'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
        strategy = apply_engagement_rate_formatting(strategy)  # Format Engagement Rate

    return strategy

def alternative_platform_suggestion(post_type, audience_gender, age_group, platform_input):
    filtered = df[
        (df['Post Type'] == post_type) &
        (df['Audience Gender'] == audience_gender) &
        (df['Age Group'] == age_group)
    ]
    if platform_input and platform_input.lower() != 'all':
        alt_platform_stats = (
            filtered.groupby('Platform')
            .agg({'Engagement Rate': 'mean'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
        alt_platform_stats = alt_platform_stats[alt_platform_stats['Platform'] != platform_input.title()]
    else:
        alt_platform_stats = (
            filtered.groupby('Platform')
            .agg({'Engagement Rate': 'mean'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
    alt_platform_stats = apply_engagement_rate_formatting(alt_platform_stats)  # Format Engagement Rate
    return alt_platform_stats.head(3)

# ============== ML: Prediksi Engagement Rate ==================
@st.cache_data
def build_engagement_model():
    features = df[['Likes', 'Comments', 'Shares', 'Impressions', 'Reach']]
    target = df['Engagement Rate']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, rmse, mae, r2

model, rmse, mae, r2 = build_engagement_model()

# ===============================
# === ANTARMUKA STREAMLIT APP ===
# ===============================


st.markdown(
    """
    <div style='text-align: center;'>
        <span style="font-size:3em;">📊</span><br>
        <span style="font-size:1.8em; font-weight: bold;">Social Media Caption & Posting Analytics</span><br>
        <span style="font-size:1.2em; color:gray;">Boost Your Engagement with Smart Caption Analysis and Optimal Posting Times</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

with st.form(key='input_form'):
    caption_input = st.text_area("Enter Your Caption:")
    post_type_input = st.selectbox("Select Post Type", sorted(df['Post Type'].unique()))
    audience_gender_input = st.selectbox("Select Audience Gender", sorted(df['Audience Gender'].unique()))
    age_group_input = st.selectbox("Select Age Group", sorted(df['Age Group'].unique()))
    platforms = list(sorted(df['Platform'].unique())) + ['All']
    platform_input = st.selectbox("Select Platform", platforms)
    submit_button = st.form_submit_button(label='Check Recommendations')

if submit_button:
    # 1. Sentiment
    sentiment_result = analyze_sentiment(caption_input)
    st.success(f"Predicted Sentiment for Your Caption: **{sentiment_result}**")

    # 2. Recommendation Pipeline
    reco_pipeline, warning_pipeline = hybrid_recommendation_pipeline_super_adaptive(
        post_type_input,
        audience_gender_input,
        age_group_input,
        sentiment_result,
        platform_input
    )

    # Main output (first row from the pipeline)
    if not reco_pipeline.empty:
        best_reco = reco_pipeline.iloc[0]
        if 'Platform' in reco_pipeline.columns:
            reco_text = f"Post at **{int(best_reco['Post Hour']):02d}:00 WIB** on **{best_reco['Post Day Name']}** using platform **{best_reco['Platform']}** for maximum engagement."
        else:
            reco_text = f"Post at **{int(best_reco['Post Hour']):02d}:00 WIB** on **{best_reco['Post Day Name']}** for maximum engagement."
        st.markdown(f"### ⏰ Posting Time Recommendation\n{reco_text}")
        with st.expander("View Top 5 Recommendations"):
            st.dataframe(reco_pipeline.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.error("No relevant recommendations based on your input.")

    # Warning pipeline
    if warning_pipeline:
        st.warning(warning_pipeline)

    # 3. Caption Strategy
    strategy_reco = strategy_recommendation(post_type_input, audience_gender_input, age_group_input)
    if not strategy_reco.empty:
        best_strategy = strategy_reco.iloc[0]
        strategy_text = f"Use **{post_type_input.lower()}** content with **{best_strategy['Sentiment'].lower()}** sentiment for **{age_group_input.lower()}**."
        st.markdown(f"### 🎯 Content Strategy\n{strategy_text}")
        with st.expander("View Caption Strategy Details"):
            st.dataframe(strategy_reco.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.error("No relevant caption strategies found.")

    # 4. Alternative Platforms
    alt_platform_reco = alternative_platform_suggestion(
        post_type_input,
        audience_gender_input,
        age_group_input,
        platform_input
    )
    if not alt_platform_reco.empty:
        if platform_input.lower() != 'all':
            alt_platform_text = f"Alternative platform you might consider: **{alt_platform_reco.iloc[0]['Platform']}**."
        else:
            alt_platform_text = "Alternative platforms you might consider: " + ", ".join(alt_platform_reco['Platform'].tolist())
        st.markdown(f"### 🔄 Alternative Platform Suggestions\n{alt_platform_text}")
        with st.expander("View Top 3 Platforms"):
            st.dataframe(alt_platform_reco.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.error("No alternative platforms recommended.")

    # 5. ML Model Prediction Scores (info, not a new prediction for the user)
    st.markdown("---")
    st.markdown(f"**Engagement Rate Model - RMSE:** {rmse:.3f} | **MAE:** {mae:.3f} | **R2:** {r2:.4f}")

st.caption("© 2024 Social Media Analytics | Powered by Streamlit")

