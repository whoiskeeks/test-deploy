import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load('best_gbr_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
features_used = joblib.load('features_used.pkl')



st.title(" Prediksi Engagement Rate")

# Input dari user
platform = st.selectbox('Platform', ['Instagram', 'Twitter', 'Facebook'])
day = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
sentiment_label = st.selectbox('Sentiment Label', ['Positive', 'Negative', 'Neutral'])
emotion_type = st.selectbox('Emotion Type', ['Confused', 'Angry', 'Sad', 'Happy', 'Excited'])
brand_name = st.selectbox('Brand Name', ['Google', 'Microsoft', 'Nike', 'Pepsi', 'Toyota', 'Coca-Cola',
       'Amazon', 'Samsung', 'Adidas', 'Apple'])
campaign_phase = st.selectbox(' Campaign phase', ['Launch', 'Post-Launch', 'Pre-Launch'])
product_name = st.selectbox(" Product Name", ['Chromebook', 'Surface Laptop', 'Epic React', 'Diet Pepsi',
       'Corolla', 'React', 'Pepsi Wild Cherry', 'Coke Zero',
       'Pixel Watch', 'Halo Band', 'Pixel Buds', 'Crystal Pepsi',
       'Galaxy S25', 'Surface Duo', 'Xbox Series X', 'Coca-Cola Vanilla',
       'Pepsi Max', 'Highlander', 'Predator', 'RAV4',
       'Xbox Elite Controller', 'Coca-Cola Cherry', 'MacBook Pro', 'iMac',
       'Ring Camera', 'Nest Hub', 'Sienna', 'Dri-FIT', 'FlyKnit',
       'Galaxy Z Fold', 'Camry', 'Yeezy', 'Pepsi Lime', 'Echo Dot',
       'Ultraboost', 'Fire Tablet', 'Samba', 'Apple Watch', 'Fanta',
       'Galaxy Buds', 'NMD', 'Prius', 'Pixel Tablet', 'iPad Air',
       'Nest Thermostat', 'AirPods Pro', 'Air Jordan', 'Tundra',
       'Stan Smith', 'Zoom Pegasus', 'Tacoma', 'Air Max', 'Sprite',
       'Mac Mini', 'Neo QLED TV', 'Air Force 1', 'Diet Coke', 'Pixel 8',
       'Galaxy Watch', 'Surface Go', 'Kindle', 'Gazelle', 'Surface Pro',
       'Eero WiFi', 'Vision Pro', 'Galaxy Tab', 'Superstar',
       'Pepsi Zero Sugar', 'Fire TV', 'iPhone 15'])
campaign_name = st.selectbox('Campaign Name', ['BlackFriday', 'PowerRelease', 'LaunchWave', 'LocalTouchpoints',
       'CyberMonday', 'GlobalCampaign', 'CustomerFirst',
       'SpringBlast2025', 'HolidaySpecial', 'ValentinesDeals',
       'InnovationX', 'WinterWonders', 'NewYearNewYou', 'SummerDreams',
       'SustainableFuture', 'EarthDay', 'BackToSchool',
       'DigitalTransformation', 'ReferralBonus', 'SummerSale',
       'FallCollection', 'LoyaltyRewards', 'NextGeneration'])

sentiment_score = st.number_input('Sentiment Score (From -1 to 1)', -1.0, 1.0, value=0.0, step=0.0001, format="%.4f")
toxicity_score = st.number_input('Toxicity Score (From 0 to 1)', 0.0, 1.0, value=0.0, step=0.0001, format="%.4f")
likes_count = st.number_input('Likes Count (From 0 to 10000)', 0, 10000)
shares_count = st.number_input('Shares Count (From 0 to 10000)', 0, 10000)
comments_count = st.number_input('Comments Count (From 0 to 10000', 0, 10000)
impressions = st.number_input('Impressions (From 0 to 10000)', 0, 10000)
user_past_sentiment_avg = st.number_input('User Past Sentiment Average (From -1 to 1)', -1.0, 1.0, value=0.0, step=0.0001, format="%.4f")
user_engagement_growth = st.number_input('User Engagement Growth (From -1 to 1)', -1.0, 1.0, value=0.0, step=0.0001, format="%.4f")
buzz_change_rate = st.number_input('Buzz Change Rate (From -100 to 100)', -100.0, 100.0, value=0.0, step=0.1, format="%.1f")

if st.button('Predict'):
    # Buat DataFrame
    data_prediction  = pd.DataFrame({
        'platform': [platform],
        'day_of_week': [day],
        'sentiment_score':[sentiment_score],
        'toxicity_score': [toxicity_score],
        'shares_count': [shares_count],
        'comments_count': [comments_count],
        'likes_count': [likes_count],
        'impressions': [impressions],
        'user_past_sentiment_avg': [user_past_sentiment_avg],
        'user_engagement_growth': [user_engagement_growth],
        'buzz_change_rate': [buzz_change_rate],
        'brand_name': [brand_name],
        'campaign_phase': [campaign_phase],
        'emotion_type': [emotion_type],
        'sentiment_label': [sentiment_label],
        'product_name': [product_name],
        'campaign_name': [campaign_name]
    })
    
    data_prediction_2 = data_prediction

    # Fitur yang digunakan
    numerical_cols = [
     'likes_count', 'comments_count', 'shares_count', 'impressions',
                      'user_past_sentiment_avg', 'user_engagement_growth',
                      'buzz_change_rate', 'toxicity_score'
    ]

    categorical_cols = ['platform', 'day_of_week', 'sentiment_label', 'emotion_type',
                        'brand_name', 'campaign_phase', 'product_name', 'campaign_name']
    
    # Encode kategori
    for col in categorical_cols:
        if col in data_prediction.columns:
            le = label_encoders.get(col)
            if le:
                # Cek apakah semua value sudah dikenal
                known_classes = set(le.classes_)
                data_prediction[col] = data_prediction[col].apply(lambda x: x if x in known_classes else le.classes_[0])
                data_prediction[col] = le.transform(data_prediction[col])
                
    #Transform pakai scaler
    data_scaled = scaler.transform(data_prediction[numerical_cols])


    # Ganti kolom numerik dengan hasil scaling
    data_scaled_df = pd.DataFrame(data_scaled, columns=[col+"_norm" for col in numerical_cols])
    data_prediction = data_prediction.drop(columns=numerical_cols)
    data_prediction = pd.concat([data_prediction, data_scaled_df], axis=1)

    #Tambahkan kolom yang belum ada
    for col in features_used:
        if col not in data_prediction.columns:
            data_prediction[col] = 0
    
    #Urutkan kolomnya
    data_prediction = data_prediction[features_used]
    
    # Prediksi
    prediction = model.predict(data_prediction)[0]
    st.success(prediction)
