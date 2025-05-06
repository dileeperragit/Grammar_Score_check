 # app.py
import streamlit as st
import librosa
import numpy as np
import joblib

# Load the trained model
model = joblib.load('grammar_score_model.pkl')

# Feature extraction function
def extract_features(file_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    return mfcc_mean.reshape(1, -1)

# Streamlit UI
st.title("Grammar Scoring Engine")
st.write("Upload an audio file (wav format, 16kHz) to predict the grammar score.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    try:
        features = extract_features("temp.wav")
        prediction = model.predict(features)[0]
        st.success(f"Predicted Grammar Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")
