import streamlit as st
import joblib
import json
import os
import pandas as pd

# ===== Bagian 1: Load model & encoding map (path relatif ke lokasi file ini) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'stunting_nb_model.pkl')
ENCODING_PATH = os.path.join(BASE_DIR, '..', 'model', 'encoding_map.json')

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(ENCODING_PATH, 'r') as f:
        encoding_map = json.load(f)
    return model, encoding_map

model, encoding_map = load_artifacts()

# ===== Bagian 2: Replikasi logika age-band, HARUS identik dengan train_model.py =====
def age_band(a):
    if a <= 11: return '0-11'
    elif a <= 23: return '12-23'
    elif a <= 35: return '24-35'
    elif a <= 47: return '36-47'
    else: return '48-59'

def compute_height_std(age_month, height_cm, encoding_map):
    band = age_band(age_month)
    band_stats = encoding_map['age_bands'].get(band)
    if band_stats is None:
        return None
    mean, std = band_stats['mean'], band_stats['std']
    return (height_cm - mean) / std

# ===== Bagian 3: UI form =====
st.title('Deteksi Dini Risiko Stunting Balita')
st.write('Masukkan data antropometri balita untuk memperkirakan risiko stunting.')

with st.form('input_form'):
    gender_label = st.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
    age_month = st.number_input('Usia (bulan)', min_value=0, max_value=59, value=24, step=1)
    height_cm = st.number_input('Tinggi Badan (cm)', min_value=30.0, max_value=130.0, value=85.0, step=0.1)
    submitted = st.form_submit_button('Prediksi')

# ===== Bagian 4: Prediksi =====
if submitted:
    gender_map = encoding_map['gender']
    gender_code = gender_map['F'] if gender_label == 'Perempuan' else gender_map['M']

    height_std = compute_height_std(age_month, height_cm, encoding_map)

    if height_std is None:
        st.error('Usia di luar rentang yang didukung model (0–59 bulan).')
    else:
        X_input = pd.DataFrame({
            'Gender_Code': [gender_code],
            'Height_std': [height_std]
        })[encoding_map['features_order']]

        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        label_map = encoding_map['label']
        hasil = label_map[str(pred)] if str(pred) in label_map else label_map[pred]

        st.subheader(f'Hasil: {hasil}')
        st.write(f"Probabilitas Not Stunted: {proba[0]*100:.1f}%")
        st.write(f"Probabilitas Stunted: {proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

        st.caption('Catatan: ini alat skrining awal, bukan diagnosis medis. Konsultasikan ke tenaga kesehatan untuk kepastian.')