import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (jika belum ada, penting untuk environment deployment)
# Streamlit Cloud biasanya menangani ini jika tercantum di requirements
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


# Load model dan vectorizer
try:
    with open('cyberbullying_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model atau Vectorizer tidak ditemukan. Pastikan file .pkl ada di direktori yang sama.")
    st.stop() # Menghentikan eksekusi script jika file tidak ada
except Exception as e:
    st.error(f"Error saat memuat model atau vectorizer: {e}")
    st.stop()


# Fungsi pre-processing (HARUS SAMA PERSIS dengan yang di Colab saat training)
lemmatizer = WordNetLemmatizer()
stop_words_set = set(stopwords.words('english')) # Gunakan set untuk pencarian lebih cepat

def preprocess_text_streamlit(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_set and len(word) > 2]
    return " ".join(words)

# Tampilan Streamlit
st.set_page_config(page_title="Cyberbullying Detection App", layout="wide")

st.title("🔍 Aplikasi Deteksi Cyberbullying")
st.markdown("""
Aplikasi ini menggunakan model Machine Learning untuk memprediksi jenis cyberbullying dari teks yang dimasukkan.
Masukkan sebuah kalimat atau tweet di bawah ini untuk melihat prediksinya.
""")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks di sini:", height=150, placeholder="Contoh: 'You are so stupid, I hate you!'")

# Tombol untuk prediksi
if st.button("Deteksi Cyberbullying"):
    if user_input:
        # 1. Pre-process input
        cleaned_input = preprocess_text_streamlit(user_input)
        if not cleaned_input.strip(): # Jika setelah preprocess hasilnya string kosong
            st.warning("Teks yang dimasukkan terlalu pendek atau hanya berisi stopwords/simbol setelah dibersihkan.")
        else:
            # 2. Vectorize input
            vectorized_input = vectorizer.transform([cleaned_input])
            # 3. Prediksi
            prediction = model.predict(vectorized_input)
            prediction_proba = model.predict_proba(vectorized_input) # Dapatkan probabilitas

            st.subheader("Hasil Prediksi:")
            st.success(f"Jenis Cyberbullying Terdeteksi: **{prediction[0]}**")

            st.subheader("Probabilitas Prediksi per Kelas:")
            # Membuat dataframe untuk menampilkan probabilitas dengan lebih rapi
            import pandas as pd
            proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
            st.dataframe(proba_df.T.rename(columns={0: 'Probabilitas'}).style.format("{:.2%}"))

            # Penjelasan singkat tentang jenis cyberbullying
            if prediction[0] == 'not_cyberbullying':
                st.info("Teks ini tidak terindikasi sebagai cyberbullying.")
            elif prediction[0] == 'gender':
                st.warning("Teks ini terindikasi sebagai cyberbullying berbasis **Gender**.")
            elif prediction[0] == 'religion':
                st.warning("Teks ini terindikasi sebagai cyberbullying berbasis **Agama**.")
            elif prediction[0] == 'age':
                st.warning("Teks ini terindikasi sebagai cyberbullying berbasis **Usia**.")
            elif prediction[0] == 'ethnicity':
                st.warning("Teks ini terindikasi sebagai cyberbullying berbasis **Etnisitas**.")
            elif prediction[0] == 'other_cyberbullying':
                st.warning("Teks ini terindikasi sebagai **Jenis Cyberbullying Lainnya**.")

    else:
        st.warning("Mohon masukkan teks untuk dideteksi.")

st.markdown("---")
st.markdown("Dibuat sebagai bagian dari tugas Machine Learning.")
st.markdown("Dataset: [Kaggle Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)")
