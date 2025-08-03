import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- BAGIAN SETUP MODEL & DATA ---

# Path ke model TERBAIK Anda
MODEL_PATH = 'model_terbaik.h5' 
print(f"[*] Memuat model dari: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[*] Model berhasil dimuat.")

# Definisikan nama kelas/makanan sesuai urutan saat training
CLASS_NAMES = ['Apple', 'Ayam Goreng', 'Bakso', 'Banana', 'Burger', 'Capcay', 'Chocolate Chip Cookie', 'Donat', 'Ikan Goreng', 'Kentang Goreng', 'Kiwi', 'Mie Goreng', 'Nasi Goreng', 'Nasi Putih', 'Nugget', 'Pempek', 'Pineapples', 'Pizza', 'Rendang Sapi', 'Sate', 'Spaghetti', 'Steak', 'Strawberry', 'Tahu Goreng', 'Telur Goreng', 'Rebus', 'Tempe Goreng', 'Terong Balado', 'Tumis Kangkung']

# Kamus kalori yang kita definisikan manual
KALORI_MAKANAN = {
    'Apple': 52,
    'Ayam Goreng': 260,
    'Bakso': 260,
    'Banana': 89,
    'Burger': 295,
    'Capcay': 80,
    'Chocolate Chip Cookie': 488,
    'Donat': 452,
    'Ikan Goreng': 200,
    'Kentang Goreng': 312,
    'Kiwi': 41,
    'Mie Goreng': 321,
    'Nasi Goreng': 165,
    'Nasi Putih': 130,
    'Nugget': 296,
    'Pempek': 215,
    'Pineapples': 50,
    'Pizza': 266,
    'Rendang Sapi': 193,
    'Sate': 275,
    'Spaghetti': 158,
    'Steak': 271,
    'Strawberry': 32,
    'Tahu Goreng': 271,
    'Telur Goreng': 196,
    'Telur Rebus': 155,
    'Tempe Goreng': 192,
    'Terong Balado': 130,
    'Tumis Kangkung': 60
}

IMG_HEIGHT = 150
IMG_WIDTH = 150

# --- AKHIR BAGIAN SETUP ---

@app.route('/estimasi', methods=['POST'])
def estimasi_kalori():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang dikirim'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if file:
        try:
            # Preprocessing gambar
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Lakukan Prediksi Nama Makanan
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)

            # Ambil kalori dari kamus, jika tidak ada, beri nilai default 0
            kalori_prediksi = KALORI_MAKANAN.get(predicted_class, 0)

            # Siapkan hasil dalam format JSON
            hasil = {
                'nama_makanan': predicted_class,
                'calories': kalori_prediksi,
                'confidence': f"{confidence:.2f}%"
            }
            return jsonify(hasil)

        except Exception as e:
            return jsonify({'error': f'Terjadi error saat memproses gambar: {e}'}), 500
    
    return jsonify({'error': 'Terjadi kesalahan tidak diketahui.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)