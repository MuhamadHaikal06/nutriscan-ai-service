import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. Persiapan Awal (Parameter & Path)
# -------------------------------------
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
INITIAL_EPOCHS = 60
FINE_TUNE_EPOCHS = 60
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# Menggunakan forward slash (/) untuk path agar kompatibel
train_dir = 'dataset_klasifikasi/train'
valid_dir = 'dataset_klasifikasi/valid'
test_dir = 'dataset_klasifikasi/test'

# 2. Memuat Dataset
# ------------------
print("Memuat dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print("Kelas makanan yang ditemukan:", class_names)
num_classes = len(class_names)

# 3. Membangun Arsitektur Model (TRANSFER LEARNING)
# --------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# 4. Compile Model (untuk training awal)
# ---------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Melatih Model (Tahap 1 - Melatih "Kepala" Baru)
# ---------------------------------------------------
print("\nMemulai proses training awal...")
history = model.fit(
    train_dataset,
    validation_data=valid_dataset, # Menggunakan validation set di sini
    epochs=INITIAL_EPOCHS
)

# 6. Fine-Tuning (Tahap 2)
# ------------------------
print("\nMemulai proses Fine-Tuning...")
base_model.trainable = True
fine_tune_at = 60

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Nama file untuk menyimpan model terbaik
best_model_path = 'model_terbaik.h5'

fine_tune_checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Lanjutkan training (fine-tuning)
history_fine = model.fit(
    train_dataset,
    validation_data=valid_dataset, # Menggunakan validation set juga di sini
    epochs=TOTAL_EPOCHS,
    initial_epoch=history.epoch[-1],
    callbacks=[fine_tune_checkpoint]
)
print("\nProses fine-tuning selesai.")

# 7. Evaluasi Model Final dengan Data Tes
# ----------------------------------------
print("\nMemuat model terbaik untuk evaluasi akhir...")
# Kita muat kembali model terbaik yang disimpan oleh checkpoint
best_model = models.load_model(best_model_path)

print("Mengevaluasi model dengan data test...")
loss, accuracy = best_model.evaluate(test_dataset)
print(f"Hasil Akhir Akurasi Model pada Data Tes: {accuracy*100:.2f}%")