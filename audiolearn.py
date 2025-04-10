import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

AUDIO_DIR = 'AudioDataSet'
SR = 16000
N_MFCC = 20
SILENCE_THRESHOLD = 5.0  # Adjust as needed

def extract_mfcc(file_path, duration=2):
    y, sr = librosa.load(file_path, sr=SR, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T

    # Filter out silent frames
    energy = np.linalg.norm(mfcc, axis=1)
    mfcc = mfcc[energy > SILENCE_THRESHOLD]

    return mfcc

# Collect features from all drone audio
features = []
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith('.wav'):
        path = os.path.join(AUDIO_DIR, filename)
        mfcc = extract_mfcc(path)
        if len(mfcc) > 0:
            features.extend(mfcc)

X = np.array(features)
print("Shape after silence filtering:", X.shape)

# Normalize
X = (X - np.mean(X)) / np.std(X)

# Define autoencoder model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(N_MFCC,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(N_MFCC)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train
history = model.fit(X, X, epochs=50, batch_size=64, validation_split=0.1)

# Save model in Keras format (recommended)
model.save('drone_autoencoder.keras')

# Save loss plot instead of showing it (fixes PyCharm issue)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.title("Autoencoder Loss")
plt.savefig("autoencoder_loss.png")
print("📉 Loss plot saved as autoencoder_loss.png")
