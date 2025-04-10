import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa

# Load trained model
model = tf.keras.models.load_model('drone_autoencoder.keras')

# Parameters
SR = 16000
DURATION = 2  # seconds
N_MFCC = 20
THRESHOLD = 0.05  # adjust if needed

print("Listening for drone sounds\nPress Ctrl+C to stop.")

def get_mfcc_from_audio(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC).T
    if np.max(np.abs(mfcc)) > 0:
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc

try:
    while True:
        audio = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Extract MFCC
        mfcc = get_mfcc_from_audio(audio)

        # Predict reconstruction error
        recon = model.predict(mfcc, verbose=0)
        errors = np.mean((mfcc - recon)**2, axis=1)
        mean_error = np.mean(errors)

        print(f" {mean_error:.5f} →", end=' ')
        if mean_error < THRESHOLD:
            print("NOT DRONE")
        else:
            print("X drone X")

except KeyboardInterrupt:
    print("\n Stopped listening.")
