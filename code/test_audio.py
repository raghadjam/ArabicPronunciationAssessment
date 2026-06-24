import librosa
import numpy as np

audio_file = "../recordings/arabi_1.wav"   # ← change as needed

y, sr = librosa.load(audio_file, sr=None)
print(f"File        : {audio_file}")
print(f"Sample rate : {sr} Hz")
print(f"Raw duration: {librosa.get_duration(y=y, sr=sr):.3f} s")

# ── Preprocessing ──
y, _ = librosa.effects.trim(y, top_db=20)
y    = librosa.util.normalize(y)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Trimmed dur : {duration:.3f} s")

# ── MFCC + deltas ──
mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
mfcc_d   = librosa.feature.delta(mfcc)
mfcc_d2  = librosa.feature.delta(mfcc, order=2)
print(f"\nMFCC means       : {np.mean(mfcc,    axis=1).round(2)}")
print(f"MFCC delta means : {np.mean(mfcc_d,  axis=1).round(2)}")
print(f"MFCC delta2 means: {np.mean(mfcc_d2, axis=1).round(2)}")

# ── Pitch (voiced frames only) ──
pitch_vals  = librosa.yin(y, fmin=75, fmax=500)
voiced      = pitch_vals[pitch_vals > 80]
pitch       = float(np.mean(voiced)) if len(voiced) > 0 else float(np.mean(pitch_vals))
print(f"\nPitch (voiced)   : {pitch:.2f} Hz  ({len(voiced)} voiced frames)")

# ── Energy & ZCR ──
energy = float(np.mean(librosa.feature.rms(y=y)))
zcr    = float(np.mean(librosa.feature.zero_crossing_rate(y)))
print(f"Energy           : {energy:.6f}")
print(f"ZCR              : {zcr:.6f}")

# ── Spectral features ──
centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y,  sr=sr)))
print(f"Spectral centroid: {centroid:.2f} Hz")
print(f"Spectral rolloff : {rolloff:.2f} Hz")

print("\n[OK] All features extracted successfully.")
print("Note: Formants (F1/F2/F3) are extracted separately via Praat.")