import os
import librosa
import numpy as np
import pandas as pd

RECORDINGS_DIR = "../recordings"
RESULTS_DIR = "../results"

os.makedirs(RESULTS_DIR, exist_ok=True)

rows = []

for filename in sorted(os.listdir(RECORDINGS_DIR)):
    if not filename.endswith(".wav"):
        continue

    file_path = os.path.join(RECORDINGS_DIR, filename)

    y, sr = librosa.load(file_path, sr=None)

    # Silence removal and normalization
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)

    # ── MFCC (12 coefficients + deltas + delta-deltas) ──
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta and delta-delta MFCCs capture dynamics (important for emphatics)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)

    # ── Duration ──
    duration = librosa.get_duration(y=y, sr=sr)

    # ── Pitch (voiced frames only — more stable) ──
    pitch_values = librosa.yin(y, fmin=75, fmax=500)
    voiced_pitch = pitch_values[pitch_values > 80]   # discard unvoiced frames
    pitch = float(np.mean(voiced_pitch)) if len(voiced_pitch) > 0 else float(np.mean(pitch_values))

    # ── Energy & ZCR ──
    energy = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # ── Spectral features (help distinguish emphatic consonants) ──
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # ── Parse word / speaker from filename ─
    base = filename.replace(".wav", "")
    parts = base.rsplit("_", 1)
    word    = parts[0] if len(parts) == 2 else base
    speaker = parts[1] if len(parts) == 2 else "unknown"

    row = {
        "file":              filename,
        "word":              word,
        "speaker":           speaker,
        "duration":          duration,
        "pitch":             pitch,
        "energy":            energy,
        "zcr":               zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff":  spectral_rolloff,
    }

    for i in range(12):
        row[f"mfcc_{i+1}"]        = mfcc_mean[i]
        row[f"mfcc_d_{i+1}"]      = mfcc_delta_mean[i]
        row[f"mfcc_d2_{i+1}"]     = mfcc_delta2_mean[i]

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "all_features.csv"), index=False)

print("Done. Features saved in results/all_features.csv")
print(df[["file", "word", "speaker", "duration", "pitch", "energy", "zcr",
          "spectral_centroid", "spectral_rolloff"]].to_string())