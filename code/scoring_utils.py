import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

MFCC_COLS = [f"mfcc_{i}" for i in range(1, 13)]
DELTA_COLS = [f"mfcc_d_{i}" for i in range(1, 13)]
DELTA2_COLS = [f"mfcc_d2_{i}" for i in range(1, 13)]
ALL_MFCC_COLS = MFCC_COLS + DELTA_COLS + DELTA2_COLS


def similarity_score(v1, v2):
    if pd.isna(v1) or pd.isna(v2):
        return 0.0
    diff = abs(float(v1) - float(v2))
    max_value = max(abs(float(v1)), abs(float(v2)), 1e-6)
    return max(0.0, 100.0 * (1.0 - diff / max_value))


def mfcc_score(mfcc1, mfcc2):
    mfcc1 = np.asarray(mfcc1, dtype=float)
    mfcc2 = np.asarray(mfcc2, dtype=float)
    if np.allclose(mfcc1, mfcc2):
        return 100.0
    sim = 1.0 - cosine(mfcc1, mfcc2)
    if np.isnan(sim):
        return 0.0
    return max(0.0, min(100.0, sim * 100.0))


def formant_score(ref, test):
    if any(pd.isna(ref.get(f)) or pd.isna(test.get(f)) for f in ["F1", "F2", "F3"]):
        return 0.0
    f1 = similarity_score(ref["F1"], test["F1"])
    f2 = similarity_score(ref["F2"], test["F2"])
    f3 = similarity_score(ref["F3"], test["F3"])
    return 0.20 * f1 + 0.60 * f2 + 0.20 * f3


def final_score(parts, include_zcr_spectral=False):
    if include_zcr_spectral:
        return (
            0.30 * parts["MFCC"] +
            0.10 * parts["Pitch"] +
            0.05 * parts["Duration"] +
            0.35 * parts["Formants"] +
            0.10 * parts.get("ZCR", 0.0) +
            0.10 * parts.get("Spectral", 0.0)
        )
    return (
        0.40 * parts["MFCC"] +
        0.20 * parts["Pitch"] +
        0.15 * parts["Duration"] +
        0.25 * parts["Formants"]
    )


def compare_rows(ref, test, include_zcr_spectral=False):
    parts = {
        "MFCC": mfcc_score(ref[ALL_MFCC_COLS].values, test[ALL_MFCC_COLS].values),
        "Pitch": similarity_score(ref["pitch"], test["pitch"]),
        "Duration": similarity_score(ref["duration"], test["duration"]),
        "Formants": formant_score(ref, test),
    }
    if include_zcr_spectral:
        parts["ZCR"] = similarity_score(ref.get("zcr"), test.get("zcr"))
        parts["Spectral"] = (
            similarity_score(ref.get("spectral_centroid"), test.get("spectral_centroid")) +
            similarity_score(ref.get("spectral_rolloff"), test.get("spectral_rolloff"))
        ) / 2.0
    parts["Total"] = final_score(parts, include_zcr_spectral)
    return {k: round(float(v), 2) for k, v in parts.items()}


def estimate_formants_lpc(y, sr):
    """Simple LPC fallback for F1-F3 when Praat formants are not available."""
    import librosa
    if len(y) < sr * 0.05:
        return {"F1": np.nan, "F2": np.nan, "F3": np.nan}
    y = librosa.effects.preemphasis(y)
    order = 2 + int(sr / 1000)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = librosa.lpc(y, order=order)
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = sorted(angles * (sr / (2 * np.pi)))
        formants = [f for f in freqs if 250 < f < 4000][:3]
        while len(formants) < 3:
            formants.append(np.nan)
        return {"F1": formants[0], "F2": formants[1], "F3": formants[2]}
    except Exception:
        return {"F1": np.nan, "F2": np.nan, "F3": np.nan}


def extract_features_from_wav(file_path):
    import librosa
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    row = {
        "file": os.path.basename(file_path),
        "duration": librosa.get_duration(y=y, sr=sr),
        "energy": float(np.mean(librosa.feature.rms(y=y))),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
    }

    pitch_values = librosa.yin(y, fmin=75, fmax=500)
    voiced_pitch = pitch_values[pitch_values > 80]
    row["pitch"] = float(np.mean(voiced_pitch)) if len(voiced_pitch) else float(np.mean(pitch_values))

    for i in range(12):
        row[f"mfcc_{i+1}"] = float(np.mean(mfcc, axis=1)[i])
        row[f"mfcc_d_{i+1}"] = float(np.mean(mfcc_delta, axis=1)[i])
        row[f"mfcc_d2_{i+1}"] = float(np.mean(mfcc_delta2, axis=1)[i])

    row.update(estimate_formants_lpc(y, sr))
    return pd.Series(row)


def load_feature_table(results_dir="../results"):
    features = pd.read_csv(os.path.join(results_dir, "all_features.csv"))
    formant_path = os.path.join(results_dir, "formants.csv")
    if os.path.exists(formant_path):
        formants = pd.read_csv(formant_path)
        features = features.merge(formants, on="file", how="left", suffixes=("", "_praat"))
    return features
