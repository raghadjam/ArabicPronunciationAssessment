import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

# ── Load features ──
df       = pd.read_csv("../results/all_features.csv")
formants = pd.read_csv("../results/formants.csv")

df = df.merge(formants, on="file", how="left")

# Column groups
mfcc_cols   = [f"mfcc_{i}"    for i in range(1, 13)]
delta_cols  = [f"mfcc_d_{i}"  for i in range(1, 13)]
delta2_cols = [f"mfcc_d2_{i}" for i in range(1, 13)]
all_mfcc_cols = mfcc_cols + delta_cols + delta2_cols


# ── Helper functions ──

def similarity_score(v1, v2):
    """Generic 0-100 similarity for scalar features."""
    diff      = abs(v1 - v2)
    max_value = max(abs(v1), abs(v2), 1e-6)
    return max(0.0, 100.0 * (1.0 - diff / max_value))


def mfcc_score(mfcc1, mfcc2):
    """Cosine similarity over concatenated MFCC + delta + delta-delta vectors."""
    sim = 1.0 - cosine(mfcc1, mfcc2)
    return max(0.0, sim * 100.0)


def formant_score(ref, test):
    """
    Compare F1, F2, F3.
    F2 is weighted 60 % because it is the primary acoustic cue that separates
    Arabic emphatic consonants (ظ ض ص ط) from their plain counterparts (ز س ت د).
    Emphatics cause a significant lowering of F2 due to pharyngealization.
    """
    if pd.isna(ref.get("F1")) or pd.isna(test.get("F1")):
        return 0.0

    f1 = similarity_score(ref["F1"], test["F1"])
    f2 = similarity_score(ref["F2"], test["F2"])   # ← most important
    f3 = similarity_score(ref["F3"], test["F3"])

    return 0.20 * f1 + 0.60 * f2 + 0.20 * f3


def feedback(score):
    if score >= 85:
        return "Excellent pronunciation"
    elif score >= 70:
        return "Good pronunciation"
    elif score >= 50:
        return "Needs improvement"
    else:
        return "Poor pronunciation"




results = []

for word in df["word"].unique():
    word_df = df[df["word"] == word]

    # Correct speakers = everyone whose speaker tag is NOT "wrong"
    ref_rows = word_df[word_df["speaker"].astype(str) != "wrong"]

    if ref_rows.empty:
        continue

    # Average reference vector
    ref_avg = ref_rows[
        all_mfcc_cols +
        ["pitch", "duration", "zcr", "spectral_centroid", "spectral_rolloff",
         "F1", "F2", "F3"]
    ].mean()

    # Score every non-reference-1 file against the average reference
    for _, test in word_df.iterrows():
        

        # ── Individual component scores ──
        mfcc_s     = mfcc_score(
            ref_avg[all_mfcc_cols].values,
            test[all_mfcc_cols].values
        )

        pitch_s    = similarity_score(ref_avg["pitch"],    test["pitch"])
        duration_s = similarity_score(ref_avg["duration"], test["duration"])
        zcr_s      = similarity_score(ref_avg["zcr"],      test["zcr"])

        spectral_s = (
            similarity_score(ref_avg["spectral_centroid"], test["spectral_centroid"]) +
            similarity_score(ref_avg["spectral_rolloff"],  test["spectral_rolloff"])
        ) / 2.0

        formant_s  = formant_score(ref_avg, test)

        
        final_score = (
            0.30 * mfcc_s     +
            0.10 * pitch_s    +
            0.05 * duration_s +
            0.35 * formant_s  +
            0.10 * zcr_s      +
            0.10 * spectral_s
        )

        results.append({
            "word":              word,
            "test_file":         test["file"],
            "speaker":           test["speaker"],
            "MFCC_score":        round(mfcc_s,     2),
            "Pitch_score":       round(pitch_s,    2),
            "Duration_score":    round(duration_s, 2),
            "Formant_score":     round(formant_s,  2),
            "ZCR_score":         round(zcr_s,      2),
            "Spectral_score":    round(spectral_s, 2),
            "Final_score":       round(final_score, 2),
            "Feedback":          feedback(final_score),
        })

out = pd.DataFrame(results)
out.to_csv("../results/pronunciation_scores.csv", index=False)

print("Done. Scores saved in results/pronunciation_scores.csv")
print()
print(out[["word", "test_file", "MFCC_score", "Pitch_score", "Duration_score",
           "Formant_score", "ZCR_score", "Spectral_score",
           "Final_score", "Feedback"]].to_string(index=False))