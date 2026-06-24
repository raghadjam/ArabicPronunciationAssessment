import os
import pandas as pd
from scoring_utils import load_feature_table, extract_features_from_wav, compare_rows

RESULTS_DIR = "../results"
OUT_FILE = os.path.join(RESULTS_DIR, "experiment_scores.csv")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_feature_table(RESULTS_DIR)

    rows = []

    # Same speaker vs same speaker
    ref = df[df["file"] == "arabi_1.wav"].iloc[0]
    test = df[df["file"] == "arabi_1.wav"].iloc[0]
    scores = compare_rows(ref, test, include_zcr_spectral=False)
    rows.append({
        "Scenario": "Same speaker vs same speaker",
        "Comparison": "arabi_1.wav vs arabi_1.wav",
        "MFCC": scores["MFCC"],
        "Pitch": scores["Pitch"],
        "Duration": scores["Duration"],
        "Formants": scores["Formants"],
        "Total": scores["Total"],
    })

    # Different speakers
    ref = df[df["file"] == "arabi_1.wav"].iloc[0]
    test = df[df["file"] == "arabi_2.wav"].iloc[0]
    scores = compare_rows(ref, test, include_zcr_spectral=False)
    rows.append({
        "Scenario": "Different speakers",
        "Comparison": "arabi_1.wav vs arabi_2.wav",
        "MFCC": scores["MFCC"],
        "Pitch": scores["Pitch"],
        "Duration": scores["Duration"],
        "Formants": scores["Formants"],
        "Total": scores["Total"],
    })

    # Correct vs incorrect pronunciation from full project output
    score_file = os.path.join(RESULTS_DIR, "pronunciation_scores.csv")
    if os.path.exists(score_file):
        score_df = pd.read_csv(score_file)
        wrong = score_df[score_df["test_file"] == "arabi_wrong.wav"]
        if not wrong.empty:
            wrong = wrong.iloc[0]
            rows.append({
                "Scenario": "Correct vs incorrect pronunciation",
                "Comparison": "arabi_reference vs arabi_wrong.wav",
                "MFCC": wrong["MFCC_score"],
                "Pitch": wrong["Pitch_score"],
                "Duration": wrong["Duration_score"],
                "Formants": wrong["Formant_score"],
                "Total": wrong["Final_score"],
            })
        else:
            raise FileNotFoundError("arabi_wrong.wav was not found in pronunciation_scores.csv. Run compare_and_score.py first.")
    else:
        raise FileNotFoundError("Missing results/pronunciation_scores.csv. Run compare_and_score.py first.")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False)

    print("\nEXPERIMENT SCORES FOR REPORT")
    print("=" * 70)
    print(out.to_string(index=False))
    print("=" * 70)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
