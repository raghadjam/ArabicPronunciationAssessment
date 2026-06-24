import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

df = pd.read_csv("../results/pronunciation_scores.csv")
os.makedirs("../plots", exist_ok=True)


# ── Final score for all test recordings ──
plt.figure(figsize=(16, 6))
colors = ["red" if "wrong" in str(f) else "#1f77b4" for f in df["test_file"]]
plt.bar(df["test_file"], df["Final_score"], color=colors)
plt.xticks(rotation=90, fontsize=7)
plt.ylabel("Final Pronunciation Score")
plt.title("Pronunciation Scores for All Test Recordings\n(red = intentionally wrong)")
plt.ylim(0, 100)
correct_patch = mpatches.Patch(color="#1f77b4", label="Correct pronunciation")
wrong_patch   = mpatches.Patch(color="red",     label="Wrong pronunciation")
plt.legend(handles=[correct_patch, wrong_patch])
plt.tight_layout()
plt.savefig("../plots/final_scores.png", dpi=150)
plt.close()
print("Saved: final_scores.png")


# ── Average score per word ───
avg_word = df.groupby("word")["Final_score"].mean().sort_values()

plt.figure(figsize=(12, 5))
plt.bar(avg_word.index, avg_word.values, color="#2ca02c")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Pronunciation Score")
plt.title("Average Pronunciation Score per Word")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("../plots/average_score_per_word.png", dpi=150)
plt.close()
print("Saved: average_score_per_word.png")


# ── Feedback distribution ──
feedback_counts = df["Feedback"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(feedback_counts.index, feedback_counts.values, color="#ff7f0e")
plt.ylabel("Number of Recordings")
plt.title("Feedback Distribution")
plt.tight_layout()
plt.savefig("../plots/feedback_distribution.png", dpi=150)
plt.close()
print("Saved: feedback_distribution.png")


# ── Helper: detect all words that have at least one "_wrong" recording ──
# A wrong recording has speaker == "wrong" (set by extract_all_features.py
# when the filename is  <word>_wrong.wav).
words_with_wrong = df[df["speaker"].astype(str) == "wrong"]["word"].unique()

score_cols   = ["MFCC_score", "Pitch_score", "Duration_score",
                "Formant_score", "ZCR_score", "Spectral_score"]
score_labels = ["MFCC", "Pitch", "Duration", "Formant\n(F2 weighted)", "ZCR", "Spectral"]

print()
print("============================================================")
print("CORRECT vs WRONG PRONUNCIATION ANALYSIS")
print("============================================================")

for word in words_with_wrong:
    word_rows = df[df["word"] == word].copy()

    # Exclude speaker 1 from the "correct" bars to avoid using the reference
    # as its own test (same as original logic for zarf)
    word_rows_display = word_rows[
        (word_rows["speaker"].astype(str) != "1") |
        (word_rows["speaker"].astype(str) == "wrong")
    ]

    if word_rows_display.empty:
        continue

    bar_colors = ["red" if "wrong" in str(f) else "#1f77b4"
                  for f in word_rows_display["test_file"]]

    # ── correct speakers vs wrong ──
    plt.figure(figsize=(9, 5))
    plt.bar(word_rows_display["test_file"], word_rows_display["Final_score"],
            color=bar_colors)
    plt.ylabel("Final Pronunciation Score")
    plt.title(f"Word '{word}': Correct Speakers vs Intentionally Wrong Pronunciation")
    plt.ylim(0, 100)
    plt.xticks(rotation=30)
    correct_patch = mpatches.Patch(color="#1f77b4", label="Correct pronunciation")
    wrong_patch   = mpatches.Patch(color="red",     label="Wrong pronunciation")
    plt.legend(handles=[correct_patch, wrong_patch])
    plt.tight_layout()
    out_path = f"../plots/{word}_speakers_vs_wrong.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {word}_speakers_vs_wrong.png")

    # ── component score breakdown ──
    x     = np.arange(len(score_cols))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(word_rows_display.iterrows()):
        color = "red" if "wrong" in str(row["test_file"]) else None
        label = row["test_file"]
        ax.bar(x + i * width, row[score_cols].values, width,
               label=label, color=color)

    ax.set_xticks(x + width * (len(word_rows_display) - 1) / 2)
    ax.set_xticklabels(score_labels)
    ax.set_ylabel("Score (0–100)")
    ax.set_title(f"Component Score Breakdown: '{word}' Recordings\n"
                 f"Note: low Formant score for wrong pronunciation reveals acoustic mismatch")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = f"../plots/{word}_component_scores.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {word}_component_scores.png")

    # ── Console report for this word ──
    print()
    print(f"Word: {word}")
    print("-" * 60)
    display_cols = ["test_file", "MFCC_score", "Pitch_score", "Duration_score",
                    "Formant_score", "ZCR_score", "Spectral_score",
                    "Final_score", "Feedback"]
    print(word_rows_display[display_cols].to_string(index=False))
    print("-" * 60)

    correct_rows = word_rows_display[
        word_rows_display["speaker"].astype(str) != "wrong"
    ]
    wrong_rows = word_rows_display[
        word_rows_display["speaker"].astype(str) == "wrong"
    ]

    if not correct_rows.empty and not wrong_rows.empty:
        correct_avg = correct_rows["Final_score"].mean()
        wrong_avg   = wrong_rows["Final_score"].mean()
        diff        = correct_avg - wrong_avg

        print(f"Average score of correct recordings : {correct_avg:.2f}")
        print(f"Average score of wrong pronunciation: {wrong_avg:.2f}")
        print(f"Difference                          : {diff:+.2f}")
        print()

        if wrong_avg < correct_avg:
            print(f"✔ Observation: The wrong pronunciation of '{word}' received a LOWER score.")
            print("  The formant weighting (F2 = 60%) successfully penalised")
            print("  the acoustic difference in the wrong recording.")
        else:
            print(f"✗ Observation: Scores are still close for '{word}'.")
            print("  Consider verifying F2 values in formants.csv via Praat.")

    print("=" * 60)

if len(words_with_wrong) == 0:
    print("No '_wrong' recordings found. Add a file named <word>_wrong.wav")
    print("to the recordings folder and re-run extract_all_features.py first.")

print()
print("Done. All plots saved in the plots/ folder.")