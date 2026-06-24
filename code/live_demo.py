import os
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

from scoring_utils import extract_features_from_wav, compare_rows

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

WORDS = [
    "arabi", "qalam", "khalid", "tareeq", "sadeeq",
    "ghurfa", "hadeka", "thalatha", "zarf", "dawa"
]

print("\nARABIC PRONUNCIATION LIVE DEMO\n")

print("Available words:")
for i, word in enumerate(WORDS, start=1):
    print(f"{i}. {word}")

choice = int(input("\nChoose word number: "))
word = WORDS[choice - 1]

reference_file = os.path.join(RECORDINGS_DIR, f"{word}_1.wav")
live_file = os.path.join(RECORDINGS_DIR, "live_input.wav")

if not os.path.exists(reference_file):
    print("Reference file not found:", reference_file)
    exit()

duration = 3
sample_rate = 44100

print(f"\nSay the word now: {word}")
recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1
)
sd.wait()

sf.write(live_file, recording, sample_rate)

print("\nRecording saved as live_input.wav")
print("Comparing live pronunciation with speaker recordings...\n")

ref_features = extract_features_from_wav(reference_file)

labels = []
scores = []
colors = []

for i in range(2, 6):
    speaker_file = os.path.join(RECORDINGS_DIR, f"{word}_{i}.wav")

    if os.path.exists(speaker_file):
        speaker_features = extract_features_from_wav(speaker_file)
        result = compare_rows(ref_features, speaker_features)

        labels.append(f"{word}_{i}.wav")
        scores.append(result["Total"])
        colors.append("blue")

live_features = extract_features_from_wav(live_file)
live_result = compare_rows(ref_features, live_features)

labels.append("live_input.wav")
scores.append(live_result["Total"])
colors.append("red")

print("LIVE DEMO RESULT")
print("----------------")
print(f"Reference: {word}_1.wav")
print(f"Live score: {live_result['Total']:.2f}%")
print(f"MFCC: {live_result['MFCC']:.2f}%")
print(f"Pitch: {live_result['Pitch']:.2f}%")
print(f"Duration: {live_result['Duration']:.2f}%")
print(f"Formants: {live_result['Formants']:.2f}%")

plt.figure(figsize=(11, 6))

bars = plt.bar(labels, scores, color=colors)

plt.ylim(0, 100)
plt.ylabel("Final Pronunciation Score")
plt.title(f"Word '{word}': Speakers vs Live Demo Pronunciation")
plt.xticks(rotation=35, ha="right")

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

chart_file = os.path.join(RESULTS_DIR, f"live_demo_comparison_{word}.png")
plt.savefig(chart_file, bbox_inches="tight")
plt.show()

print("\nChart saved to:")
print(chart_file)