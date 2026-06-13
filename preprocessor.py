"""
Arabic Pronunciation Dataset Preprocessor

"""

import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

# ── settings ────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "recordings"
OUTPUT_FOLDER = "clean_data"  
PLOTS_FOLDER  = "plots"       

TARGET_SR    = 16000
TOP_DB       = 30
FRAME_LENGTH = 512             # ~32ms at 16kHz
HOP_LENGTH   = 256             # 50% overlap
# ────────────────────────────────────────────────────────────────────────────


def remove_silence(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return y_trimmed


def normalise(y):
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val


def apply_framing_windowing(y):
    frames = librosa.util.frame(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    frames = frames.T  # (num_frames, frame_length)
    window = np.hanning(FRAME_LENGTH)
    windowed_frames = frames * window
    return windowed_frames


def save_waveform_plot(y, sr, filename):
    plt.figure(figsize=(8, 2.5))
    librosa.display.waveshow(y, sr=sr, color="#4c6ef5")
    plt.title(filename, fontsize=11)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_FOLDER, filename.replace(".wav", ".png"))
    plt.savefig(plot_path, dpi=120)
    plt.close()


def process_all():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER,  exist_ok=True)

    wav_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No .wav files found in '{INPUT_FOLDER}'. Check the folder name.")
        return

    print(f"Found {len(wav_files)} files. Processing...\n")

    for filename in sorted(wav_files):
        input_path  = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        try:
            y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
            print(f"Processing: {filename}")

            # step 1: silence removal
            y = remove_silence(y)

            # step 2: normalisation
            y = normalise(y)

            # step 3: save clean file 
            sf.write(output_path, y, sr)

            # step 4: framing + windowing 
            windowed_frames = apply_framing_windowing(y)

            # step 5: waveform plot 
            save_waveform_plot(y, sr, filename)

            print(f"  Done — {windowed_frames.shape[0]} frames | clean file saved\n")

        except Exception as e:
            print(f"  ERROR on {filename}: {e}\n")

    print("=" * 50)


if __name__ == "__main__":
    process_all()