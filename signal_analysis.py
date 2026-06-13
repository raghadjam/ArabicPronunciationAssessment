"""
Arabic Pronunciation - Signal Analysis
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# Arabic fix libraries
import arabic_reshaper
from bidi.algorithm import get_display


CLEAN_DATA_FOLDER = "clean_data"
OUTPUT_FOLDER     = "plots/analysis"
TARGET_SR         = 16000
FRAME_LENGTH      = 512
HOP_LENGTH        = 256


WORDS = [
    ("daw'",    "ضوء",  "(ض)"),
    ("arabi",   "عربي", "(ع)"),
    ("khalid",  "خالد", "(خ)"),
]

SPEAKER = 1


# ---------- Arabic fix ----------
def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))


# ---------- Audio loading ----------
def load_audio(word_prefix):
    filename = f"{word_prefix}_{SPEAKER}.wav"
    path = os.path.join(CLEAN_DATA_FOLDER, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y, sr, filename


# ---------- Plots ----------
def plot_waveform(y, sr, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4c6ef5")
    ax.set_title(f"Waveform — {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_spectrogram(y, sr, title, save_path):
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)),
        ref=np.max
    )

    fig, ax = plt.subplots(figsize=(9, 3))
    img = librosa.display.specshow(
        D,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma"
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Spectrogram — {title}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_short_time_energy(y, sr, title, save_path):
    frames = librosa.util.frame(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    energy = np.sum(frames ** 2, axis=0)
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, energy, color="#f03e3e", linewidth=1.2)
    ax.fill_between(times, energy, alpha=0.2, color="#f03e3e")

    ax.set_title(f"Short-Time Energy — {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_zcr(y, sr, title, save_path):
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, zcr, color="#2f9e44", linewidth=1.2)
    ax.fill_between(times, zcr, alpha=0.2, color="#2f9e44")

    ax.set_title(f"Zero-Crossing Rate — {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ZCR")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_pitch(y, sr, title, save_path):
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=HOP_LENGTH
    )

    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, f0, color="#e67700", linewidth=1.5)
    ax.set_title(f"Pitch Contour — {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


# ---------- Main processing ----------
def process_word(word_prefix, arabic, category):
    print(f"\nProcessing: {arabic} ({category})")

    y, sr, filename = load_audio(word_prefix)

    label = f"{fix_arabic(arabic)} — {category}"
    base = os.path.join(OUTPUT_FOLDER, word_prefix)

    plot_waveform(y, sr, label, f"{base}_1_waveform.png")
    print("Waveform")

    plot_spectrogram(y, sr, label, f"{base}_2_spectrogram.png")
    print("Spectrogram")

    plot_short_time_energy(y, sr, label, f"{base}_3_energy.png")
    print("Energy")

    plot_zcr(y, sr, label, f"{base}_4_zcr.png")
    print("ZCR")

    plot_pitch(y, sr, label, f"{base}_5_pitch.png")
    print("Pitch")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for word_prefix, arabic, category in WORDS:
        try:
            process_word(word_prefix, arabic, category)
        except FileNotFoundError as e:
            print(f"SKIPPED — {e}")

    print("\nDone.")
    print(f"Saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()