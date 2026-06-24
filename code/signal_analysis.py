import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


CLEAN_DATA_FOLDER = "clean_data"
OUTPUT_FOLDER = "plots/analysis"
TARGET_SR = 16000
FRAME_LENGTH = 512
HOP_LENGTH = 256


WORDS = [
    ("arabi", "arabi", "pharyngeal /ayn/"),
    ("hadeka", "hadeka", "pharyngeal /haa/"),
    ("qalam", "qalam", "uvular /qaf/"),
    ("daw'", "daw'", "emphatic /dad/"),
    ("ghurfa", "ghurfa", "uvular /ghayn/"),
    ("khalid", "khalid", "uvular /khaa/"),
    ("sadeeq", "sadeeq", "emphatic /sad/"),
    ("tareeq", "tareeq", "emphatic /taa/"),
    ("zarf", "zarf", "emphatic /zaa/"),
    ("thalatha", "thalatha", "regular /thaa/"),
]

SPEAKERS = range(1, 6)


def load_audio(word_prefix, speaker):
    filename = f"{word_prefix}_{speaker}.wav"
    path = os.path.join(CLEAN_DATA_FOLDER, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y, sr


def plot_waveform(y, sr, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4c6ef5")
    ax.set_title(f"Waveform - {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_spectrogram(y, sr, title, save_path):
    spectrum = librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    db_spectrum = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)

    fig, ax = plt.subplots(figsize=(9, 3))
    img = librosa.display.specshow(
        db_spectrum,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Spectrogram - {title}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_short_time_energy(y, sr, title, save_path):
    frames = librosa.util.frame(
        y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    energy = np.sum(frames**2, axis=0)
    times = librosa.frames_to_time(
        np.arange(len(energy)), sr=sr, hop_length=HOP_LENGTH
    )

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, energy, color="#f03e3e", linewidth=1.2)
    ax.fill_between(times, energy, alpha=0.2, color="#f03e3e")
    ax.set_title(f"Short-Time Energy - {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_zcr(y, sr, title, save_path):
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, zcr, color="#2f9e44", linewidth=1.2)
    ax.fill_between(times, zcr, alpha=0.2, color="#2f9e44")
    ax.set_title(f"Zero-Crossing Rate - {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ZCR")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_pitch(y, sr, title, save_path):
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, f0, color="#e67700", linewidth=1.5)
    ax.set_title(f"Pitch Contour - {title}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def process_recording(word_prefix, label, category, speaker):
    print(f"Processing: {word_prefix}_{speaker}.wav")
    y, sr = load_audio(word_prefix, speaker)

    title = f"{label} speaker {speaker} - {category}"
    base = os.path.join(OUTPUT_FOLDER, f"{word_prefix}_{speaker}")
    output_paths = [
        f"{base}_waveform.png",
        f"{base}_spectrogram.png",
        f"{base}_energy.png",
        f"{base}_zcr.png",
        f"{base}_pitch.png",
    ]

    if all(os.path.exists(path) for path in output_paths):
        print("Already complete, skipping.")
        return

    plot_waveform(y, sr, title, output_paths[0])
    plot_spectrogram(y, sr, title, output_paths[1])
    plot_short_time_energy(y, sr, title, output_paths[2])
    plot_zcr(y, sr, title, output_paths[3])
    plot_pitch(y, sr, title, output_paths[4])


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    total = 0
    skipped = 0
    for word_prefix, label, category in WORDS:
        for speaker in SPEAKERS:
            try:
                process_recording(word_prefix, label, category, speaker)
                total += 1
            except FileNotFoundError as exc:
                skipped += 1
                print(f"SKIPPED - {exc}")

    print("\nDone.")
    print(f"Processed recordings: {total}")
    print(f"Skipped recordings: {skipped}")
    print(f"Saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
