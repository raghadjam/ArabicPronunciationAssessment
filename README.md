# Arabic Pronunciation Assessment


A Python project for assessing Arabic word pronunciation from `.wav` audio files.

The system extracts acoustic features from recordings, compares pronunciations, calculates similarity scores, and generates result plots.

## What It Does

- Extracts MFCC, pitch, duration, formant, ZCR, and spectral features.
- Compares speaker recordings against reference pronunciations.
- Gives a pronunciation score from `0` to `100`.
- Labels results as excellent, good, needs improvement, or poor.
- Generates CSV results and visual plots.
- Includes a live microphone demo.

## Project Structure

```text
code/         Python scripts
results/      CSV outputs and demo results
plots/        Generated graphs
praat plots/  Praat analysis images
paper.pdf     Project paper
```

## Requirements

Install the needed Python packages:

```powershell
pip install librosa numpy pandas scipy matplotlib soundfile sounddevice
```

## Expected Audio Files

The scripts expect a `recordings/` folder in the project root.

Example file names:

```text
arabi_1.wav
arabi_2.wav
zarf_5.wav
arabi_wrong.wav
```

Format:

```text
<word>_<speaker>.wav
```

## How to Run

Preprocess recordings:

```powershell
python code\preprocessor.py
```

Extract features:

```powershell
cd code
python extract_all_features.py
```

Score pronunciations:

```powershell
python compare_and_score.py
```

Generate plots:

```powershell
python plot_results.py
```

Run the live demo:

```powershell
python live_demo.py
```

## Main Outputs

```text
results/all_features.csv
results/formants.csv
results/pronunciation_scores.csv
results/experiment_scores.csv
plots/*.png
```
