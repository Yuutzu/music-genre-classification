# ResNet18 with Spectrogram in Classifying Music Genres

A leakage-aware music genre classifier. Each original 30-second GTZAN song is converted into a Mel spectrogram image and classified into one of 10 genres by a pretrained ResNet18.

**Group 4**: Justine Keith Q. Polinar and Noel Evan Salamanes
BS Computer Science, College of Computing Education, University of Mindanao

---

## Overview

Many genre-classification studies cut each song into short clips to get more training samples. When clips from the same song land in both the training and test sets, the model is tested on audio it has effectively already heard, and the accuracy is inflated.

This project avoids that problem:

- Every **30-second song stays whole** and is treated as one sample.
- The data is split **at the song level**, so no part of a test song is seen during training.
- Each song becomes **one Mel spectrogram image** (224 x 224 x 3) and is classified by a **pretrained ResNet18**.

Two training setups are compared:

| Setup | Role | What learns |
|---|---|---|
| **Frozen** | Baseline | Only the new final layer (512 -> 10), 5,130 trainable parameters |
| **Fine-tuned** | Comparison | Every layer, starting from the pretrained ImageNet weights (about 11.2 million trainable parameters) |

## Key results

Evaluated on 150 unseen test songs (15 per genre).

| Metric | Frozen (baseline) | Fine-tuned |
|---|---|---|
| Test accuracy | 69.33% | 77.33% |
| Macro precision | 69.60% | 77.42% |
| Macro recall | 69.33% | 77.33% |
| Macro F1 | 67.96% | 75.83% |
| Best validation accuracy | 64.00% | 74.67% |
| Final train accuracy | 85.12% | 98.28% |
| Training time (30 epochs) | 82.0 s | 107.1 s |

- Classical and Jazz are the easiest genres to classify. Rock is the hardest.
- Fine-tuning improved 7 of 10 genres, left Classical unchanged, and lowered F1 for Blues and Reggae.
- Fine-tuning overfits more (98.28% train vs. 74.67% best validation).

## Dataset

[GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

- 1,000 tracks, 30 seconds each, mono WAV at 22,050 Hz
- 10 genres, 100 songs each: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
- Split at the song level: 700 train / 150 validation / 150 test

## Method

1. **Audio to image:** each song becomes a Mel spectrogram (frequency over time on a Mel scale, in dB), saved as a 224 x 224 x 3 image.
2. **ResNet18 body:** the convolution and residual blocks read the spectrogram and extract patterns.
3. **Average pooling:** the body's output is reduced to **512 numbers** per song.
4. **Final layer:** a new 512 -> 10 layer turns those numbers into 10 genre scores. The highest probability is the predicted genre.

### Frozen vs. fine-tuned

Both models use the same forward pass. They differ only in which weights may change during training:

- **Frozen:** the ResNet18 body is locked, so it acts as a fixed feature extractor. Only the new final layer is trained.
- **Fine-tuned:** the body also keeps learning, starting from the pretrained ImageNet weights, so it adapts from photo patterns to spectrogram patterns.

### Training settings (same for both)

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 30 |
| Loss | Cross-entropy (10 classes) |
| Input size | 224 x 224 pixels |

### Evaluation

Accuracy, macro precision, macro recall, macro F1, per-genre F1, and a confusion matrix for each setup.

## Repository contents

| File | Description |
|---|---|
| `ResNet18_Spectrogram_Music_Genre_FULL_with_Results_updated.docx` | Full manuscript |
| `ML1_PPT_updated.pptx` | Presentation slides |
| `[notebook or script name]` | Training and evaluation code (add the filename here) |

## How to run

> Fill this section in with the exact steps for your code.

1. Download the GTZAN dataset and place the genre folders in a local data directory.
2. Generate the Mel spectrogram images from the original 30-second WAV files.
3. Split the songs into train, validation, and test sets at the song level.
4. Train the frozen baseline, then the fine-tuned model, using the settings above.
5. Evaluate both on the held-out test songs.

Requirements: Python, PyTorch and torchvision (pretrained ResNet18), and an audio library for Mel spectrograms. Add exact versions here.

## Limitations

- Only 1,000 songs and 15 test songs per genre, so a single mistake moves a genre's score noticeably. Treat the numbers as a baseline, not a leaderboard.
- No fixed random seed was used, so some run-to-run variation is expected, especially for the fine-tuned model.
- GTZAN is known to contain some mislabeled and duplicate tracks, which may affect results.
- Each song is classified from a single spectrogram, so the model may not capture all temporal and musical characteristics.

## Future work

- Repeat the experiments with several random seeds and report the spread.
- Add dropout, weight decay, and spectrogram augmentation.
- Use an artist-aware split or a cleaned GTZAN list.
- Classify several windows per song and vote, keeping all windows of a song in the same split.

## Google Drive Files
https://drive.google.com/drive/folders/11Dayqxbk4fb60xYeLsdZ7evwZYKvjRJW?usp=sharing
