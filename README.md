# BRISCOLA_project_CV

Computer vision pipeline for Briscola game videos: card detection (traditional CV, OpenCV/C++) and card classification (deep learning, ResNet18 in `DL/Briscola_DL.ipynb`).

## Requirements

- CMake >= 3.10
- C++20 compiler
- OpenCV 4

## Expected folder layout

```
Briscola/
  game1/ ... game4/         # round videos, named like game3round10.mp4
  Briscola_Trentine/        # scans of the Trentine cards (40 images)
labels/                     # ground truth per game: game<N>output*.txt
output/                     # detection results (created by the program)
dataset/                    # generated crops + CSV labels
```

Paths are resolved from the project root (`PROJECT_SOURCE_DIR`), so the executable can be launched from any directory.

## Build

```bash
cmake -S . -B build
```

```bash
cmake --build build
```

The executable is `build/bin/main`.

## Run modes

### 1. Detection (default)

Processes every round video of a game, in round order, and detects the round leader (North/South) through motion detection, while showing the detected cards.

```bash
./build/bin/main 3
```

- Argument: game number (`1`-`4`). Without arguments game `1` is used.
- Input: `Briscola/game<N>/*.mp4`
- Output: `output/game<N>output.txt` (`Round X` + `Leader: North|South` for each round)
- Windows: `Video Frame`, `Motion`. `ESC` skips the current video; `ESC` during the pause between rounds stops the program.

### 2. Dataset build from game videos

Extracts the card crops from the videos and labels them with the ground truth in `labels/`. No windows are opened (faster).

```bash
./build/bin/main 3 --dataset
```

- Input: `Briscola/game<N>/*.mp4` + `labels/game<N>output*.txt`
- Output:
  - `dataset/images/g<game>_r<round>_<north|south>_f<frame>.png`
  - `dataset/labels_game<N>.csv` (`filename,class_id`)
- Rounds without a label are skipped.

To build the whole dataset:

```bash
for g in 1 2 3 4; do ./build/bin/main $g --dataset; done
```

### 3. Dataset build from Trentine scans

Adds one image per card (40 classes) from the Trentine scans.

```bash
./build/bin/main --trentine
```

- Input: `Briscola/Briscola_Trentine/`
- Output: `dataset/images/` + `dataset/labels_trentine.csv`

`class_id = suit*10 + (rank - 1)`, rank 1-10, suit 0-3 = coins, cups, spades, clubs.

### 4. Card classification (deep learning)

Notebook `DL/Briscola_DL.ipynb`, made for Google Colab (GPU recommended).

1. Open the notebook in Colab.
2. Run all cells: the dataset is downloaded from the GitHub release `dataset-v1`.
3. Training uses leave-one-game-out cross-validation (4 folds, test fold k = game k).
4. The trained models are exported as ONNX: `models/resnet18_fold<k>.onnx`.
