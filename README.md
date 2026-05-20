# How AI Recognizes Faces Using Linear Algebra

This project teaches face recognition from a beginner-friendly linear algebra perspective. Instead of treating AI as a black box, the code shows how a face image becomes a matrix, how that matrix becomes a vector, how many vectors form a dataset matrix, and how algorithms such as Euclidean distance, KNN, and PCA work together in a real recognition pipeline.

The repository is organized concept by concept so you can run each script independently and learn one mathematical idea at a time.

## What You Will Learn

1. An image can be represented as a matrix of pixel intensities.
2. A matrix can be flattened into a 4096-dimensional vector.
3. Many face vectors can be stacked into a dataset matrix.
4. Euclidean distance measures similarity between two face vectors.
5. KNN can classify a face using its nearest neighbors.
6. PCA can reduce a 4096-dimensional vector to a smaller representation.
7. High-dimensional vectors can be visualized in 2D.
8. A webcam pipeline combines detection, preprocessing, PCA, and KNN.
9. Neural networks also rely on linear algebra through matrix multiplication.

## Project Structure

```text
face-recoganition-linear-algebra/
├── concepts/
├── data/
│   └── <person_name>/
├── outputs/
│   ├── images/
│   ├── models/
│   └── plots/
├── shared/
├── requirements.txt
└── README.md
```

## Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

Store images like this:

```text
data/
├── Aryamann/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── Person2/
│   ├── image1.jpg
│   └── image2.jpg
```

Each folder name becomes the class label used by the recognition system.

## How to Collect Images

1. Create one folder per person inside `data/`.
2. Add several clear face photos for each person.
3. Use front-facing images with good lighting when possible.
4. Include at least 2 different identity folders for recognition demos such as KNN, PCA-based live recognition, and 2D clustering.

The early scripts can still run with limited data and will explain what is missing. Full recognition requires at least 2 labeled people.

## How to Run Each Module

Run these commands from the project root:

```bash
python concepts/01_image_as_matrix.py
python concepts/02_flatten_to_vector.py
python concepts/03_build_dataset_matrix.py
python concepts/04_euclidean_distance.py
python concepts/05_knn_face_recognition.py
python concepts/06_pca_dimension_reduction.py
python concepts/07_visualize_vector_space.py
python concepts/08_live_webcam_recognition.py
python concepts/09_neural_network_matrix_demo.py
```

## Concept Guide

### 01 Image as Matrix
- Loads one image.
- Converts it to grayscale.
- Resizes it to `64x64`.
- Shows that a face is a matrix `A in R^(64x64)`.

### 02 Flatten to Vector
- Turns the `64x64` matrix into a `4096`-dimensional vector.
- Demonstrates the core feature representation used by simple ML models.

### 03 Build Dataset Matrix
- Loads all images from the dataset.
- Stacks them into a matrix `X in R^(n x 4096)`.

### 04 Euclidean Distance
- Measures how similar two faces are using:

```text
d(x, y) = sqrt(sum((x_i - y_i)^2))
```

### 05 KNN Face Recognition
- Trains a `KNeighborsClassifier`.
- Predicts a sample identity from the dataset.

### 06 PCA Dimension Reduction
- Reduces 4096 dimensions to a compact representation.
- Saves the PCA model to `outputs/models/pca.pkl`.

### 07 Visualize Vector Space
- Projects face vectors into 2D with PCA.
- Saves a labeled scatter plot to `outputs/plots/07_pca_2d.png`.

### 08 Live Webcam Recognition
- Detects faces with OpenCV Haar cascades.
- Preprocesses each face to `64x64`.
- Flattens to `4096` features.
- Transforms the vector with PCA.
- Predicts identity with KNN.
- Displays overlays for matrix size, vector length, PCA size, identity, and nearest-neighbor distance.

If saved models are missing, the webcam script trains them automatically. If a face is too far from the known training set, it is labeled `Unknown`.

### 09 Neural Network Matrix Demo
- Shows how one neural network layer computes:

```text
y = W x + b
```

- Prints shapes and saves a compact visualization of the matrices.

## Output Files

The scripts automatically create:

- `outputs/images/` for saved demo figures
- `outputs/plots/` for PCA scatter plots
- `outputs/models/` for saved PCA and KNN models

## Video Explanation Summary

This project follows the same story you might tell in a teaching video:

1. A face starts as a grid of numbers.
2. That grid is flattened into a long vector.
3. Many vectors are stacked into a dataset matrix.
4. Distances between vectors tell us which faces look similar.
5. KNN uses nearby examples to predict identity.
6. PCA compresses the vector while keeping important information.
7. The webcam pipeline applies the same math live to real camera input.

## Notes for Beginners

- If a script says the dataset is missing or too small, add more images under `data/<person_name>/`.
- Unreadable files are skipped automatically with a console message.
- The webcam demo depends on local camera permissions and available hardware.
- Saved models in `outputs/models/` are generated automatically by the training scripts.

