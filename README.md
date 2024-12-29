# AI-Generated vs Real Images Classification and Visualization

This project implements various techniques to classify and visualize AI-generated images and real-world images using both traditional machine learning models and deep learning models. It also includes functionalities for image processing, data augmentation, model evaluation, and result visualization.

---

## Features

### 1. **Data Loading and Visualization**
- Loads images from directories, categorizes them into `AI-generated` and `Real Art`.
- Supports `.jpg`, `.png`, and `.jpeg` formats.
- Displays a grid of randomly selected images from the dataset.

### 2. **Traditional Machine Learning Models**
- Logistic Regression and Linear Discriminant Analysis (LDA) models for classification.
- Evaluates accuracy and log loss for both models.

### 3. **Deep Learning Models**
- Custom CNN model using TensorFlow/Keras for binary classification.
- Transfer learning using MobileNetV2 for enhanced classification accuracy.

### 4. **Metrics and Evaluation**
- Tracks metrics like accuracy, loss, and F1 score during training and validation.
- Visualizes learning curves for loss, accuracy, and F1 scores.

### 5. **Interactive Predictions**
- Allows image upload for real-time predictions using the trained models.
- Predicts whether the uploaded image is `AI-generated` or `Real Art`.

### 6. **PyTorch Model**
- Custom dataset and data loader implementation with PyTorch.
- Training and validation using the `rexnet_150` model for multi-class classification.
- Tracks and visualizes learning progress.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Kaggle or Colab environment (optional for dataset compatibility)
- Required Python libraries:
  - `numpy`, `pandas`, `matplotlib`, `torch`, `timm`, `torchmetrics`, `tensorflow`
  - `pillow`, `scikit-learn`, `tqdm`

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For PyTorch models, install additional dependencies:
   ```bash
   pip install timm torchmetrics
   ```

4. Ensure your environment has Tesseract OCR installed for OCR tasks (optional):
   ```bash
   sudo apt-get install tesseract-ocr
   ```

---

## Usage

### 1. **Dataset Preparation**
- Place the images in the following directory structure:
  ```
  root/
  ├── AiArtData/
  │   ├── image1.jpg
  │   ├── ...
  ├── RealArt/
      ├── image1.jpg
      ├── ...
  ```

### 2. **Run the Script**
- **Visualization:**
  ```python
  python visualize_images.py
  ```
  This script displays a grid of randomly selected images.

- **Traditional Models:**
  ```python
  python train_ml_models.py
  ```
  Train Logistic Regression and LDA models and evaluate their performance.

- **Deep Learning Models:**
  ```python
  python train_dl_models.py
  ```
  Train CNN and MobileNetV2 models for classification.

- **Interactive Prediction:**
  ```python
  python upload_and_predict.py
  ```
  Upload an image and classify it using trained models.

- **PyTorch Training:**
  ```python
  python train_pytorch_models.py
  ```
  Train a PyTorch-based model using the dataset.

---

## Results and Visualization
- **Learning Curves:** Visualize training and validation loss, accuracy, and F1 scores.
- **Bar Charts:** Compare accuracy and loss across models.


---

## Contributions
Contributions are welcome! Feel free to fork the repository, submit a pull request, or open an issue for suggestions or bugs.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- Kaggle dataset for AI-generated and real images.
- TensorFlow, PyTorch, and scikit-learn for machine learning and deep learning implementations.

```

Copy and paste this into a `README.md` file in your repository. Adjust paths and file names as necessary.
