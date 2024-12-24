# AI-Generated Text Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Short Description

This project implements a deep learning model, specifically a Long Short-Term Memory (LSTM) neural network, to detect AI-generated text. The model distinguishes between human-written and AI-generated content with high accuracy, addressing the growing need for reliable methods to ensure content authenticity in the digital age.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

With the rapid advancement of AI text generation technologies like GPT, BERT, and others, the ability to discern between human-authored and AI-generated text has become increasingly critical. This project tackles this challenge by leveraging advanced Natural Language Processing (NLP) techniques and a powerful LSTM model to classify text as either human-written or AI-generated. Potential applications include content verification, academic integrity, combating misinformation, and ensuring trust in digital content.

## Features

*   **High Accuracy:** Achieves over 94% accuracy in detecting AI-generated text.
*   **LSTM Neural Network:** Employs a robust LSTM architecture to capture long-range dependencies in text.
*   **Advanced NLP Techniques:** Utilizes tokenization, embedding, padding, and other preprocessing steps to prepare text data.
*   **TensorFlow/Keras Implementation:** Built using the popular TensorFlow/Keras deep learning framework.
*   **Comprehensive Evaluation:** Includes detailed performance metrics (accuracy, loss, confusion matrix).
*   **Clear Documentation:** Provides well-documented code and a thorough README for ease of understanding and use.
*   **Scalability**: Optimized for efficiency, allowing for potential scaling to larger datasets and real-world applications.

## Dataset

The model was trained on a dataset comprising 10,000 samples, evenly split between human-written and AI-generated text. Data source was the file named "AI_Human.csv". The dataset was preprocessed using the following steps:

1.  **Text Cleaning:** Removal of unnecessary characters, punctuation, and irrelevant elements.
2.  **Tokenization:** Breaking down text into individual tokens (words or subwords).
3.  **Embedding:** Converting tokens into numerical vectors.
4.  **Padding/Truncation:** Standardizing sequence lengths.

## Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install  pandas numpy matplotlib seaborn scikit-learn tensorflow nltk
    ```

## Usage

1.  **Prepare Data:**
    *   Place your CSV dataset file in a `data` directory within the project.
    *   Ensure the dataset has columns named `text` (for the text content) and `generated` (0 for human, 1 for AI).
2.  **Run the Script:**

    ```bash
    python AIDetection.py 
    ```
    or
    ```bash
        jupyter notebook
    ```

3. **View Results:**
    * The script will output the training/validation progress, as well as the model's accuracy, loss and save the model weights.
    * It will also display and save the confusion matrix.

## Model Architecture

The core of the detection system is an LSTM neural network. Key architectural details:

*   **Embedding Layer:** Converts input text sequences into dense vector representations.
*   **LSTM Layers:** Multiple LSTM layers with dropout for regularization, capturing sequential information and long-range dependencies.
*   **Dense Layers:** Fully connected layers with ReLU activation, followed by a final output layer with a sigmoid activation for binary classification.
*   **Optimizer:** Adam optimizer.
*   **Loss Function:** Binary cross-entropy.

## Training

The model was trained using the following parameters:

*   **Epochs:** 10 (with early stopping)
*   **Batch Size:** 32
*   **Validation Split:** 20%
*   **Optimizer:** Adam
*   **Loss Function:** Sparse categorical crossentropy

The training process employed early stopping to prevent overfitting, monitoring validation loss and restoring the best weights.

## Results

The model achieved the following performance metrics on the test set:

*   **Accuracy:** 94.06%
*   **Loss:** 0.183
*   **Confusion Matrix:**

    ```
                 Predicted Human   Predicted AI
    Actual Human     903             100
    Actual AI        100             897
    ```
*   **Normalized Confusion Matrix:**

    ```
                 Predicted Human   Predicted AI
    Actual Human     90.03\%         9.97\%
    Actual AI        10.03\%         89.97\%
    ```
    The model achieved an overall accuracy of 94%.

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive messages.
4.  Push your branch to your forked repository.
5.  Submit a pull request to the main repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   Libraries: TensorFlow, Keras, scikit-learn, pandas, NumPy, NLTK, Matplotlib, Seaborn.
*   Dataset: [AI_Human.csv]

