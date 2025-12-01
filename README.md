# Next Word Prediction using LSTM

A deep learning project that implements a Long Short-Term Memory (LSTM) neural network for next word prediction. The model is trained on Sherlock Holmes text and can predict the most likely next word given a sequence of words.

## Overview

This project uses PyTorch to build an LSTM-based language model capable of predicting the next word in a sequence. The model learns patterns and relationships between words by training on literary text, specifically from the Sherlock Holmes stories.

## Features

- **LSTM Neural Network**: Implements a deep learning model with embedding layers and LSTM units
- **Text Preprocessing**: Handles tokenization, vocabulary building, and sequence preparation
- **Custom Dataset**: PyTorch Dataset implementation for efficient data loading
- **Word Prediction**: Generates next word predictions given input text
- **Model Evaluation**: Calculates prediction accuracy on the training dataset
- **Text Generation**: Can generate multiple sequential words to form sentences

## Requirements

```
torch
numpy
nltk
```

## Project Structure

The notebook is organized into the following sections:

1. **Library Imports**: Import necessary PyTorch, NumPy, and NLTK libraries
2. **Data Loading and Preprocessing**: Load text data and clean it by removing Project Gutenberg metadata
3. **Tokenization and Vocabulary Building**: Tokenize text and create word-to-index mappings
4. **Dataset Preparation**: Create custom PyTorch Dataset with input sequences and target words
5. **Model Architecture**: Define the LSTM neural network structure
6. **Model Training**: Train the model using cross-entropy loss and Adam optimizer
7. **Text Generation**: Generate predictions for new input sequences
8. **Model Evaluation**: Calculate accuracy metrics

## Model Architecture

The LSTM model consists of:
- **Embedding Layer**: Converts word indices to dense vector representations
- **LSTM Layer**: Processes sequential data and captures long-term dependencies
- **Fully Connected Layer**: Maps LSTM outputs to vocabulary size for word prediction
- **Dropout**: Regularization to prevent overfitting

## Dataset

The model is trained on text from Sherlock Holmes stories. The preprocessing pipeline:
- Removes Project Gutenberg headers/footers
- Removes commas and splits text by periods
- Tokenizes using NLTK's word_tokenize
- Builds a vocabulary from unique tokens
- Creates sequences of variable length for training

## Training

- **Epochs**: 60
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **GPU**: Trained on Google Colab with A100 GPU
- **Final Loss**: ~2149.91 (Epoch 60)
- **Model Accuracy**: 86.95%

## Usage

### Single Word Prediction

```python
prediction(model, vocab, "I had seen little of", max_len_list)
# Output: 'I had seen little of holmes'
```

### Multi-Word Generation

```python
num_tokens = 10
input_text = "To Sherlock Holmes "

for i in range(num_tokens):
    output_text = prediction(model, vocab, input_text, max_len_list)
    print(output_text)
    input_text = output_text
```

Example output:
```
To Sherlock Holmes she
To Sherlock Holmes she is
To Sherlock Holmes she is always
To Sherlock Holmes she is always _the_
To Sherlock Holmes she is always _the_ woman
```

## Key Functions

- `text_to_indices()`: Converts tokenized text to numerical indices using vocabulary
- `NWPDataset`: Custom PyTorch Dataset class for loading sequences and targets
- `NWPModel`: LSTM neural network model definition
- `prediction()`: Predicts the next word given an input text sequence
- `calculate_accuracy()`: Evaluates model performance on the dataset

## Performance

The model achieves approximately **86.95% accuracy** on the training dataset, demonstrating strong capability in predicting the next word based on learned patterns from the Sherlock Holmes text corpus.

## Hardware Requirements

- **Recommended**: GPU (NVIDIA A100 or similar)
- **Minimum**: CPU (training will be significantly slower)
- **Memory**: At least 8GB RAM recommended

## Future Improvements

- Implement train/validation/test split for better evaluation
- Add temperature sampling for more diverse text generation
- Experiment with deeper LSTM architectures or bidirectional LSTMs
- Fine-tune on different text corpora
- Add beam search for better prediction quality
- Implement perplexity as an additional evaluation metric

## License

This project is for educational purposes. The Sherlock Holmes text is in the public domain.

## Acknowledgments

- Training data sourced from Project Gutenberg
- Built with PyTorch and NLTK
- Developed in Google Colab environment
