# Voice Emotion Classification

## Objective
This project aims to classify emotions from voice recordings using deep learning techniques. The model processes audio data to predict emotions such as fear, neutrality, and others by extracting meaningful features from the audio files.

## Features
- Classifies emotions from voice recordings into 7 categories.
- Uses Mel-frequency cepstral coefficients (MFCCs) for feature extraction.
- Deep Learning model with LSTM layers for emotion classification.
- Includes visualization of waveforms, spectrograms, and emotion distributions.

## Dataset
- The dataset consists of audio files stored in directories. Each file is labeled based on the emotion it conveys, extracted from the file name.
- Emotions include fear, neutrality, and other common emotional states.

## Methodology

### 1. Data Loading and Preprocessing
- Paths to audio files are collected, and labels are derived from the file names.
- A pandas DataFrame is created with columns for speech (file path) and label (emotion).
- A countplot visualizes the distribution of emotions.

### 2. Exploratory Data Analysis
- Waveforms and spectrograms for emotions like fear and neutral are plotted to understand their frequency and time-domain characteristics.

### 3. Feature Extraction
- Mel-frequency cepstral coefficients (MFCCs) are extracted from the audio files using `librosa`.
- 40 MFCC coefficients are computed per file, averaged across the time axis for a fixed-length feature vector.

### 4. Data Preparation
- Input features are stored as NumPy arrays with shape `(number of samples, 40, 1)`, preparing the data for the LSTM model.
- Labels are one-hot encoded to convert categorical labels into numerical arrays.

### 5. Model Architecture
- The model is a Sequential deep learning model with:
  - LSTM layer (123 units) to capture temporal dependencies.
  - Dense layers (64 and 32 units) with ReLU activation.
  - Dropout layers (20%) to prevent overfitting.
  - Output layer with softmax activation to classify emotions into 7 categories.
- Compiled with:
  - Loss function: `categorical_crossentropy`.
  - Optimizer: `Adam`.
  - Metric: `Accuracy`.

### 6. Training
- The model was trained for 10 epochs with a batch size of 51.
- Validation split: 20% of the data was used for validation.

## Results

### 1. Training and Validation Accuracy
- The model showed a gradual improvement in both training and validation accuracy.
- Final training accuracy reached approximately 90%, while validation accuracy stabilized around 88%.

### 2. Training and Validation Loss
- Loss decreased consistently during training, with validation loss remaining close to the training loss, indicating minimal overfitting.

### 3. Observations
- The model performed well in classifying emotions based on MFCC features.
- Some closely related emotions (e.g., fear vs. surprise) posed challenges due to overlapping audio characteristics.

## Conclusions

1. **Model Performance**: The model achieved high accuracy in classifying emotions from voice data.
2. **Key Insight**: MFCC features were robust for emotion classification, and the LSTM architecture effectively captured temporal patterns in the audio.
3. **Future Work**: 
   - Enhance the dataset with more samples and diverse audio recordings.
   - Experiment with additional features like chroma or spectral contrast for improved classification.
   - Deploy the model for real-world applications, such as virtual assistants or customer service analytics.
4. **Challenges**: 
   - Closely related emotions may require more nuanced features or larger datasets.
   - The performance depends on the quality of audio recordings and preprocessing steps.


