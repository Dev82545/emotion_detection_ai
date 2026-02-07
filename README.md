Speech Emotion Recognition using CNNs

This project is an implementation of a Speech Emotion Recognition (SER) system that predicts human emotions from short speech recordings. The main idea is to convert audio signals into log-Mel spectrograms and treat them as images, which are then classified using a Convolutional Neural Network (CNN).
The project covers the full pipeline — from dataset preparation and feature extraction to model training, evaluation, and live inference.

What this project does:
Given a .wav audio file (around 3 seconds long), the system predicts one of the following emotions:

neutral, calm, happy, sad, angry, fearful, disgust, surprised

The focus of this project is not just model accuracy, but also:

•	clean data handling

•	reproducible training

•	clear evaluation

•	usable inference code

________________________________________
Dataset

The model is trained on the RAVDESS dataset, which contains emotional speech recordings from professional actors.
Key points about the dataset:

•	Balanced across 8 emotion classes

•	Contains both male and female speakers

•	All clips are short and clean (around 3 seconds)

•	Audio is sampled at 22,050 Hz

The dataset itself is not included in the repository.

________________________________________
Feature extraction:

Raw audio cannot be fed directly into a CNN, so each audio clip is processed. All audio is first trimmed to remove still spaces from it, then the audio is parsed and padded. After everything the audio is augmented to make sure there is no overfitting done by the model and support model generalization. This audio is then used to create a Mel Spectrogram converted to lg scale using “librosa.power_to_db.
This results in a fixed-size log-Mel spectrogram, which captures frequency and temporal information relevant to emotion.
________________________________________
Data augmentation:

To improve robustness and reduce overfitting, data augmentation is applied only to the training set.
The following augmentations are used:

•	Additive noise

•	Pitch shifting

•	Time stretching

Each augmented sample keeps the same emotion and gender label as the original sample.
________________________________________
Model architecture:

The model is a CNN designed specifically for spectrogram inputs.
Main design choices:

•	Convolutional layers for spatial feature extraction

•	ReLU activations

•	Global Average Pooling instead of large fully connected layers

•	Final linear layer for emotion classification

Global Average Pooling helps reduce overfitting and keeps the model lightweight.
________________________________________
Training process:

•	Data is split into 80% training, 10% validation, and 10% test, stratified by emotion.

•	Cross-entropy loss is used.

•	Adam optimizer is used with a fixed learning rate.

•	Training loss is tracked across epochs and visualized in the notebook.

The goal during training was stable convergence rather than aggressive overfitting.
________________________________________
Evaluation:

Model performance is evaluated on the held-out test set using:

•	Overall accuracy

•	Macro F1 score

•	Confusion matrix

•	Gender-wise Macro F1 (when applicable)

All evaluation results, plots, and observations are documented in the notebook:

notebooks/SER_EDA_Training_Evaluation.ipynb
________________________________________
Saved model:

The best-performing model (based on validation Macro F1 score) is saved as:
cnn_model/ser_cnn_best.pth

This file contains only the trained weights and is used for inference.
________________________________________
Inference:

Live emotion prediction can be performed using:
python predict.py --wav path/to/audio.wav --model cnn_model/ser_cnn_best.pth

The script outputs:

•	The predicted emotion

•	Probability scores for all emotion classes
________________________________________
Project structure:

EMOTION_DETECTION_AI/

├── cnn_model/            # Model architecture, validation, evaluation

├── dataset_builder/     # Dataset creation and augmentation

├── mel_spectrogram/     # Log-Mel feature extraction

├── notebooks/           # EDA, training curves, evaluation

├── data/                # Dataset (not included)

├── predict.py           # Inference script

├── config.py            # Shared configuration

├── requirements.txt     # Dependencies

└── README.md
________________________________________
How to run the project:

1.	Create and activate a virtual environment

2.	Install dependencies:

3.	pip install -r requirements.txt

4.	Run training or evaluation scripts from the project root

5.	Use predict.py for inference

6.	Open the notebook for analysis and visualization
________________________________________
Limitations and future work:

•	Emotion recognition from speech is inherently subjective

•	Some emotions are acoustically similar and get confused

•	Performance could be improved with:

o	pretrained audio models (wav2vec 2.0)

o	temporal models (CRNN / Transformers)

o	speaker normalization

o	real-time microphone inference


Final notes

This project was built to understand the complete lifecycle of an ML system:
data preparation, feature engineering, model design, evaluation, and deployment-ready inference.
The emphasis was on clarity, correctness, and reproducibility, rather than chasing benchmark scores
