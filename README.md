# Speech Emotion Recognition App 🎙️🎭

## Overview  
This project is a **Speech Emotion Recognition (SER) App** that analyzes and classifies emotions from speech using **Deep Learning** and **Convolutional Neural Networks (CNNs)**. The app provides real-time predictions for uploaded audio files and visualizes the detected emotions.

## Features  
✅ Predicts emotions such as **happy, sad, angry, fear, neutral, surprised, calm, and disgust**  
✅ Uses **Mel Spectrograms** for audio feature extraction  
✅ Built with **Streamlit** for an interactive web interface  
✅ Stores user feedback in **SQLite database**  
✅ Supports **.wav audio files** for prediction  

## Technologies Used  
- **Python** (for AI model development)  
- **TensorFlow/Keras** (for Deep Learning)  
- **Librosa** (for audio feature extraction)  
- **Streamlit** (for web application)  
- **SQLite** (for database management)  

## Installation & Setup  
To run the project locally, follow these steps:  

### 1. Clone the Repository  
```sh
git clone https://github.com/umeshkhanal/SpeechEmotionRecognitionApp.git
cd SpeechEmotionRecognitionApp
```

### 2. Install Dependencies  
```sh
pip install -r requirements.txt
```

### 3. Run the Application  
```sh
streamlit run app.py
```

## How It Works  
1. **Upload an audio file (.wav)**  
2. **App processes the audio and extracts features**  
3. **CNN model predicts the emotion**  
4. **Results are displayed with a probability chart**  


---  
### Developed by Umesh Khanal  
If you have any questions or suggestions, feel free to reach out! 🚀


