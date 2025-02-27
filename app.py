import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # Adjusted import for TensorFlow/Keras
from sklearn.preprocessing import LabelEncoder
import IPython.display as ipd

# Function to extract log mel spectrograms
def extract_mel_spectrogram(file, sampling_rate=22050, target_length=128):
    y, sr = librosa.load(file, sr=sampling_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to target length
    if mel_db.shape[1] < target_length:
        pad_width = target_length - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :target_length]

    return mel_db

# Load the trained model
model_path = 'speech_emotion_recognition_model.h5'  # Adjusted path
model = load_model(model_path)

# Define all possible emotions from both datasets
emotions = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Load LabelEncoder with combined emotions
lb = LabelEncoder()
lb.fit(emotions)

# Function to predict emotion
def predict_emotion(file):
    mel_spectrogram = extract_mel_spectrogram(file)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)   # Add batch dimension
    prediction = model.predict(mel_spectrogram)
    predicted_class = np.argmax(prediction)
    predicted_emotion = lb.inverse_transform([predicted_class])
    return predicted_emotion[0], prediction

# function to visualizw the output of the model
def plot_prediction(prediction, emotions):
    fig, ax = plt.subplots()
    ax.bar(emotions, prediction[0])
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Prediction')
    return fig

import sqlite3
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'email' not in st.session_state:
    st.session_state.email = ""

# Function to create a database connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn
# Function to create the signup table
def create_table(conn):
    sql_create_signup_table = """
        CREATE TABLE IF NOT EXISTS signup (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL
        );
    """
    sql_create_feedback_table = """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            feedback TEXT NOT NULL
        );
    """
    try:
        c = conn.cursor()
        c.execute(sql_create_signup_table)
        c.execute(sql_create_feedback_table)
    except sqlite3.Error as e:
        print(e)

# Function to insert signup data into the database
def insert_signup(conn, username, password, email):
    sql = ''' INSERT INTO signup(username, password, email)
              VALUES(?, ?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, (username, password, email))
    conn.commit()
    return cur.lastrowid

# Function to check login credentials
def check_login(conn, username, password):
    sql = ''' SELECT * FROM signup WHERE username = ? AND password = ? '''
    cur = conn.cursor()
    cur.execute(sql, (username, password))
    return cur.fetchone()

# Function to insert feedback into the database
def insert_feedback(conn, username, email, feedback):
    sql = ''' INSERT INTO feedback(username, email, feedback)
              VALUES(?, ?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, (username, email, feedback))
    conn.commit()
    return cur.lastrowid

# Function to check if feedback already exists for the user
def feedback_exists(conn, username, email):
    sql = ''' SELECT * FROM feedback WHERE username = ? AND email = ? '''
    cur = conn.cursor()
    cur.execute(sql, (username, email))
    return cur.fetchone() is not None
# Streamlit interface

# Main function to handle the app logic
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Speech Emotion Recognition App", page_icon=":sound:", layout="wide")

    # Create connection to SQLite database
    conn = create_connection('SER.db')
    create_table(conn)

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px 5px 0 0;
        }
        .nav-links {
            list-style-type: none;
            padding: 0;
            overflow: hidden;
            background-color: #333;
            border-radius: 5px;
        }
        .nav-links li {
            float: left;
        }
        .nav-links li a {
            display: block;
            color: white;
            text-align: center;
            padding: 14px 16px;
            text-decoration: none;
        }
        .nav-links li a:hover {
            background-color: #111;
        }
        .about-content {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )



    # Handle file upload and emotion prediction only when nav choice is "Home"
    nav_choice = st.sidebar.radio(
        "Navigation",
        ["Home","Predict Emotion", "About the Project", "Feedback","About us"]
    )

    if nav_choice == "Home":
        st.markdown('<div class="container">'
            '<div class="header"><h1>Speech Emotion Recognition App</h1></div>'
            '<div class="main-content" id="main-content">'
            '<center><h2>Explore emotions through our Speech Emotion Recognition App!</h2></center>'
            '</div>'
            '</div><br><br>', unsafe_allow_html=True)
        

        home(conn)
    elif(nav_choice=="Predict Emotion"):
        st.markdown('<div class="container">'
            '<div class="header"><h1>Speech Emotion Recognition App</h1></div>'
            '<div class="main-content" id="main-content">'
            '<center><h2>Predict emotion!</h2></center>'
            '</div>'
            '</div><br><br>', unsafe_allow_html=True)
        if st.session_state.logged_in:
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

            if uploaded_file is not None:
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.audio("temp.wav", format='audio/wav')

                if st.button('Predict Emotion'):
                    emotion, prediction = predict_emotion("temp.wav")
                    st.header(f'***Predicted Emotion for uploaded audio is {emotion}***')
                    st.markdown("""
                        ### Visualisation of prediction made by SER model
                    """)
                    fig = plot_prediction(prediction, emotions)
                    st.pyplot(fig)


        else:
            st.error("You are not logged in to our application. Please log in to explore more.")

    elif nav_choice == "About the Project":
        st.markdown('<div class="container">'
            '<div class="header"><h1>Speech Emotion Recognition App</h1></div>'
            '<div class="main-content" id="main-content">'
            '<center><h2>Know about our project!</h2></center>'
            '</div>'
            '</div><br><br>', unsafe_allow_html=True)
        
        # Creating columns for image and text
        col1, col2 = st.columns([1.5, 2])
        
        with col1:
            st.image("block.jpg", caption="", width=300)
        
        with col2:
            st.markdown("""
            Creating an emotion detection model using CNNs and audio clips involves several key steps, each leveraging
different libraries and techniques to preprocess data, extract features, and train the model. The first step is audio
preprocessing, where we load and process the audio data using libraries like Librosa. This involves resampling
the audio to ensure a consistent sampling rate and normalizing the audio signals to maintain uniform loudness
levels. """)
        st.markdown("""
            Feature extraction is a crucial part of this process, where we convert audio signals into visual
representations such as spectrograms or extract Mel-Frequency Cepstral Coefficients (MFCCs). Spectrograms
display the intensity of different frequencies over time, while MFCCs capture the short-term power spectrum of
the audio signal, both essential for capturing the nuances of emotion in the audio.
            """)
        st.markdown(
            """
                For the model architecture, we use Convolutional Neural Networks (CNNs) due to their efficacy in extracting
hierarchical features from 2D data like spectrograms. CNNs are preferred because they can automatically learn
and detect patterns such as edges and textures in the input data, which are vital for distinguishing different
emotions. The CNN architecture typically starts with an input layer that takes the spectrogram or MFCCs as input.
This is followed by several convolutional layers that apply filters to detect local patterns, interspersed with
activation functions like ReLU to introduce non-linearity. Pooling layers are used to reduce the spatial dimensions
of the feature maps, helping to reduce computational complexity and capture invariant features. Fully connected
layers at the end of the network perform high-level reasoning on the extracted features, and the output layer, using
a softmax activation function, provides a probability distribution over the emotion classes.""")
        col1, col2 = st.columns([1.5, 1.5])
        with col1:
            st.image("speech.jpg", caption="CNN Block Diagram", width=400)
        
        with col2:
            st.markdown("""
Training the CNN model requires a labeled dataset of audio clips categorized by emotion, such as TESS. The
model is trained using a loss function like categorical cross-entropy, optimized with algorithms such as Adam or
SGD to minimize the loss and update the network weights. During training, techniques like data augmentation
(adding noise, pitch shifting, time stretching) are used to enhance the dataset and prevent overfitting. The training
process also involves regular evaluation on a validation set, with early stopping to halt training when validation
loss stops improving, ensuring the model does not overfit.""")
        st.markdown("""
Once trained, the model is evaluated on a separate test set to assess its generalization capability, using metrics like
accuracy, precision, recall, and F1-score. For emotion detection during inference, the trained model processes an
input audio clip through the network to output the predicted emotion. The entire pipeline, from preprocessing to
prediction, leverages the strengths of CNNs in handling and extracting features from 2D data representations,
making them particularly suitable for tasks like emotion detection in audio. This approach ensures that the model
effectively captures the complex patterns in audio signals that correspond to different emotions, leading to accurate
and reliable emotion detection.
            """
        )

    elif nav_choice == "Feedback":
        st.markdown('<div class="container">'
            '<div class="header"><h1>Speech Emotion Recognition App</h1></div>'
            '<div class="main-content" id="main-content">'
            '<center><h2>Please give us your feedback about our app!</h2></center>'
            '</div>'
            '</div><br><br>', unsafe_allow_html=True)
        # Check login status
        if st.session_state.logged_in:
            feedback_text = st.text_area("Leave your feedback here")
            if st.button("Submit Feedback"):
                if feedback_text:
                    conn = create_connection('SER.db')
                    with conn:
                        # Check if feedback already exists for the user
                        if feedback_exists(conn, st.session_state.username, st.session_state.email):
                            st.error("You have already submitted feedback.")
                        else:
                            # Insert feedback into the database
                            insert_feedback(conn, st.session_state.username, st.session_state.email, feedback_text)
                            st.success("Thank you for your feedback {}.".format(st.session_state.username))
                else:
                    st.error("Please enter feedback before submitting.")
        else:
            st.error("You must be logged in to submit feedback.")
    if nav_choice == "About us":
        st.markdown(
            """
            <div class="container">
                <div class="header"><h1>Speech Emotion Recognition App</h1></div>
                <div class="main-content" id="main-content">
                    <center><h2>Know about our team!</h2></center>
                </div>
            </div><br><br>
            """,
            unsafe_allow_html=True
        )

        st.markdown("""
        Welcome to the Speech Emotion Recognition App project. This project is designed to help recognize emotions from speech using advanced machine learning algorithms. Our team is dedicated to delivering high-quality, accurate emotion recognition tools.
        """)
        st.header("Meet the Team")
        team_members = [
            {"name": "Umesh Khanal","usn":"1CD21IS188", "email":"khanalumesh14@gmail.com","role":"Project Lead", "photo": "team/umesh.jpg"},
            {"name": "Matrika Dhamala","usn":"1HK21CS074", "email":"1hk21cs074@hkbk.edu.in","photo": "team/matu.jpg"}
        ]

        for member in team_members:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(member["photo"], width=100)
            with col2:
                st.markdown(f"**{member['name']}**")
                st.markdown(f"*USN: {member['usn']}*")
                st.markdown(f"*Email: {member['email']}*")
                if(member['usn']=='1CD21IS188'):
                    st.markdown(f"**Role: {member['role']}**")


# Home interface
def home(conn):
    st.subheader("Login")

    # Login form
    username_login = st.text_input("Username")
    password_login = st.text_input("Password", type="password")

    if st.button("Login"):
        user = check_login(conn, username_login, password_login)
        if user:
            st.success("Logged in as {}".format(username_login))
            st.session_state.logged_in = True  # Set logged_in status in session state
            st.session_state.show_signup = False  # Hide the signup form after successful login
            st.session_state.username = user[1]  # Store username in session state
            st.session_state.email = user[3]  # Store email in session state
        else:
            st.error("Invalid username or password. Please check username and password or try to signup.")
            st.session_state.logged_in = False
            st.session_state.show_signup = True  # Show signup form if login fails

    if st.session_state.show_signup:
        st.subheader("Sign Up")
        # Sign up form
        username_signup = st.text_input("New Username")
        password_signup = st.text_input("New Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        email_signup = st.text_input("Email")

        if st.button("Sign Up"):
            if username_signup and password_signup and email_signup:
                if password_confirm==password_signup:
                    # Insert signup data into the database
                    insert_signup(conn, username_signup, password_signup, email_signup)
                    st.success("Sign up successful! Your username is {}".format(username_signup))
                    st.session_state.show_signup = False  # Hide the signup form after successful signup
                else:
                    st.error("Confirm password is not matching with new password.")
            else:
                st.error("Please fill in all fields.")
# Run the main function
if __name__ == "__main__":
    main()
