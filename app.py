import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

app = Flask(__name__)

# Load the tabular dataset
data = pd.read_csv('asd_data_csv.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
feature_names = data.columns[:-1]  # Extract feature names

# Train the SVM model
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_classifier = SVC(probability=True)
svm_classifier.fit(X_train_scaled, Y_train)

# Load the CNN model
cnn_model = tf.keras.models.load_model('C:/Users/User/my_project/models/cnn_model.h5')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_tabular', methods=['GET', 'POST'])
def predict_tabular():
    if request.method == 'POST':
        features = [float(request.form[feature]) for feature in feature_names]
        prediction = predict_svm(features)
        return render_template('predict_tabular.html', prediction=prediction, feature_names=feature_names)
    return render_template('predict_tabular.html', feature_names=feature_names)

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            prediction = predict_cnn(file_path, cnn_model)
            return render_template('predict_image.html', prediction=prediction)
    return render_template('predict_image.html')

# Function to predict using the SVM model
def predict_svm(tabular_data):
    tabular_data = np.asarray(tabular_data).reshape(1, -1)
    tabular_data = scaler.transform(tabular_data)
    svm_prediction_prob = svm_classifier.predict_proba(tabular_data)[0][1]
    return "The person is with Autism spectrum disorder" if svm_prediction_prob >= 0.5 else "The person is not with Autism spectrum disorder"

# Function to preprocess the input image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict using the CNN model
def predict_cnn(img_path, model):
    img_array = preprocess_image(img_path)
    cnn_prediction_prob = model.predict(img_array)[0][0]
    return "The person is with Autism spectrum disorder" if cnn_prediction_prob >= 0.5 else "The person is not with Autism spectrum disorder"

if __name__ == '__main__':
    app.run(debug=True)
