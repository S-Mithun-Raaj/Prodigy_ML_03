import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the trained SVM model from file
with open("svm_model.sav", "rb") as f:
    model = pickle.load(f)

# Function to preprocess a single image
def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img.flatten()
    return img

# Function to predict and display an image
def predict_and_display_image(test_img_path, model, img_size):
    # Preprocess the test image
    test_image = preprocess_image(test_img_path, img_size)
    test_image = test_image.reshape(1, -1)  # Reshape to match SVM input shape

    # Predict using the SVM model
    prediction = model.predict(test_image)
    categories = ["Cat", "Dog"]  # Assuming 0 = Cat, 1 = Dog
    predicted_label = categories[prediction[0]]

    # Print prediction to terminal
    print(f"Prediction for {test_img_path}: {predicted_label}")

    # Display the test image and predicted label using Matplotlib
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib display
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()

# Path to the test folder
test_folder_path = r'C:\Users\mithu\Desktop\task3\test1'

# Process each image in the test folder
for filename in os.listdir(test_folder_path):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        test_img_path = os.path.join(test_folder_path, filename)
        predict_and_display_image(test_img_path, model, img_size=50)
