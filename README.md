# Intelligent-systems
# Face detection female or male


import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Directory path containing the dataset
dataset_path = r'C:\Users\Francine\OneDrive\Desktop\Bs(Cybersecurity)\Tri 3 (Cybersecurity notes)\Intelligent Systems (CSG2132)\Gender\data'


def load_images_and_extract_features_with_accuracy(dataset_path, svm_classifier):
    images = []
    labels = []
    label_names = ['female']

    category_path = os.path.join(dataset_path, 'female')
    num_images = len(os.listdir(category_path))  # Total number of images in 'female' category
    female_count = 0  # Counter for female images

    correct_female_predictions = 0  # Counter for correct predictions of 'female'

    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Perform LBP feature extraction
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        flattened_lbp = lbp.flatten()
        images.append(flattened_lbp)
        labels.append(0)  # Label 0 for 'female'

        # Predict label for the image using the trained SVM classifier
        predicted_label = svm_classifier.predict([flattened_lbp])[0]

        # Check if the predicted label is 'female' (label 0)
        if predicted_label == 0:
            correct_female_predictions += 1

        # Display the image for 'female' category
        plt.imshow(img, cmap='gray')
        plt.title(f'female - {image_name}')
        plt.axis('off')
        plt.show()

        female_count += 1  # Increment female image count

    accuracy_percentage = (correct_female_predictions / num_images) * 100
    print(f"Percentage of images predicted as female: {accuracy_percentage:.2f}%")

    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images_and_extract_features(dataset_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# Initialize and train Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = svm_classifier.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
