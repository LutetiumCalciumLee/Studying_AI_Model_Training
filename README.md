# Python-based API Server Development
***
## Core Workflow: TensorFlow-Keras and Flask
- This chapter describes the process of developing a deep learning model with TensorFlow-Keras and deploying it as a web service using the Flask framework.
- The overall workflow involves :
    1.  **Model Training**: A machine learning model is trained using TensorFlow-Keras and its weights are saved.
    2.  **Flask Server**: A Flask web server loads the pre-trained model.
    3.  **API Endpoint**: The server exposes an API endpoint (e.g., `/predict`) that accepts data, processes it, and returns the model's prediction.
    4.  **Client**: A client application (like a web page) sends requests to the API and displays the results.

## Environment Setup
- The document outlines setting up the development environment using **Docker**, which allows for containerizing the application and its dependencies.
- It provides instructions for running a TensorFlow container with GPU support, mapping ports, and mounting volumes to share files between the host machine and the container.
- **Key command for running the Docker container** :
    ```bash
    docker run -it -d --gpus all -p 80:5000 --name tensorflow tensorflow/tensorflow:2.16.2-gpu /bin/bash
    ```
- It also mentions installing necessary packages like `matplotlib` and `python3-tk` inside the container for plotting and graphical interfaces.

## Model Training and Implementation (MNIST)
- **Goal**: To train a Convolutional Neural Network (CNN) to classify handwritten digits from the **MNIST dataset**.
- **Process**:
    - **Data Loading & Preprocessing**: The MNIST dataset is loaded using `keras.datasets.mnist.load_data()`, and the image data is normalized by dividing pixel values by 255.0.
    - **Model Building**: A sequential model is constructed using Keras with `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
    - **Training**: The model is compiled with an 'adam' optimizer and 'sparse_categorical_crossentropy' loss function. It is then trained using the `model.fit()` method, with callbacks for saving the best model weights (`ModelCheckpoint`) and stopping early if performance plateaus (`EarlyStopping`).
    - **Evaluation**: The trained model is evaluated on the test dataset to check its accuracy.
    ```python
    # Simplified model building and training flow
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Build model
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile and train
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # Save model
    model.save('model.weights.h5')
    ```

## Flask API Server Implementation
- **Purpose**: To create a web API that receives an image and returns a prediction from the trained MNIST model.
- **Key Components**:
    - **`app.py`**: The main Flask application file. It loads the saved Keras model and defines the routes.
    - **`utils.py`**: Contains helper functions for preprocessing the input image to match the model's input requirements (e.g., resizing, converting to grayscale) and for post-processing the model's output.
    - **Routes**:
        - `@app.route('/', methods=['GET'])`: Renders the main HTML page (`index.html`).
        - `@app.route('/predict', methods=['POST'])`: Handles file uploads. It receives an image, preprocesses it, passes it to the model for prediction, post-processes the result, and returns the prediction as a JSON response.
- **Example Flask Code Snippet** :
    ```python
    from flask import Flask, request, jsonify, render_template
    import tensorflow as tf
    from utils import preprocess_image, postprocess_prediction

    app = Flask(__name__)
    model = tf.keras.models.load_model('model.weights.h5')

    @app.route('/predict', methods=['POST'])
    def predict():
        file = request.files['file']
        if not file:
            return jsonify({'error': 'no file'}), 400
        
        # Preprocess the image and get prediction
        img_array = preprocess_image(file)
        prediction = model.predict(img_array)
        result = postprocess_prediction(prediction)
        
        return jsonify(result)

    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)
    ```
---
# Principal Component Analysis (PCA)
***
## What is Principal Component Analysis (PCA)?
- PCA is a **dimensionality reduction** technique used to transform a large set of variables into a smaller one that still contains most of the information in the large set.
- It works by identifying new variables, called **principal components (PCs)**, which are linear combinations of the original variables. The first principal component (PC1) accounts for the largest possible variance in the data, the second (PC2) for the second-largest, and so on.
- The goal is to reduce the number of dimensions in a dataset while minimizing information loss, which is useful for visualization and improving machine learning model performance.

## Core Concepts of PCA
- **Variance and Covariance**: PCA relies on the covariance matrix of the data to find the principal components. The directions of the principal components are the eigenvectors of this matrix.
- **Data Scaling**: Before applying PCA, it is crucial to **standardize** the data (i.e., give it a mean of 0 and a variance of 1) by using a tool like `StandardScaler`. This is because PCA is sensitive to the scale of the features.
- **Scree Plot**: A scree plot is used to determine how many principal components to keep. It plots the explained variance for each principal component. Often, the "elbow point" of the plot is used to select the number of components that capture a significant amount of variance (e.g., >90%).
- **Loading Plot**: This plot shows how the original variables contribute to the principal components. It helps in interpreting what the principal components represent in the context of the original features.

## Python Implementation with `scikit-learn`
- **Goal**: To apply PCA on a dataset to reduce its dimensionality.
- **Process**:
    1.  **Load Data**: Load the dataset, for example, using `pandas`.
    2.  **Standardize Data**: Use `StandardScaler` from `scikit-learn` to scale the features.
    3.  **Apply PCA**: Create an instance of the `PCA` class from `sklearn.decomposition`. The `n_components` parameter can be set to the desired number of dimensions.
    4.  **Transform Data**: Use the `fit_transform()` method to apply PCA to the standardized data.
- **Example Code Snippet** :
    ```python
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Example data with 2 features
    X = np.array([[50, 73], [65, 75], [75, 80], [80, 82], [95, 85]])

    # 1. Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 2. Apply PCA
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_std)

    # Explained variance ratio
    print(pca.explained_variance_ratio_)
    # Output might be: [0.988 0.011]
    # This means PC1 explains 98.8% of the variance

    # Plot the transformed data
    plt.scatter(Z[:,0], Z[:,1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    ```

## Example Application: Iris Dataset
- The document demonstrates applying PCA to the well-known **Iris dataset**, which has 4 features (sepal length, sepal width, petal length, petal width) and 3 target classes (setosa, versicolor, virginica).
- By applying PCA, the 4-dimensional feature space is reduced to 2 dimensions (PC1 and PC2). A scatter plot of these two components shows that the three species of iris form distinct clusters, demonstrating how PCA can be used for data visualization and to see patterns in high-dimensional data.
- After applying PCA, PC1 and PC2 together explain over 95% of the total variance, meaning that the two components effectively represent the original 4 features with minimal information loss.
