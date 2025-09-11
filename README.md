# Machine Learning Basics
***
## What is k-Nearest Neighbors (k-NN)?
- A simple and intuitive algorithm for **supervised learning** used in classification and regression.
- It operates by classifying a new data point based on a majority vote of its `k` nearest neighbors in the feature space. The "k" in k-NN is a user-defined constant representing the number of neighbors to consider.
- It is distinct from clustering, as classification works with data that already has known labels, whereas clustering groups data that has no prior labeling.

## Core Concepts of k-NN
- **How it Works**: To classify a new data point, the algorithm identifies the `k` closest training examples in the feature space and assigns the class that is most common among those neighbors.
- **Pros & Cons**:
    - **Pros**: The algorithm is very simple, intuitive, and requires no special preparation or pre-training time.
    - **Cons**: It can be slow and require a large amount of memory and computation time, especially with large datasets, as it needs to store all data instances.
- **Example Implementation (Dog Breed Classification)**:
    - **Goal**: To classify dog breeds (Dachshund vs. Samoyed) based on their body length and height.
    - **Process**: The provided document demonstrates using `scikit-learn` to implement a k-NN classifier. It involves preparing data, creating labels (e.g., 0 for Dachshund, 1 for Samoyed), and then fitting the model.
    - **Prediction**: A new data point is classified by the trained model. For example, a dog with `length=79` and `height=35` is classified as a Dachshund when `k=3`.
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    # Data for Dachshunds (labeled as 0) and Samoyeds (labeled as 1)
    d_data = np.column_stack(([77, 78, 85, 83, 73, 77, 73, 80], [25, 28, 19, 30, 21, 22, 17, 35]))
    d_label = np.zeros(len(d_data))
    s_data = np.column_stack(([75, 77, 86, 86, 79, 83, 83, 88], [56, 57, 50, 53, 60, 53, 49, 61]))
    s_label = np.ones(len(s_data))

    # Combine data and create the model
    dogs = np.concatenate((d_data, s_data))
    labels = np.concatenate((d_label, s_label))
    dog_classes = {0:'Dachshund', 1:'Samoyed'}

    k = 3 
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(dogs, labels)

    # Predict a new data point
    newdata = [[79, 35]]
    y_pred = knn.predict(newdata)
    print('Data', newdata, ', Prediction:', dog_classes[y_pred])
    ```

## Performance Evaluation & Advanced Topics
- **Performance Metrics**:
    - Relying solely on **accuracy** can be misleading, especially with imbalanced datasets, a problem known as **sampling bias**.
    - A **confusion matrix** is used to provide a more detailed evaluation by showing correct and incorrect predictions for each class.
    - Key metrics derived from it include:
        - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
        - **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all actual positives.
        - **F1 Score**: The harmonic mean of Precision and Recall, used as a balanced measure.
- **Ensemble Learning**:
    - A method that combines multiple classifiers to achieve better performance than any single classifier. Its effectiveness relies on the **diversity** of the models.
    - **Techniques**:
        - **Bagging**: Trains multiple models on different random samples of the training data, with replacement.
        - **Pasting**: Similar to bagging but samples the data without replacement.
        - **Boosting**: Trains models sequentially, where each new model is trained to correct the errors of its predecessor.
- **Unsupervised Learning: k-Means Clustering**:
    - An **unsupervised learning** algorithm that groups unlabeled data into a specified number of clusters (`k`).
    - Unlike k-NN, it does not use labeled data for training. Instead, it identifies patterns and groups similar data points together based on their features.
    - The algorithm is simple and effective but requires the number of clusters to be defined in advance.
