<details>
<summary>ENG (English Version)</summary>

## **Machine Learning Basics – k-NN and Beyond**

### **What is k-Nearest Neighbors (k-NN)?**
- A simple **supervised learning** algorithm used for classification and regression.
- Classifies a new data point based on the majority label among its `k` nearest neighbors.
- The value of `k` is user-defined.
- Different from clustering: classification uses labeled data, while clustering does not.

### **Core Concepts of k-NN**
- **How It Works**:
  - Computes distances between a new point and all training samples.
  - Selects the `k` closest samples.
  - Assigns the most frequent class among them.
- **Pros**:
  - Easy to understand and implement.
  - No training phase required.
- **Cons**:
  - Slow with large datasets.
  - High memory and computation cost.

### **Example: Dog Breed Classification**
- **Goal**: Classify dog breeds (Dachshund vs. Samoyed) using body length and height.
- **Method**: Use `scikit-learn`’s `KNeighborsClassifier`.
- **Labels**:
  - Dachshund → 0
  - Samoyed → 1
- **Prediction Example**:
  - Input: `length=79`, `height=35`
  - Result (k=3): Dachshund

### **Performance Evaluation**
- **Accuracy Limitations**:
  - Can be misleading with imbalanced datasets (sampling bias).
- **Confusion Matrix**:
  - Shows true/false positives and negatives.
- **Key Metrics**:
  - **Precision**: Correct positive predictions / all positive predictions.
  - **Recall**: Correct positive predictions / all actual positives.
  - **F1 Score**: Harmonic mean of precision and recall.

### **Ensemble Learning**
- Combines multiple models to improve performance.
- Effectiveness depends on **model diversity**.
- **Techniques**:
  - **Bagging**: Sampling with replacement.
  - **Pasting**: Sampling without replacement.
  - **Boosting**: Sequential learning focusing on previous errors.

### **Unsupervised Learning: k-Means**
- An **unsupervised learning** algorithm.
- Groups unlabeled data into `k` clusters.
- Unlike k-NN, no labels are used.
- Simple and efficient, but `k` must be predefined.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## **머신러닝 기초 – k-NN과 확장 개념**

### **k-최근접 이웃(k-NN)이란?**
- 분류와 회귀에 사용되는 **지도학습 알고리즘**.
- 새로운 데이터 포인트를 가장 가까운 `k`개의 이웃 데이터의 다수결로 분류한다.
- `k` 값은 사용자가 직접 지정한다.
- 군집화와 달리, 이미 **정답 라벨이 있는 데이터**를 사용한다.

### **k-NN의 핵심 개념**
- **동작 원리**:
  - 새 데이터와 모든 학습 데이터 간의 거리를 계산.
  - 가장 가까운 `k`개 이웃 선택.
  - 가장 많이 등장한 클래스로 분류.
- **장점**:
  - 구조가 단순하고 직관적.
  - 별도의 학습 과정이 필요 없음.
- **단점**:
  - 데이터가 많을수록 속도가 느림.
  - 메모리와 계산 비용이 큼.

### **예제: 강아지 품종 분류**
- **목표**: 몸 길이와 키를 이용해 닥스훈트와 사모예드를 분류.
- **방법**: `scikit-learn`의 `KNeighborsClassifier` 사용.
- **라벨 설정**:
  - 닥스훈트 → 0
  - 사모예드 → 1
- **예측 예시**:
  - 입력값: 길이 79, 키 35
  - 결과(k=3): 닥스훈트

### **성능 평가**
- **정확도(Accuracy)의 한계**:
  - 데이터 불균형이 있으면 왜곡 가능(샘플링 편향).
- **혼동 행렬(Confusion Matrix)**:
  - 분류 결과를 상세히 분석 가능.
- **주요 지표**:
  - **정밀도(Precision)**: 예측한 양성 중 실제 양성 비율.
  - **재현율(Recall)**: 실제 양성 중 맞게 예측한 비율.
  - **F1 점수**: 정밀도와 재현율의 조화 평균.

### **앙상블 학습**
- 여러 모델을 결합해 단일 모델보다 성능을 향상시키는 기법.
- 핵심 요소는 **모델의 다양성**.
- **대표 기법**:
  - **배깅(Bagging)**: 중복 허용 샘플링.
  - **페이스팅(Pasting)**: 중복 없이 샘플링.
  - **부스팅(Boosting)**: 이전 모델의 오답을 보완하며 순차 학습.

### **비지도 학습: k-평균(k-Means) 군집화**
- **비지도 학습 알고리즘**으로 라벨 없는 데이터를 그룹화.
- 미리 지정한 `k`개의 군집으로 데이터를 분류.
- k-NN과 달리 정답 데이터 없이 패턴을 찾는다.
- 단순하고 효과적이지만, 군집 개수를 사전에 정해야 한다.

</details>
