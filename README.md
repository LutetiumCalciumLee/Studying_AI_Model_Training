<details>
<summary>ENG (English Version)</summary>

## **Python-based API Server Development 2**

### **Overall Workflow (TensorFlow-Keras + Flask)**
- Develop a deep learning model with TensorFlow-Keras and deploy it as a web API using Flask.
- Workflow:
  1. Train a model and save weights.
  2. Load the trained model in a Flask server.
  3. Expose a prediction API endpoint (e.g., `/predict`).
  4. Client sends data and receives prediction results.

### **Environment Setup with Docker**
- Use Docker to containerize the TensorFlow and Flask environment.
- GPU-enabled TensorFlow container is used for training and inference.
- Example command:
```bash
docker run -it -d --gpus all -p 80:5000 --name tensorflow tensorflow/tensorflow:2.16.2-gpu /bin/bash
````

* Additional libraries such as `matplotlib` and `python3-tk` are installed for visualization.

### **Model Training: MNIST CNN**

* **Goal**: Classify handwritten digits (0–9) from the MNIST dataset.
* **Steps**:

  * Load and normalize image data.
  * Build a CNN with Conv2D, MaxPooling, Dense layers.
  * Compile with Adam optimizer and categorical cross-entropy loss.
  * Train the model and save the weights.
* The trained model achieves accuracy validation on test data.

### **Flask API Server**

* **Purpose**: Serve predictions from the trained MNIST model.
* **Key Files**:

  * `app.py`: Flask server, model loading, route definitions.
  * `utils.py`: Image preprocessing and prediction postprocessing.
* **Main Routes**:

  * `/` (GET): Render main HTML page.
  * `/predict` (POST): Receive image, preprocess, predict, return JSON result.
* Enables real-time inference via HTTP requests.


## **Principal Component Analysis (PCA)**

### **What is PCA?**

* A dimensionality reduction technique.
* Transforms original features into fewer **principal components** while preserving most variance.
* Useful for visualization and improving model performance.

### **Core Concepts**

* **Variance & Covariance**: PCA is based on eigenvectors of the covariance matrix.
* **Standardization**: Features must be scaled before PCA.
* **Scree Plot**: Helps decide the number of components.
* **Loading Plot**: Shows feature contributions to each component.

### **Python Implementation**

* Use `StandardScaler` and `PCA` from `scikit-learn`.
* Apply `fit_transform()` to reduce dimensions.
* Explained variance ratio indicates how much information is preserved.

### **Application: Iris Dataset**

* Original 4D feature space reduced to 2D (PC1, PC2).
* Visualization shows clear clustering of species.
* PC1 + PC2 explain over 95% of total variance.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## **파이썬 기반 API 서버 개발 2**

### **전체 흐름 (TensorFlow-Keras + Flask)**

* TensorFlow-Keras로 딥러닝 모델을 학습한 뒤 Flask로 웹 API 형태로 배포한다.
* 개발 흐름:

  1. 딥러닝 모델 학습 및 가중치 저장
  2. Flask 서버에서 학습된 모델 로드
  3. `/predict`와 같은 API 엔드포인트 제공
  4. 클라이언트가 데이터를 보내고 예측 결과 수신

### **Docker 기반 환경 구성**

* Docker를 사용해 개발·실행 환경을 컨테이너화한다.
* GPU를 활용하는 TensorFlow 컨테이너 사용.
* 주요 실행 명령:

```bash
docker run -it -d --gpus all -p 80:5000 --name tensorflow tensorflow/tensorflow:2.16.2-gpu /bin/bash
```

* 시각화를 위해 `matplotlib`, `python3-tk` 등 추가 패키지 설치.

### **모델 학습: MNIST CNN**

* **목표**: MNIST 손글씨 숫자(0~9) 분류.
* **과정**:

  * 데이터 로드 후 정규화(0~1).
  * CNN 모델 구성(합성곱, 풀링, 완전연결 계층).
  * Adam 옵티마이저와 손실 함수 설정.
  * 모델 학습 후 가중치 저장.
* 테스트 데이터를 통해 정확도 평가 수행.

### **Flask API 서버 구현**

* **목적**: 학습된 MNIST 모델을 웹 API로 제공.
* **구성 파일**:

  * `app.py`: Flask 서버, 모델 로딩, 라우팅 정의.
  * `utils.py`: 이미지 전처리 및 결과 후처리 함수.
* **주요 라우트**:

  * `/` (GET): 메인 HTML 페이지 렌더링.
  * `/predict` (POST): 이미지 업로드 → 전처리 → 예측 → JSON 응답.
* 웹 기반 실시간 추론 환경 구현.


## **주성분 분석 (PCA)**

### **PCA란?**

* 차원 축소 기법으로, 정보 손실을 최소화하면서 변수 개수를 줄인다.
* 새로운 축인 **주성분(Principal Component)**을 생성한다.
* 데이터 시각화 및 모델 성능 개선에 활용된다.

### **핵심 개념**

* **분산·공분산**: 공분산 행렬의 고유벡터가 주성분 방향.
* **표준화**: PCA 적용 전 반드시 데이터 스케일링 필요.
* **스크리 플롯**: 유지할 주성분 개수 결정.
* **로딩 플롯**: 원본 변수와 주성분의 관계 해석.

### **Python 구현**

* `StandardScaler`로 데이터 표준화.
* `PCA` 객체로 차원 축소 수행.
* `explained_variance_ratio_`로 정보 보존 비율 확인.

### **Iris 데이터셋 적용 예**

* 4차원 특징을 2차원(PC1, PC2)으로 축소.
* 산점도에서 품종별 군집이 명확히 구분됨.
* PC1과 PC2가 전체 분산의 95% 이상을 설명.

</details>
