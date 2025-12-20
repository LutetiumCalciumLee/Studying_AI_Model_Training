<details>
<summary>ENG (English Version)</summary>

## **AI Model Training Summary**

**File Overview**: Compilation of Docker, Flask, and ML/DL project practice notes.

### **1. Docker Containerization**
- VM vs Container comparison (containers lighter/faster)
- Docker Desktop, WSL2 installation guide
- Apache HTTP, MariaDB practice (`docker run -p 80:80 httpd`)
- Python dev containers (`python:3.12-slim`)

### **2. Flask Web API Development**
- Basic routing, templates, static file handling
- Iris classifier ML API (`train_model.py` → `app.py`)
- Docker + VS Code Dev Containers integration

### **3. Machine Learning Basics**
- **k-NN**: Nearest neighbor classification (dog breed example)
- Performance metrics: Confusion matrix, Precision/Recall/F1
- Ensemble methods: Bagging, Boosting
- **k-Means**: Unsupervised clustering

### **4. Deep Learning Models**
- **MNIST CNN**: TensorFlow/Keras handwritten digit classification
- **VAE**: Variational Autoencoder for face image generation
- Docker GPU containers (`tensorflow:2.16.2-gpu`)

### **5. Data Preprocessing**
- **PCA**: Dimensionality reduction (Iris 4D → 2D visualization)
- 95% variance preservation demonstrated

**Overall Flow**: Docker environment setup → ML/DL model training → Flask REST API deployment → Web-based real-time prediction service.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## **AI 모델 학습 요약**

**파일 개요**: Docker, Flask, 머신러닝/DL 프로젝트 실습 노트 모음.

### **1. Docker 컨테이너화**
- VM vs 컨테이너 비교 (컨테이너가 더 가볍고 빠름)
- Docker Desktop, WSL2 설치 가이드
- Apache HTTP, MariaDB 실습 (`docker run -p 80:80 httpd`)
- Python 개발 컨테이너 (`python:3.12-slim`)

### **2. Flask 웹 API 개발**
- 기본 라우팅, 템플릿, 정적 파일 처리
- Iris 분류기 ML API (`train_model.py` → `app.py`)
- Docker + VS Code Dev Containers 연동

### **3. 머신러닝 기초**
- **k-NN**: 최근접 이웃 분류 (강아지 품종 예제)
- 성능 평가: 혼동행렬, Precision/Recall/F1
- 앙상블: Bagging, Boosting
- **k-Means**: 비지도 군집화

### **4. 딥러닝 모델**
- **MNIST CNN**: TensorFlow/Keras 손글씨 숫자 분류
- **VAE**: 변분 오토인코더로 얼굴 이미지 생성
- Docker GPU 컨테이너 (`tensorflow:2.16.2-gpu`)

### **5. 데이터 전처리**
- **PCA**: 차원 축소 (Iris 4D → 2D 시각화)
- 95% 분산 보존 효과 입증

**전체 흐름**: Docker 환경 구축 → ML/DL 모델 학습 → Flask REST API 배포 → 웹 기반 실시간 예측 서비스.

</details>
