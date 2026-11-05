<details>
<summary>ENG (English Version)</summary>

# Week 03 — MNIST Digit Recognition with TensorFlow and Flask, and PCA

CNN training in Docker, handwritten digit prediction through a Flask web application, and dimensionality reduction of the Iris dataset with PCA.

## 1. Project Overview

Use of the Docker development environment from Week 01 and the Flask model integration from Week 02. Extension of the prediction workflow from numerical measurements for species classification to image uploads for digit classification.

```text
MNIST → CNN training and validation → Saved Keras model

Browser → Image upload → Flask → Preprocessing → Model inference → JSON response
```

| Stage | Implementation |
|---|---|
| Model training | Classification of digits 0–9 with a TensorFlow/Keras CNN |
| Model storage | Full model storage at `models/mnist.keras` |
| Image preprocessing | Grayscale conversion, inversion, black padding, and normalization |
| API | Image upload and JSON response through `POST /predict` |
| Web page | Image upload and result display using HTML, CSS, and JavaScript |
| PCA exercise | Projection of standardized Iris features onto two principal components |

---

## 2. Docker Development Environment

**PowerShell** for Windows commands and **Bash inside the container** for Python commands.

### 2.1. Container Creation

Execution of either the GPU or CPU command from the project directory in PowerShell. NVIDIA GPU and Docker GPU access required for the GPU option.

**GPU Environment**

```powershell
docker run -dit --gpus all --name mnist-api -p 127.0.0.1:5000:5000 -v "${PWD}:/workspace" -w /workspace tensorflow/tensorflow:2.16.2-gpu bash
```

**CPU Environment**

```powershell
docker run -dit --name mnist-api -p 127.0.0.1:5000:5000 -v "${PWD}:/workspace" -w /workspace tensorflow/tensorflow:2.16.2 bash
```

| Option | Meaning |
|---|---|
| `--gpus all` | GPU access inside the container |
| `-p 127.0.0.1:5000:5000` | Mapping of local port 5000 to container port 5000 |
| `-v "${PWD}:/workspace"` | Sharing of the current project directory with `/workspace` |
| `-w /workspace` | Assignment of the container working directory |

Shell access inside the running container.

```powershell
docker exec -it mnist-api bash
```

### 2.2. Project Structure and Dependency Installation

Creation of the example files in the following structure. Generation of `models/mnist.keras` and `iris_pca.png` through the training and PCA scripts, respectively.

```text
project/
├── requirements.txt
├── train.py
├── utils.py
├── app.py
├── pca_iris.py
├── models/
│   └── mnist.keras
├── templates/
│   └── index.html
└── static/
    └── style.css
```

Additional dependencies in `requirements.txt`. TensorFlow and NumPy already included in the container image.

```text
Flask
Pillow
scikit-learn
matplotlib
```

Package installation and TensorFlow environment verification inside the container.

```bash
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## 3. MNIST Digit Classification Model Training

Normalization of `28 × 28` grayscale images to a pixel range of `0–1`. Addition of a channel dimension to produce the shape `(number of samples, 28, 28, 1)`.

| Setting | Value |
|---|---|
| CNN architecture | Two convolution and pooling blocks → Flatten → Dense 128 → Softmax 10 |
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Validation split | 10% of the training data |
| Batch size | 128 |
| Maximum epochs | 20 |
| Model selection | Lowest validation loss, `val_loss` |

### train.py

```python
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers


MODEL_PATH = Path(__file__).resolve().parent / "models" / "mnist.keras"


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    keras.utils.set_random_seed(42)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    x_test = x_test[..., None].astype("float32") / 255.0

    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=128,
        epochs=20,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        ],
        verbose=2,
    )

    best_model = keras.models.load_model(MODEL_PATH)
    loss, accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- Full model storage through `ModelCheckpoint` on improvement in validation loss
- Training termination through `EarlyStopping` after three epochs without improvement in validation loss
- Test evaluation and Flask inference using the same saved checkpoint
- Network access for the MNIST download on the first run

| Storage target | Saving method | Loading method |
|---|---|---|
| Full model | `model.save("mnist.keras")` or a full-model checkpoint | `keras.models.load_model("mnist.keras")` |
| Weights only | `model.save_weights("model.weights.h5")` | Reconstruction of the same model architecture followed by `load_weights()` |

---

## 4. Flask and Model Integration

### 4.1. Image Preprocessing and Result Conversion

Input image with a single dark digit on a white background. Brightness inversion to match MNIST and black padding during resizing.

```text
Uploaded image → Grayscale conversion → Inversion → 28 × 28 resizing
    → 0–1 normalization → (1, 28, 28, 1) array → Model input
```

Creation of `utils.py`.

```python
import numpy as np
from PIL import Image, ImageOps


def prepare_input(stream):
    with Image.open(stream) as source:
        grayscale = ImageOps.exif_transpose(source).convert("L")
        inverted = ImageOps.invert(grayscale)
        resized = ImageOps.pad(
            inverted,
            (28, 28),
            method=Image.Resampling.LANCZOS,
            color=0,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    return pixels[None, ..., None]


def describe_prediction(scores):
    digit = int(np.argmax(scores))
    return {"digit": digit, "prob": float(scores[digit])}
```

Storage of the highest-scoring digit in `digit` and its softmax score in `prob`. Possible variation in prediction results for images with digit placement or margins substantially different from the training images.

### 4.2. Model Loading and Prediction API

Creation of `app.py`. Model loading once at application startup, followed by image preprocessing and inference for each prediction request.

```python
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from tensorflow import keras

from utils import describe_prediction, prepare_input


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

model_path = Path(__file__).resolve().parent / "models" / "mnist.keras"
model = keras.models.load_model(model_path, compile=False)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="Select an image file."), 400

    try:
        image = prepare_input(upload.stream)
    except (OSError, ValueError):
        return jsonify(error="Unable to read the image."), 400

    scores = model.predict(image, verbose=0)[0]
    return jsonify(describe_prediction(scores))


@app.errorhandler(413)
def request_too_large(error):
    return jsonify(error="The upload request must be 5 MiB or smaller."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

| Request | Processing |
|---|---|
| `GET /` | Rendering of the image upload page |
| `POST /predict` | Image retrieval from multipart field `file` and return of the prediction |
| Missing or unreadable image | JSON error message and HTTP 400 |
| Request body exceeding 5 MiB | JSON error message and HTTP 413 |

### 4.3. Upload Page and Static Files

Creation of `templates/index.html`. Matching of the file input's `name="file"` with the key read through `request.files` in Flask.

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MNIST Digit Recognition</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <main>
        <h1>MNIST Digit Recognition</h1>
        <p>Select an image of a single dark digit on a white background.</p>

        <form id="upload-form" action="/predict" method="post"
              enctype="multipart/form-data">
            <label for="digit-image">Digit image</label>
            <input id="digit-image" type="file" name="file" accept="image/*" required>
            <button type="submit">Predict</button>
        </form>

        <p id="result" role="status" aria-live="polite"></p>
    </main>

    <script>
        const form = document.querySelector('#upload-form');
        const result = document.querySelector('#result');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            result.textContent = 'Predicting…';

            try {
                const response = await fetch(form.action, {
                    method: 'POST',
                    body: new FormData(form),
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Prediction request failed.');
                }

                const score = (data.prob * 100).toFixed(2);
                result.textContent = `Predicted digit: ${data.digit} / Model score: ${score}%`;
            } catch (error) {
                result.textContent = error.message;
            }
        });
    </script>
</body>
</html>
```

Creation of `static/style.css`.

```css
body {
    margin: 0;
    padding: 24px;
    background: #f1f5f9;
    color: #1e293b;
    font-family: system-ui, sans-serif;
}

main {
    max-width: 640px;
    margin: 48px auto;
    padding: 24px;
    background: white;
    border-radius: 12px;
}

form {
    display: grid;
    gap: 12px;
}

button {
    padding: 12px;
    border: 0;
    border-radius: 6px;
    background: #1d4ed8;
    color: white;
    cursor: pointer;
}
```

Image submission through `FormData` and display of prediction results or errors through JavaScript.

---

## 5. Execution and Verification

### 5.1. Model Training and Server Startup

Execution in order inside the container.

```bash
python train.py
python app.py
```

Verification of `models/mnist.keras` generation and test accuracy output after training. Browser access at `http://localhost:5000` and image upload.

### 5.2. API Response Verification

Execution from PowerShell with replacement of `digit.png` by the actual image path.

```powershell
curl.exe -F "file=@digit.png" http://localhost:5000/predict
```

Example JSON response format.

```json
{"digit": 7, "prob": 0.98}
```

Actual prediction and score dependent on the uploaded image and saved model.

### 5.3. Verification Checklist

- Model file generation and test accuracy output after training
- Display of the predicted digit and score after image upload
- HTTP 400 response for a missing or unreadable image

---

## 6. Principal Component Analysis (PCA)

### 6.1. PCA Overview

A linear dimensionality reduction method for representing the original features with fewer axes, starting with the directions that account for the most variance.

Standardization of Iris sepal and petal measurements, followed by projection onto two principal components for visualization of species distributions.

| Concept | Description |
|---|---|
| Principal component | A new axis formed by a linear combination of the original features |
| Standardization | Adjustment of each feature to a mean of 0 and a standard deviation of 1 |
| Explained variance ratio | Proportion of total variance explained by each principal component |

### 6.2. Iris Data Visualization

Creation of `pca_iris.py`.

```python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


iris = load_iris()
scaled = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2)
coordinates = pca.fit_transform(scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print(f"Total retained variance: {pca.explained_variance_ratio_.sum():.2%}")

figure, axis = plt.subplots(figsize=(7, 5))

for label, name in enumerate(iris.target_names):
    selected = iris.target == label
    axis.scatter(
        coordinates[selected, 0],
        coordinates[selected, 1],
        label=name,
        alpha=0.75,
    )

axis.set(xlabel="PC1", ylabel="PC2", title="Iris PCA")
axis.legend()
figure.tight_layout()
figure.savefig("iris_pca.png", dpi=150)
plt.close(figure)
```

Execution inside the container.

```bash
python pca_iris.py
```

Retention of approximately **95.8%** of the variance in the standardized Iris data across two principal components. Inspection of species distributions and areas of overlap in `iris_pca.png`.

For PCA as a predictive model preprocessing step, fitting of the scaler and PCA on the training split only. Application of the fitted transformations to validation and test data.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 3주차 — TensorFlow와 Flask를 활용한 MNIST 손글씨 분류 및 PCA

Docker 환경에서의 CNN 학습, Flask 웹 애플리케이션을 통한 손글씨 숫자 예측, PCA를 활용한 Iris 데이터 차원 축소 실습.

## 1. 프로젝트 개요

1주차의 Docker 개발 환경과 2주차의 Flask 모델 연동 방식 활용. 수치 입력 기반 품종 분류에서 이미지 업로드 기반 숫자 분류로 예측 흐름 확장.

```text
MNIST → CNN 학습·검증 → Keras 모델 저장

브라우저 → 이미지 업로드 → Flask → 전처리 → 모델 추론 → JSON 응답
```

| 구분 | 구현 내용 |
|---|---|
| 모델 학습 | TensorFlow/Keras CNN으로 숫자 0~9 분류 |
| 모델 저장 | `models/mnist.keras`에 전체 모델 저장 |
| 이미지 전처리 | 흑백 변환, 명암 반전, 검은색 여백 추가 및 정규화 |
| API | `POST /predict`를 통한 이미지 수신 및 JSON 응답 |
| 웹페이지 | HTML, CSS, JavaScript를 활용한 이미지 업로드 및 결과 표시 |
| PCA 실습 | 표준화한 Iris 데이터를 두 주성분에 투영 |

---

## 2. Docker 개발 환경 구성

Windows 명령어는 **PowerShell**, Python 명령어는 **컨테이너 내부 Bash 기준**.

### 2.1. 컨테이너 생성

PowerShell에서 프로젝트 폴더로 이동 후 GPU 또는 CPU 명령 중 하나 실행. GPU 환경에서는 NVIDIA GPU와 Docker의 GPU 접근 설정 필요.

**GPU 환경**

```powershell
docker run -dit --gpus all --name mnist-api -p 127.0.0.1:5000:5000 -v "${PWD}:/workspace" -w /workspace tensorflow/tensorflow:2.16.2-gpu bash
```

**CPU 환경**

```powershell
docker run -dit --name mnist-api -p 127.0.0.1:5000:5000 -v "${PWD}:/workspace" -w /workspace tensorflow/tensorflow:2.16.2 bash
```

| 옵션 | 의미 |
|---|---|
| `--gpus all` | 컨테이너의 GPU 접근 허용 |
| `-p 127.0.0.1:5000:5000` | 로컬 PC의 5000 포트와 컨테이너의 5000 포트 연결 |
| `-v "${PWD}:/workspace"` | 현재 프로젝트 폴더와 컨테이너의 `/workspace` 공유 |
| `-w /workspace` | 컨테이너 작업 디렉터리 지정 |

실행 중인 컨테이너에 접속.

```powershell
docker exec -it mnist-api bash
```

### 2.2. 파일 구성 및 의존성 설치

아래 구조에 맞춰 실습 파일 작성. `models/mnist.keras`와 `iris_pca.png`는 학습 및 PCA 스크립트 실행 시 각각 생성.

```text
project/
├── requirements.txt
├── train.py
├── utils.py
├── app.py
├── pca_iris.py
├── models/
│   └── mnist.keras
├── templates/
│   └── index.html
└── static/
    └── style.css
```

`requirements.txt`에 추가 패키지 목록 작성. TensorFlow와 NumPy는 위 컨테이너 이미지에 포함.

```text
Flask
Pillow
scikit-learn
matplotlib
```

컨테이너 내부에서 패키지 설치 및 TensorFlow 실행 환경 확인.

```bash
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## 3. MNIST 손글씨 분류 모델 학습

`28 × 28` 흑백 이미지의 픽셀값을 `0~1` 범위로 정규화. 채널 차원을 추가해 `(샘플 수, 28, 28, 1)` 형태로 변환.

| 구분 | 설정 |
|---|---|
| CNN 구조 | 합성곱·풀링 2단 → Flatten → Dense 128 → Softmax 10 |
| 옵티마이저 | Adam |
| 손실 함수 | Sparse Categorical Cross-Entropy |
| 검증 데이터 | 학습용 데이터의 10% |
| 배치 크기 | 128 |
| 최대 학습 횟수 | 20 에포크 |
| 모델 선택 기준 | 가장 낮은 검증 손실 `val_loss` |

### train.py

```python
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers


MODEL_PATH = Path(__file__).resolve().parent / "models" / "mnist.keras"


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    keras.utils.set_random_seed(42)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    x_test = x_test[..., None].astype("float32") / 255.0

    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=128,
        epochs=20,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        ],
        verbose=2,
    )

    best_model = keras.models.load_model(MODEL_PATH)
    loss, accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- `ModelCheckpoint`를 통한 검증 손실 개선 시 전체 모델 저장
- 검증 손실이 3에포크 동안 개선되지 않을 경우 `EarlyStopping`을 통한 학습 종료
- 저장된 체크포인트로 테스트 데이터 평가 및 Flask에서 같은 모델 파일 재사용
- 첫 학습 시 MNIST 데이터셋 다운로드를 위한 네트워크 연결 필요

| 저장 대상 | 저장 방법 | 불러오는 방법 |
|---|---|---|
| 전체 모델 | `model.save("mnist.keras")` 또는 전체 모델 체크포인트 | `keras.models.load_model("mnist.keras")` |
| 가중치만 | `model.save_weights("model.weights.h5")` | 동일한 모델 구조 생성 후 `load_weights()` 호출 |

---

## 4. Flask와 모델 연동

### 4.1. 이미지 전처리 및 결과 변환

흰 배경에 검은 숫자 하나가 그려진 이미지 사용. MNIST의 명암 방향에 맞춘 이미지 반전 및 크기 조정 시 검은색 여백 추가.

```text
업로드 이미지 → 흑백 변환 → 명암 반전 → 28 × 28 크기 조정
    → 0~1 정규화 → (1, 28, 28, 1) 배열 → 모델 입력
```

`utils.py` 작성.

```python
import numpy as np
from PIL import Image, ImageOps


def prepare_input(stream):
    with Image.open(stream) as source:
        grayscale = ImageOps.exif_transpose(source).convert("L")
        inverted = ImageOps.invert(grayscale)
        resized = ImageOps.pad(
            inverted,
            (28, 28),
            method=Image.Resampling.LANCZOS,
            color=0,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    return pixels[None, ..., None]


def describe_prediction(scores):
    digit = int(np.argmax(scores))
    return {"digit": digit, "prob": float(scores[digit])}
```

`digit`에는 가장 높은 점수를 받은 숫자, `prob`에는 해당 숫자의 Softmax 점수 저장. 숫자 위치나 여백이 학습 이미지와 크게 다른 경우 예측 결과 변동 가능.

### 4.2. 모델 불러오기 및 예측 API

`app.py` 작성. 서버 시작 시 모델을 한 번 불러오고, 요청마다 이미지 전처리와 추론 수행.

```python
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from tensorflow import keras

from utils import describe_prediction, prepare_input


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

model_path = Path(__file__).resolve().parent / "models" / "mnist.keras"
model = keras.models.load_model(model_path, compile=False)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="이미지 파일을 선택해 주세요."), 400

    try:
        image = prepare_input(upload.stream)
    except (OSError, ValueError):
        return jsonify(error="읽을 수 없는 이미지입니다."), 400

    scores = model.predict(image, verbose=0)[0]
    return jsonify(describe_prediction(scores))


@app.errorhandler(413)
def request_too_large(error):
    return jsonify(error="업로드 요청은 5 MiB 이하여야 합니다."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

| 요청 | 처리 내용 |
|---|---|
| `GET /` | 이미지 업로드 페이지 렌더링 |
| `POST /predict` | `multipart/form-data`의 `file` 필드를 읽고 예측 결과 반환 |
| 파일 누락 또는 이미지 읽기 실패 | JSON 오류 메시지 및 HTTP 400 반환 |
| 요청 본문 5 MiB 초과 | JSON 오류 메시지 및 HTTP 413 반환 |

### 4.3. 업로드 페이지 및 정적 파일

`templates/index.html` 작성. 파일 입력의 `name="file"`과 Flask의 `request.files` 조회 키 일치 필요.

```html
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MNIST 손글씨 분류</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <main>
        <h1>MNIST 손글씨 분류</h1>
        <p>흰 배경에 검은 숫자 하나가 그려진 이미지를 선택하세요.</p>

        <form id="upload-form" action="/predict" method="post"
              enctype="multipart/form-data">
            <label for="digit-image">숫자 이미지</label>
            <input id="digit-image" type="file" name="file" accept="image/*" required>
            <button type="submit">예측하기</button>
        </form>

        <p id="result" role="status" aria-live="polite"></p>
    </main>

    <script>
        const form = document.querySelector('#upload-form');
        const result = document.querySelector('#result');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            result.textContent = '예측 중…';

            try {
                const response = await fetch(form.action, {
                    method: 'POST',
                    body: new FormData(form),
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || '예측 요청에 실패했습니다.');
                }

                const score = (data.prob * 100).toFixed(2);
                result.textContent = `예측 숫자: ${data.digit} / 모델 점수: ${score}%`;
            } catch (error) {
                result.textContent = error.message;
            }
        });
    </script>
</body>
</html>
```

`static/style.css` 작성.

```css
body {
    margin: 0;
    padding: 24px;
    background: #f1f5f9;
    color: #1e293b;
    font-family: system-ui, sans-serif;
}

main {
    max-width: 640px;
    margin: 48px auto;
    padding: 24px;
    background: white;
    border-radius: 12px;
}

form {
    display: grid;
    gap: 12px;
}

button {
    padding: 12px;
    border: 0;
    border-radius: 6px;
    background: #1d4ed8;
    color: white;
    cursor: pointer;
}
```

`FormData`를 이용한 이미지 전송 및 JavaScript를 통한 예측 결과·오류 메시지 표시.

---

## 5. 실행 및 동작 확인

### 5.1. 모델 학습 및 서버 시작

컨테이너 내부에서 순서대로 실행.

```bash
python train.py
python app.py
```

`train.py` 실행 후 `models/mnist.keras` 생성 및 테스트 정확도 출력 확인. 브라우저에서 `http://localhost:5000` 접속 후 이미지 업로드.

### 5.2. API 응답 확인

PowerShell에서 `digit.png`를 실제 이미지 경로로 바꿔 실행.

```powershell
curl.exe -F "file=@digit.png" http://localhost:5000/predict
```

JSON 응답 형식 예시.

```json
{"digit": 7, "prob": 0.98}
```

업로드 이미지와 저장된 모델에 따라 실제 예측 숫자 및 점수 결정.

### 5.3. 확인 항목

- 학습 후 모델 파일 생성 및 테스트 정확도 출력 여부 확인
- 이미지 업로드 후 웹페이지의 예측 숫자 및 점수 표시 확인
- 파일 누락 또는 이미지 읽기 실패 시 HTTP 400 응답 확인

---

## 6. 주성분 분석(PCA)

### 6.1. PCA 개요

데이터의 분산을 크게 설명하는 방향부터 찾아, 원래 특성을 더 적은 수의 축으로 표현하는 선형 차원 축소 기법.

Iris의 꽃받침·꽃잎 길이와 너비를 표준화한 뒤, 두 주성분에 투영해 품종별 분포 시각화.

| 개념 | 설명 |
|---|---|
| 주성분 | 원래 변수의 선형 결합으로 구성한 새 축 |
| 표준화 | 변수별 평균을 0, 표준편차를 1로 조정 |
| 설명 분산 비율 | 각 주성분이 전체 분산에서 설명하는 비율 |

### 6.2. Iris 데이터 시각화

`pca_iris.py` 작성.

```python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


iris = load_iris()
scaled = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2)
coordinates = pca.fit_transform(scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print(f"Total retained variance: {pca.explained_variance_ratio_.sum():.2%}")

figure, axis = plt.subplots(figsize=(7, 5))

for label, name in enumerate(iris.target_names):
    selected = iris.target == label
    axis.scatter(
        coordinates[selected, 0],
        coordinates[selected, 1],
        label=name,
        alpha=0.75,
    )

axis.set(xlabel="PC1", ylabel="PC2", title="Iris PCA")
axis.legend()
figure.tight_layout()
figure.savefig("iris_pca.png", dpi=150)
plt.close(figure)
```

컨테이너 내부에서 실행.

```bash
python pca_iris.py
```

표준화한 Iris 데이터의 분산 중 약 **95.8%**를 두 주성분으로 설명. `iris_pca.png`를 통한 품종별 분포 및 겹치는 영역 확인.

예측 모델의 전처리에 PCA를 사용할 경우, 데이터 분할 후 학습 데이터에만 표준화와 PCA의 `fit` 적용. 검증·테스트 데이터에는 학습 데이터에서 구한 변환 사용.

</details>
