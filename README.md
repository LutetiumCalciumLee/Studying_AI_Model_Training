<details>
<summary>ENG (English Version)</summary>

# Week 02 — Python API Server Development with Flask and Docker

An introduction to Flask routing, request handling, and template rendering, followed by an Iris classification web application in a Docker environment.

## 1. Flask Overview

Flask: a lightweight Python web framework for building web applications and API servers.

A minimal core with support for extensions such as database integration and authentication.

Applications in AI services: receiving input data, running a trained model, and returning prediction results.

### Core Features

| Feature | Description |
|---|---|
| Routing | Mapping of URL paths to Python functions |
| Request handling | Access to query parameters, form fields, and JSON request bodies |
| JSON responses | Structured data exchange with API clients |
| Template rendering | Dynamic HTML generation through Jinja2 |
| Static file serving | Delivery of CSS, JavaScript, and image files |

### Request–Response Flow

```text
Client
    │ HTTP request
    ▼
Flask route
    │ Input processing and application logic
    ▼
Response
    ├── HTML page
    └── JSON data
```

---

## 2. Docker Development Environment

Continuation of the Docker Desktop and VS Code environment prepared in Week 01.

**PowerShell** as the shell for Windows commands; **Bash inside the container** for Python commands.

### 2.1. Container Creation

Execution from the project folder in PowerShell:

```powershell
docker run -dit --name flaskserver -p 127.0.0.1:5000:5000 -v "${PWD}:/workspace" -w /workspace python:3.12-slim bash
```

| Option | Meaning |
|---|---|
| `-dit` | Background execution with standard input and terminal support |
| `--name flaskserver` | Container name assignment |
| `-p 127.0.0.1:5000:5000` | Mapping of local PC port 5000 to container port 5000 |
| `-v "${PWD}:/workspace"` | Sharing of the current project folder with the container |
| `-w /workspace` | Container working directory assignment |
| `python:3.12-slim` | Python base image |
| `bash` | Shell selection |

Container shell access:

```powershell
docker exec -it flaskserver bash
```

### 2.2. Dependency Installation

Creation of `requirements.txt` in the project folder:

```text
flask
numpy
scikit-learn
```

Installation inside the container:

```bash
python -m pip install -r requirements.txt
```

Package version recording after verification for more consistent environment reproduction.

### 2.3. VS Code Integration

1. Installation of the **Dev Containers** extension
2. Selection of `Dev Containers: Attach to Running Container...`
3. Connection to `flaskserver`
4. Opening of `/workspace`
5. Installation of the Python extension in the connected container environment

Source file sharing between `/workspace` and the connected PC folder.

---

## 3. Basic Flask Application

Creation of `app.py`:

```python
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

| Code | Role |
|---|---|
| `Flask(__name__)` | Application creation |
| `@app.route("/")` | URL-to-function mapping |
| `return "Hello, World!"` | Response body |
| `host="0.0.0.0"` | Listening on all container network interfaces |
| `port=5000` | Server port assignment |

Execution inside the container:

```bash
python app.py
```

Browser verification at `http://localhost:5000`.

Server termination through `Ctrl+C`.

> `0.0.0.0` is a listening address.
> Browser access through `localhost` on the host computer.
> Flask's built-in server is intended for development and testing.

---

## 4. Routing and Request Handling

Independent examples for addition to the basic application above.

### 4.1. Multiple URLs and Dynamic Paths

```python
from flask import jsonify


@app.route("/index")
@app.route("/home")
def main():
    return "Welcome to the Flask server!"


@app.route("/user/<username>")
def user_profile(username):
    return jsonify({"username": username})


@app.route("/post/<int:post_id>")
def post_detail(post_id):
    return jsonify({"post_id": post_id})
```

- Multiple URL mappings for a single function
- `<username>`: String value extracted from the path
- `<int:post_id>`: Integer path converter
- Routes accepting GET requests by default

### 4.2. Query Parameters

Example request: `/search?q=flask`

```python
from flask import request


@app.route("/search")
def search():
    keyword = request.args.get("q", "")
    return jsonify({"keyword": keyword})
```

### 4.3. POST Requests and Form Data

```python
@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form.get("user_input", "")
    return jsonify({"received": user_input})
```

Explicit HTTP method configuration through `methods=["POST"]`.

HTML form field names used as keys in `request.form`.

### 4.4. JSON Responses

```python
@app.route("/api/data")
def get_data():
    return jsonify({
        "message": "Flask API is running",
        "number": 42,
    })
```

Input access methods by request format:

| Input format | Flask interface |
|---|---|
| URL query parameters | `request.args` |
| HTML form data | `request.form` |
| JSON request body | `request.get_json()` |

Separate handling of request and response formats.

For example, form input followed by a JSON response, or form input followed by an HTML response.

---

## 5. Templates and Static Files

Dynamic HTML generation through the Jinja2 template engine.

### 5.1. Template Rendering

Creation of `templates/hello.html`:

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Greeting</title>
</head>
<body>
    <h1>Hello, {{ name }}!</h1>
</body>
</html>
```

Route addition to `app.py`:

```python
from flask import render_template


@app.route("/hello/<username>")
def hello_user(username):
    return render_template("hello.html", name=username)
```

- `render_template()`: HTML generation from a template
- `name=username`: Variable transfer from Python to the template
- `{{ name }}`: Variable output in Jinja2

### 5.2. Static Files

Storage of CSS, JavaScript, and images under `static/`.

Example stylesheet reference inside a template:

```html
<link rel="stylesheet"
      href="{{ url_for('static', filename='style.css') }}">
```

File location in this example: `static/style.css`.

---

## 6. Iris Classification Model Training

Prediction of Iris flower species from four measurements.

### Input and Output

| Category | Description |
|---|---|
| Sepal length | Sepal length in centimeters |
| Sepal width | Sepal width in centimeters |
| Petal length | Petal length in centimeters |
| Petal width | Petal width in centimeters |
| Prediction | `setosa`, `versicolor`, or `virginica` |

### Project Structure

```text
project/
├── requirements.txt
├── train_model.py
├── app.py
├── model/
│   └── iris_model.pkl
└── templates/
    └── index.html
```

Generation of `model/iris_model.pkl` through the training script.

### train_model.py

```python
from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "model"
model_dir.mkdir(parents=True, exist_ok=True)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target,
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

with (model_dir / "iris_model.pkl").open("wb") as file:
    pickle.dump(model, file)
```

- `load_iris()`: Iris dataset loading
- `train_test_split()`: Training/test data separation
- `random_state=42`: Fixed random seed for data splitting
- `stratify=iris.target`: Class proportion preservation during splitting
- `KNeighborsClassifier(n_neighbors=3)`: KNN classifier with three neighbors
- `model.score()`: Classification accuracy on the test set
- `pickle.dump()`: Trained model serialization

---

## 7. Flask and Model Integration

Replacement of the basic `app.py` with the following application.

### 7.1. Model Loading and Prediction

```python
from pathlib import Path
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model_path = Path(__file__).resolve().parent / "model" / "iris_model.pkl"

with model_path.open("rb") as file:
    model = pickle.load(file)

species = ["setosa", "versicolor", "virginica"]

fields = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    status = 200

    if request.method == "POST":
        try:
            values = [float(request.form[field]) for field in fields]
            features = np.array([values])

            if not np.isfinite(features).all() or (features <= 0).any():
                raise ValueError("Invalid measurements")

            label = int(model.predict(features)[0])
            prediction = species[label]

        except (KeyError, ValueError):
            error = "Enter a positive number for each measurement."
            status = 400

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
    ), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

Model loading once during application startup.

Input conversion to a numeric array with shape `(1, 4)`, preserving the feature order used during training.

| Request | Processing |
|---|---|
| GET `/` | Input form display |
| Valid POST `/` | Input conversion, prediction, and result display |
| Invalid POST `/` | Error message with HTTP status 400 |

### 7.2. Input Form

Creation of `templates/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Iris Classifier</title>
</head>
<body>
    <h1>Iris Species Prediction</h1>
    <p>Enter measurements in centimeters.</p>

    <form method="post" action="/">
        {% for field, label in [
            ('sepal_length', 'Sepal length'),
            ('sepal_width', 'Sepal width'),
            ('petal_length', 'Petal length'),
            ('petal_width', 'Petal width')
        ] %}
        <p>
            <label>
                {{ label }}
                <input type="number" name="{{ field }}"
                       step="any" min="0.01" required>
            </label>
        </p>
        {% endfor %}

        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
    <p>Prediction: {{ prediction }}</p>
    {% endif %}

    {% if error %}
    <p role="alert">{{ error }}</p>
    {% endif %}
</body>
</html>
```

Matching of HTML `name` attributes with the keys read through `request.form`.

Conditional result and error display through Jinja2.

---

## 8. Execution and Verification

### 8.1. Model Training and Server Startup

Execution inside the container:

```bash
python train_model.py
python app.py
```

Training before server startup to generate the required model file.

Browser access at `http://localhost:5000`.

### 8.2. Prediction Flow

```text
Browser form
    │ POST request with four measurements
    ▼
Flask
    │ Input validation and numeric conversion
    ▼
KNN model
    │ Species prediction
    ▼
Jinja2 template
    │ HTML rendering
    ▼
Browser result
```

This example provides predictions through an HTML form.

JSON prediction services can be implemented by adding an endpoint that receives JSON input and returns the prediction through `jsonify()`.

### 8.3. Restarting the Development Environment

Execution from PowerShell when the container is stopped:

```powershell
docker start flaskserver
docker exec -it flaskserver bash
```

Server startup inside the container:

```bash
python app.py
```

No retraining required when reusing the saved model in the same compatible environment.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 2주차 — Flask와 Docker를 활용한 Python API 서버 개발

Flask의 라우팅·요청 처리·템플릿 렌더링 학습 및 Docker 환경에서의 Iris 붓꽃 분류 웹 애플리케이션 구성.

## 1. Flask 개요

Flask는 웹 애플리케이션과 API 서버를 개발하는 경량 Python 웹 프레임워크.

핵심 기능을 작게 유지하고, 데이터베이스 연동이나 인증 등은 확장 기능으로 추가하는 구조.

AI 서비스에서는 입력 데이터 수신, 학습된 모델의 예측 실행, 결과 반환 등에 활용 가능.

### 주요 기능

| 기능 | 설명 |
|---|---|
| 라우팅 | URL 경로와 Python 함수 연결 |
| 요청 처리 | 쿼리 파라미터, 폼 데이터, JSON 요청 본문 접근 |
| JSON 응답 | API 클라이언트와 구조화된 데이터 교환 |
| 템플릿 렌더링 | Jinja2를 활용한 동적 HTML 생성 |
| 정적 파일 제공 | CSS, JavaScript, 이미지 파일 전달 |

### 요청과 응답의 흐름

```text
클라이언트
    │ HTTP 요청
    ▼
Flask 라우트
    │ 입력 처리 및 애플리케이션 로직 실행
    ▼
응답
    ├── HTML 페이지
    └── JSON 데이터
```

---

## 2. Docker 개발 환경 구성

1주차에서 준비한 Docker Desktop 및 VS Code 환경을 기반으로 실습 진행.

Windows 명령어는 **PowerShell**, Python 명령어는 **컨테이너 내부 Bash 기준**.

### 2.1. 컨테이너 생성

PowerShell에서 프로젝트 폴더로 이동 후 실행.

```powershell
docker run -dit --name flaskserver -p 127.0.0.1:5000:5000 -v "${PWD}:/workspace" -w /workspace python:3.12-slim bash
```

| 옵션 | 의미 |
|---|---|
| `-dit` | 표준 입력과 터미널을 유지하며 백그라운드 실행 |
| `--name flaskserver` | 컨테이너 이름 지정 |
| `-p 127.0.0.1:5000:5000` | 로컬 PC의 5000 포트와 컨테이너의 5000 포트 연결 |
| `-v "${PWD}:/workspace"` | 현재 프로젝트 폴더와 컨테이너 내부 경로 공유 |
| `-w /workspace` | 컨테이너 작업 디렉터리 지정 |
| `python:3.12-slim` | Python 기반 이미지 지정 |
| `bash` | 실행할 셸 지정 |

컨테이너 내부 접속.

```powershell
docker exec -it flaskserver bash
```

### 2.2. 의존성 설치

프로젝트 폴더에 `requirements.txt` 작성.

```text
flask
numpy
scikit-learn
```

컨테이너 내부에서 패키지 설치.

```bash
python -m pip install -r requirements.txt
```

환경 검증 후 패키지 버전을 기록하여 실행 환경의 재현성 향상 가능.

### 2.3. VS Code 연동

1. **Dev Containers** 확장 설치
2. `Dev Containers: Attach to Running Container...` 선택
3. `flaskserver` 연결
4. `/workspace` 폴더 열기
5. 연결된 컨테이너 환경에 Python 확장 설치

`/workspace`에서 작성한 소스 파일은 연결된 PC 폴더에도 반영.

---

## 3. Flask 기본 애플리케이션

`app.py` 작성.

```python
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

| 코드 | 역할 |
|---|---|
| `Flask(__name__)` | 애플리케이션 생성 |
| `@app.route("/")` | URL과 함수 연결 |
| `return "Hello, World!"` | 응답 본문 반환 |
| `host="0.0.0.0"` | 컨테이너의 모든 네트워크 인터페이스에서 요청 수신 |
| `port=5000` | 서버 포트 지정 |

컨테이너 내부에서 실행.

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속 후 동작 확인.

`Ctrl+C` 입력으로 서버 종료.

> `0.0.0.0`은 서버의 수신 주소.
> 컴퓨터의 브라우저에서는 `localhost`를 통해 접속.
> Flask 내장 서버는 개발·테스트 용도로 사용.

---

## 4. 라우팅 및 요청 처리

앞의 기본 애플리케이션에 추가하여 확인할 수 있는 개별 예제.

### 4.1. 여러 URL 연결 및 동적 경로

```python
from flask import jsonify


@app.route("/index")
@app.route("/home")
def main():
    return "Welcome to the Flask server!"


@app.route("/user/<username>")
def user_profile(username):
    return jsonify({"username": username})


@app.route("/post/<int:post_id>")
def post_detail(post_id):
    return jsonify({"post_id": post_id})
```

- 하나의 함수에 여러 URL 연결 가능
- `<username>`: 경로에서 문자열 추출
- `<int:post_id>`: 정수형 경로 변환기
- 기본적으로 GET 요청 처리

### 4.2. 쿼리 파라미터

요청 예시: `/search?q=flask`

```python
from flask import request


@app.route("/search")
def search():
    keyword = request.args.get("q", "")
    return jsonify({"keyword": keyword})
```

### 4.3. POST 요청 및 폼 데이터

```python
@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form.get("user_input", "")
    return jsonify({"received": user_input})
```

`methods=["POST"]`를 통한 HTTP 메서드 지정.

HTML 폼 입력 항목의 이름을 `request.form`의 키로 사용.

### 4.4. JSON 응답

```python
@app.route("/api/data")
def get_data():
    return jsonify({
        "message": "Flask API is running",
        "number": 42,
    })
```

요청 형식에 따른 데이터 접근 방법.

| 입력 형식 | Flask 인터페이스 |
|---|---|
| URL 쿼리 파라미터 | `request.args` |
| HTML 폼 데이터 | `request.form` |
| JSON 요청 본문 | `request.get_json()` |

요청 데이터 형식과 응답 형식은 별도로 처리.

예를 들어 폼 데이터를 입력받은 뒤 JSON 또는 HTML로 응답 가능.

---

## 5. 템플릿 및 정적 파일

Jinja2 템플릿 엔진을 활용한 동적 HTML 생성.

### 5.1. 템플릿 렌더링

`templates/hello.html` 작성.

```html
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>인사 페이지</title>
</head>
<body>
    <h1>안녕하세요, {{ name }}님!</h1>
</body>
</html>
```

`app.py`에 라우트 추가.

```python
from flask import render_template


@app.route("/hello/<username>")
def hello_user(username):
    return render_template("hello.html", name=username)
```

- `render_template()`: 템플릿을 사용한 HTML 생성
- `name=username`: Python에서 템플릿으로 변수 전달
- `{{ name }}`: Jinja2의 변수 출력 구문

### 5.2. 정적 파일

CSS, JavaScript, 이미지 등을 `static/` 폴더에 저장.

템플릿에서 CSS 파일을 연결하는 예시.

```html
<link rel="stylesheet"
      href="{{ url_for('static', filename='style.css') }}">
```

위 예시의 파일 경로는 `static/style.css`.

---

## 6. Iris 붓꽃 분류 모델 학습

네 가지 측정값을 사용한 Iris 붓꽃 품종 예측.

### 입력 및 출력

| 구분 | 설명 |
|---|---|
| Sepal length | 꽃받침 길이, 단위 cm |
| Sepal width | 꽃받침 너비, 단위 cm |
| Petal length | 꽃잎 길이, 단위 cm |
| Petal width | 꽃잎 너비, 단위 cm |
| 예측 결과 | `setosa`, `versicolor`, `virginica` 중 하나 |

### 파일 구성

```text
project/
├── requirements.txt
├── train_model.py
├── app.py
├── model/
│   └── iris_model.pkl
└── templates/
    └── index.html
```

`model/iris_model.pkl`은 학습 스크립트 실행 시 생성.

### train_model.py

```python
from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "model"
model_dir.mkdir(parents=True, exist_ok=True)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target,
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

with (model_dir / "iris_model.pkl").open("wb") as file:
    pickle.dump(model, file)
```

- `load_iris()`: Iris 데이터셋 불러오기
- `train_test_split()`: 학습·테스트 데이터 분리
- `random_state=42`: 데이터 분할의 난수 시드 고정
- `stratify=iris.target`: 분할 시 클래스 비율 유지
- `KNeighborsClassifier(n_neighbors=3)`: 이웃 수가 3인 KNN 분류기
- `model.score()`: 테스트 데이터의 분류 정확도 확인
- `pickle.dump()`: 학습된 모델 직렬화 및 저장

---

## 7. Flask와 모델 연동

앞에서 작성한 기본 `app.py`를 다음 애플리케이션으로 교체.

### 7.1. 모델 불러오기 및 예측

```python
from pathlib import Path
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model_path = Path(__file__).resolve().parent / "model" / "iris_model.pkl"

with model_path.open("rb") as file:
    model = pickle.load(file)

species = ["setosa", "versicolor", "virginica"]

fields = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    status = 200

    if request.method == "POST":
        try:
            values = [float(request.form[field]) for field in fields]
            features = np.array([values])

            if not np.isfinite(features).all() or (features <= 0).any():
                raise ValueError("Invalid measurements")

            label = int(model.predict(features)[0])
            prediction = species[label]

        except (KeyError, ValueError):
            error = "모든 측정값에 양수를 입력해 주세요."
            status = 400

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
    ), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

애플리케이션 시작 시 모델을 한 번 불러오는 구조.

학습에 사용한 특성 순서를 유지하여 입력값을 `(1, 4)` 형태의 숫자 배열로 변환.

| 요청 | 처리 내용 |
|---|---|
| GET `/` | 입력 폼 표시 |
| 정상 POST `/` | 입력값 변환, 모델 예측, 결과 표시 |
| 잘못된 POST `/` | 오류 메시지 및 HTTP 상태 코드 400 반환 |

### 7.2. 입력 폼 작성

`templates/index.html` 작성.

```html
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>Iris 분류기</title>
</head>
<body>
    <h1>Iris 붓꽃 품종 예측</h1>
    <p>측정값을 cm 단위로 입력해 주세요.</p>

    <form method="post" action="/">
        {% for field, label in [
            ('sepal_length', '꽃받침 길이'),
            ('sepal_width', '꽃받침 너비'),
            ('petal_length', '꽃잎 길이'),
            ('petal_width', '꽃잎 너비')
        ] %}
        <p>
            <label>
                {{ label }}
                <input type="number" name="{{ field }}"
                       step="any" min="0.01" required>
            </label>
        </p>
        {% endfor %}

        <button type="submit">예측하기</button>
    </form>

    {% if prediction %}
    <p>예측 결과: {{ prediction }}</p>
    {% endif %}

    {% if error %}
    <p role="alert">{{ error }}</p>
    {% endif %}
</body>
</html>
```

HTML의 `name` 속성과 `request.form`에서 조회하는 키의 일치 필요.

Jinja2 조건문을 통한 예측 결과 및 오류 메시지 표시.

---

## 8. 실행 및 동작 확인

### 8.1. 모델 학습 및 서버 시작

컨테이너 내부에서 실행.

```bash
python train_model.py
python app.py
```

서버 실행에 필요한 모델 파일 생성을 위해 학습 스크립트 먼저 실행.

브라우저에서 `http://localhost:5000` 접속.

### 8.2. 예측 처리 흐름

```text
브라우저 입력 폼
    │ 네 가지 측정값을 POST 요청으로 전송
    ▼
Flask
    │ 입력값 검증 및 숫자 변환
    ▼
KNN 모델
    │ 붓꽃 품종 예측
    ▼
Jinja2 템플릿
    │ HTML 렌더링
    ▼
브라우저 결과 표시
```

이 예제는 HTML 폼을 통해 예측 결과를 제공하는 웹 애플리케이션.

JSON 기반 예측 서비스는 JSON 입력을 받고 `jsonify()`로 결과를 반환하는 엔드포인트를 추가하여 구현 가능.

### 8.3. 개발 환경 재시작

컨테이너가 중지된 경우 PowerShell에서 실행.

```powershell
docker start flaskserver
docker exec -it flaskserver bash
```

컨테이너 내부에서 서버 시작.

```bash
python app.py
```

동일한 호환 환경에서 저장된 모델을 재사용하는 경우 재학습 없이 실행 가능.


</details>
