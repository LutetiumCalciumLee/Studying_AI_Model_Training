<details>
<summary>ENG (English Version)</summary>

## **Python-based API Server Development with Flask**

### **What is Flask?**
- Lightweight Python web framework for routing, request handling, and template rendering.
- “Micro” means minimal by default, but easily extensible with features like DB, auth, etc.

### **Core Concepts**
- **Basic App Setup**:
```python
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello, World!'
app.run(host='0.0.0.0')
````

* **Routing**:

  * Supports multiple routes per function (`/`, `/index`, `/home`)
  * Dynamic paths (e.g., `/user/<username>`)
  * HTTP method control with `methods=['POST']`
* **Templates (Jinja2)**: Use `render_template()` to pass data to HTML.
* **Static Files**: Served from `/static`, accessed via `url_for()`.
* **Request Data**: Access URL params (`request.args`), form data (`request.form`).

### **Docker + VS Code Setup**

* **Container Setup**:

```bash
docker run -d -it -p 5000:5000 --name flaskserver -v d:\dlproject\flask:/workspace python:3.12-slim
```

* **Install Flask**:

```bash
docker exec -it flaskserver /bin/bash
pip install flask
```

* **VS Code Integration**:

  * Use Dev Containers extension to connect and edit inside the container.

### **Project: ML REST API (Iris Classifier)**

* **Goal**: Build an API that predicts iris flower species from 4 input features.
* **Structure**:

  * `train_model.py`: Train & save KNN model (`iris_model.pkl`)
  * `app.py`: Flask server to load model and return predictions
  * `templates/index.html`: HTML form for user input/output

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## **Flask를 활용한 파이썬 기반 API 서버 개발**

### **Flask란?**

* 라우팅, 요청 처리, 템플릿 렌더링을 제공하는 가벼운 웹 프레임워크.
* "마이크로"는 기능이 최소화되어 있다는 의미이며, DB 연동, 인증 등은 필요 시 확장 가능.

### **핵심 개념**

* **기본 앱 구조**:

```python
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello, World!'
app.run(host='0.0.0.0')
```

* **라우팅(Routing)**:

  * 하나의 함수에 여러 URL 매핑 가능 (`/`, `/index`, `/home`)
  * 동적 URL 사용 가능 (`/user/<username>`)
  * `methods=['POST']`로 HTTP 메서드 지정 가능
* **템플릿(Jinja2)**: `render_template()`로 Python 변수를 HTML에 전달.
* **정적 파일**: `/static` 폴더에서 CSS, JS, 이미지 제공. `url_for()`로 접근.
* **요청 데이터 처리**: `request.args`, `request.form`을 통해 데이터 접근.

### **Docker & VS Code 개발 환경 구성**

* **컨테이너 생성**:

```bash
docker run -d -it -p 5000:5000 --name flaskserver -v d:\dlproject\flask:/workspace python:3.12-slim
```

* **Flask 설치**:

```bash
docker exec -it flaskserver /bin/bash
pip install flask
```

* **VS Code 연동**:

  * Dev Containers 확장 설치 후 컨테이너에 직접 연결하여 개발 가능

### **프로젝트: Iris 꽃 분류 머신러닝 REST API**

* **목표**: 4가지 측정값을 받아서 꽃의 품종을 예측하는 API 서버 구축
* **구성 파일**:

  * `train_model.py`: KNN 모델 학습 후 `iris_model.pkl`로 저장
  * `app.py`: 모델 로드 및 예측 반환하는 Flask 서버
  * `templates/index.html`: 사용자 입력을 받아 결과를 보여주는 HTML 폼

</details>
