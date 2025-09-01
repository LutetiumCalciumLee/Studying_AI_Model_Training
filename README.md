# Chapter 2. Python-based API Server Development

---

## What is Flask?
- A Python "micro" web framework that provides essential features like routing, request handling, and template rendering.
- Its "micro" design means it is lightweight and highly extensible, allowing developers to add features like database integration or authentication as needed.

## Core Concepts
- Basic App Structure: A minimal Flask app initializes the `Flask` class and uses a route decorator to map a URL to a Python function.
  ```python
  from flask import Flask
  app = Flask(__name__)
  @app.route('/')
  def hello():
      return 'Hello, World!'
  if __name__ == '__main__':
      app.run(host='0.0.0.0')
  ```
- Routing: Connects URLs to functions.
    - Multiple URLs: A single function can be mapped to multiple routes (e.g., `/`, `/index`, `/home`).
    - Dynamic Routing: URL parts can be used as variables (e.g., `/user/<username>`).
    - HTTP Methods: Routes can be configured to handle specific methods like `POST` using `methods=['POST']`.
- Templates (Jinja2): The `render_template()` function passes Python variables to an HTML file in the `templates` folder for dynamic content rendering.
- Static Files: CSS, JS, and images are served from a `static` folder and linked using `url_for('static', filename='...')`.
- Request Handling: Access incoming data via the `request` object (e.g., `request.args.get('name')` for URL parameters or `request.form['user_input']` for form data).

## Hands-on: Docker & VS Code Setup
- Create Container: Run a Python container, mapping a local project directory and port.
  ```bash
  docker run -d -it -p 5000:5000 --name flaskserver -v d:\dlproject\flask:/workspace python:3.12-slim
  ```
- Install Flask: Access the container's shell and install Flask.
  ```bash
  docker exec -it flaskserver /bin/bash
  pip install flask
  ```
- Develop in VS Code: Use the Dev Containers extension to connect VS Code directly to the `flaskserver` container for a seamless development experience.

## Project: Machine Learning REST API for Iris Classification
- Goal: Build a web API that takes four measurements of an iris flower and predicts its species using a pre-trained K-Nearest Neighbors (KNN) model.
- Project Structure:
    - `train_model.py`: Script to train the KNN model and save it as `iris_model.pkl`.
    - `app.py`: The Flask server that loads the model, handles requests, and returns predictions.
    - `templates/index.html`: The user-facing HTML form for input and displaying results.
