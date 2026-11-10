<details>
<summary>ENG (English Version)</summary>

# Week 07 — Image Generation with Stable Diffusion and ComfyUI

Text-to-image generation with a pretrained Stable Diffusion 1.5 checkpoint, a node-based ComfyUI workflow, and a Flask interface connected to the local ComfyUI API.

## 1. Image Generation and Latent Diffusion

GANs, VAEs, and diffusion models represent different approaches to image generation. This exercise uses latent diffusion. CLIP encodes text prompts into conditioning vectors. The denoiser uses those vectors to guide the sampling process in latent space, and the VAE decoder converts the result into pixels. A separate text-to-image embedding prior is part of architectures such as DALL·E 2.

```text
Load Checkpoint ─┬─ MODEL ─────────────────────┐
                 ├─ CLIP → CLIP Text Encode ×2 ├→ KSampler → VAE Decode → Save Image
                 │        Empty Latent Image ──┘                   ↑
                 └─ VAE ───────────────────────────────────────────┘
```

| Node | Role |
| --- | --- |
| Load Checkpoint | Load the diffusion model, CLIP text encoder, and VAE |
| CLIP Text Encode ×2 | Encode positive and negative prompts separately |
| Empty Latent Image | Set the output dimensions and batch size |
| KSampler | Initialize noise from a seed and iteratively denoise the latent representation |
| VAE Decode → Save Image | Convert the result to pixels and save it under `ComfyUI/output/` |

---

## 2. Docker and Model Setup

Requirements include Docker with Linux containers and access to an NVIDIA GPU compatible with the CUDA 12.8 PyTorch build. The example uses Python 3.12 and ComfyUI v0.3.63.

Run on the host.

```powershell
docker run -dit --name comfyui --gpus all -p 127.0.0.1:8188:8188 python:3.12-slim bash
docker exec -it comfyui bash
```

Run inside the container. A result of `12.8 True` from the PyTorch check confirms CUDA is visible.

```bash
apt-get update && apt-get install -y git wget ca-certificates
git clone --depth 1 --branch v0.3.63 https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI
cd /opt/ComfyUI
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt "transformers<5"
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

wget -O models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors \
  https://huggingface.co/Comfy-Org/stable-diffusion-1.5/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
python main.py --listen 0.0.0.0 --port 8188
```

Open [http://127.0.0.1:8188](http://127.0.0.1:8188). Open the default text-to-image workflow, select `v1-5-pruned-emaonly-fp16.safetensors` in **Load Checkpoint**, and click **Run** after entering the prompts.

---

## 3. Prompt and Generation Controls

```text
Positive: A rocky coast at sunset, golden light on dark blue waves, cinematic landscape photography
Negative: blurry, text, watermark
```

| Control | Starting point and comparison |
| --- | --- |
| Sampler / scheduler | Start with `euler` / `normal`. Compare a different sampler with the other settings fixed. |
| Steps | Start at `20`. More denoising steps increase runtime, and the effect on detail depends on the setup. |
| CFG | Start at `7`. Higher guidance can emphasize the prompt and may introduce artifacts. |
| Seed | Start at `42` with `fixed` for comparisons. Use `randomize` in the UI to vary the seed between runs. |
| Resolution | Start at `512 × 512` for SD 1.5. The latent node sets the output dimensions. |

**LoRA exercise** — Place an SD 1.5-compatible LoRA in `models/loras/`. Connect the checkpoint's MODEL and CLIP outputs to **Load LoRA**, then route the adapted MODEL to KSampler and the adapted CLIP to both text encoders. Compare the outputs using the same seed and different LoRA strengths.

---

## 4. ComfyUI API and Flask

| Endpoint | Role |
| --- | --- |
| `POST /prompt` | Submit the API-format node graph and receive a `prompt_id` |
| `GET /history/{prompt_id}` | Poll for execution results |
| `GET /view` | Retrieve a saved image using `filename`, `subfolder`, and `type` |
| `WS /ws` | Receive execution progress events |
| `GET /queue`, `GET /system_stats` | Inspect the queue and system resources |
| `POST /upload/image` | Upload an input image for workflows that use one |

Create the following three files on the host in a separate Flask project folder. This example assumes the browser, Flask, and the published ComfyUI port are on the same host.

**`workflow_api.json`** — A ready-to-use API graph. Nodes `3` and `4` accept the prompts, and node `7` saves the image. For a custom workflow, export it as **API Format** and match those node IDs in the Flask code.

```json
{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
    "ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"
  }},
  "2": {"class_type": "EmptyLatentImage", "inputs": {
    "width": 512, "height": 512, "batch_size": 1
  }},
  "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
  "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
  "5": {"class_type": "KSampler", "inputs": {
    "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
    "latent_image": ["2", 0], "seed": 42, "steps": 20, "cfg": 7,
    "sampler_name": "euler", "scheduler": "normal", "denoise": 1
  }},
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
  "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "Week07"}}
}
```

**`app.py`** — Submit the prompts, poll for a result, and construct the image URL.

```python
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from flask import Flask, render_template, request

app = Flask(__name__)
COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW = Path(__file__).with_name("workflow_api.json")
DEFAULT_POS = (
    "A rocky coast at sunset, golden light on dark blue waves, cinematic landscape photography"
)
DEFAULT_NEG = "blurry, text, watermark"


def generate_image(pos, neg):
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    graph["3"]["inputs"]["text"] = pos
    graph["4"]["inputs"]["text"] = neg

    response = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph}, timeout=10)
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
        response.raise_for_status()
        record = response.json().get(prompt_id)
        if record:
            images = record.get("outputs", {}).get("7", {}).get("images", [])
            if not images:
                raise RuntimeError("ComfyUI finished without a saved image.")
            image = images[0]
            params = {key: image[key] for key in ("filename", "subfolder", "type")}
            return f"{COMFY_URL}/view?{urlencode(params)}"
        time.sleep(1)

    raise TimeoutError("No ComfyUI result within 120 seconds.")


@app.route("/", methods=["GET", "POST"])
def index():
    pos = request.form.get("pos", DEFAULT_POS).strip()
    neg = request.form.get("neg", DEFAULT_NEG).strip()
    image_url = error = None

    if request.method == "POST":
        if not pos:
            error = "Enter a positive prompt."
        else:
            try:
                image_url = generate_image(pos, neg)
            except (requests.RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
                error = str(exc)

    return render_template("image.html", pos=pos, neg=neg, image_url=image_url, error=error)
```

**`templates/image.html`** — Prompt form and image preview.

```html
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Week 07 | ComfyUI</title>
<h1>Text to Image</h1>
<form method="post">
  <label for="pos">Positive prompt</label><br>
  <textarea id="pos" name="pos" rows="3" cols="60" required>{{ pos }}</textarea><br>
  <label for="neg">Negative prompt</label><br>
  <textarea id="neg" name="neg" rows="2" cols="60">{{ neg }}</textarea><br>
  <button type="submit">Generate</button>
</form>
{% if error %}<p role="alert">{{ error }}</p>{% endif %}
{% if image_url %}<img src="{{ image_url }}" alt="Generated image">{% endif %}
</html>
```

From the Flask project folder, start the development server and open [http://127.0.0.1:5000](http://127.0.0.1:5000).

```bash
python -m pip install Flask requests
python -m flask --app app run
```

---

## 5. Process Management and Experiment Log

If Node.js and npm are available in the container, PM2 can manage the ComfyUI process. Stop the foreground ComfyUI process first, then run from `/opt/ComfyUI`.

```bash
npm install -g pm2
pm2 start main.py --name comfyui --interpreter python -- --listen 0.0.0.0 --port 8188
pm2 logs comfyui
```

- Compare prompt or CFG changes with the checkpoint, seed, and resolution fixed.
- Record the prompt pair, checkpoint, seed, sampler, scheduler, steps, CFG, and LoRA strength alongside the output.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# Week 07 — Stable Diffusion과 ComfyUI를 활용한 이미지 생성

사전 학습된 Stable Diffusion 1.5 체크포인트로 텍스트 기반 이미지 생성. ComfyUI의 노드 워크플로우 구성 및 로컬 API와 Flask 웹 화면 연동.

## 1. 이미지 생성과 잠재 확산

GAN, VAE, 확산 모델은 이미지 생성에 활용되는 서로 다른 접근 방식. 이번 실습은 잠재 확산을 적용해 CLIP으로 프롬프트를 조건 벡터로 변환하고, 이 조건을 바탕으로 잠재 공간의 노이즈를 단계적으로 제거한 뒤 VAE 디코더로 픽셀 이미지 복원. 텍스트에서 이미지 임베딩을 만드는 별도 Prior는 DALL·E 2와 같은 구조에서 사용.

```text
Load Checkpoint ─┬─ MODEL ─────────────────────┐
                 ├─ CLIP → CLIP Text Encode ×2 ├→ KSampler → VAE Decode → Save Image
                 │        Empty Latent Image ──┘                   ↑
                 └─ VAE ───────────────────────────────────────────┘
```

| 노드 | 역할 |
| --- | --- |
| Load Checkpoint | 확산 모델, CLIP 텍스트 인코더, VAE 로드 |
| CLIP Text Encode ×2 | 긍정·부정 프롬프트를 각각 인코딩 |
| Empty Latent Image | 출력 해상도와 배치 크기 설정 |
| KSampler | 시드로 초기 노이즈를 결정하고 잠재 표현의 노이즈를 단계적으로 제거 |
| VAE Decode → Save Image | 결과를 픽셀 이미지로 변환한 뒤 `ComfyUI/output/`에 저장 |

---

## 2. Docker 환경 및 모델 설치

Linux 컨테이너를 실행할 수 있는 Docker와 CUDA 12.8용 PyTorch를 지원하는 NVIDIA GPU 환경 준비. 실습 버전은 Python 3.12와 ComfyUI v0.3.63.

호스트에서 컨테이너 생성 및 접속.

```powershell
docker run -dit --name comfyui --gpus all -p 127.0.0.1:8188:8188 python:3.12-slim bash
docker exec -it comfyui bash
```

컨테이너 내부에서 패키지와 체크포인트 설치 후 ComfyUI 실행. PyTorch 확인 결과가 `12.8 True`이면 CUDA 사용 가능 상태.

```bash
apt-get update && apt-get install -y git wget ca-certificates
git clone --depth 1 --branch v0.3.63 https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI
cd /opt/ComfyUI
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt "transformers<5"
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

wget -O models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors \
  https://huggingface.co/Comfy-Org/stable-diffusion-1.5/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
python main.py --listen 0.0.0.0 --port 8188
```

[http://127.0.0.1:8188](http://127.0.0.1:8188) 접속. 기본 Text-to-Image 워크플로우를 열고 **Load Checkpoint**에서 `v1-5-pruned-emaonly-fp16.safetensors` 선택 후 프롬프트를 입력해 **Run** 실행.

---

## 3. 프롬프트와 생성 설정

```text
Positive: A rocky coast at sunset, golden light on dark blue waves, cinematic landscape photography
Negative: blurry, text, watermark
```

| 설정 | 시작값과 비교 방법 |
| --- | --- |
| Sampler / Scheduler | `euler` / `normal`로 시작하고 나머지 설정을 고정해 다른 샘플러와 비교 |
| Steps | `20`으로 시작. 단계 수가 늘수록 처리 시간이 증가하며 세부 묘사의 변화는 설정에 따라 달라짐 |
| CFG | `7`로 시작. 높은 값은 프롬프트 조건의 영향을 강화하며 지나치면 이미지 왜곡 가능 |
| Seed | `42`와 `fixed`로 비교 조건 고정. 실행마다 시드를 바꿀 때는 화면에서 `randomize` 선택 |
| 해상도 | SD 1.5에서는 `512 × 512`로 시작. 잠재 이미지 노드의 크기가 출력 해상도를 결정 |

**LoRA 실습** — SD 1.5 호환 LoRA 파일을 `models/loras/`에 배치. 체크포인트의 MODEL·CLIP 출력을 **Load LoRA**에 연결하고, 적용된 MODEL은 KSampler로, CLIP은 두 텍스트 인코더로 연결. 같은 시드로 LoRA 적용 강도에 따른 결과 비교.

---

## 4. ComfyUI API와 Flask 연동

| 엔드포인트 | 역할 |
| --- | --- |
| `POST /prompt` | API 형식의 노드 그래프 제출 및 `prompt_id` 수신 |
| `GET /history/{prompt_id}` | 작업 완료와 실행 결과 확인 |
| `GET /view` | `filename`, `subfolder`, `type`을 이용한 저장 이미지 조회 |
| `WS /ws` | 실행 진행 이벤트 수신 |
| `GET /queue`, `GET /system_stats` | 작업 대기열과 시스템 자원 조회 |
| `POST /upload/image` | 입력 이미지가 필요한 워크플로우에 이미지 업로드 |

호스트에 별도 Flask 실습 폴더를 만들고 아래 세 파일 저장. 브라우저와 Flask 서버에서 같은 호스트에 공개한 ComfyUI 포트에 접속하는 구성.

**`workflow_api.json`** — 실습용 API 그래프. 프롬프트 입력 노드는 `3`, `4`, 이미지 저장 노드는 `7` 사용. 직접 만든 워크플로우는 **API Format**으로 내보낸 뒤 Flask 코드의 노드 ID와 일치하도록 수정.

```json
{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
    "ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"
  }},
  "2": {"class_type": "EmptyLatentImage", "inputs": {
    "width": 512, "height": 512, "batch_size": 1
  }},
  "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
  "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
  "5": {"class_type": "KSampler", "inputs": {
    "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
    "latent_image": ["2", 0], "seed": 42, "steps": 20, "cfg": 7,
    "sampler_name": "euler", "scheduler": "normal", "denoise": 1
  }},
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
  "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "Week07"}}
}
```

**`app.py`** — 프롬프트 제출, 완료 결과 확인 및 이미지 URL 구성.

```python
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from flask import Flask, render_template, request

app = Flask(__name__)
COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW = Path(__file__).with_name("workflow_api.json")
DEFAULT_POS = (
    "A rocky coast at sunset, golden light on dark blue waves, cinematic landscape photography"
)
DEFAULT_NEG = "blurry, text, watermark"


def generate_image(pos, neg):
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    graph["3"]["inputs"]["text"] = pos
    graph["4"]["inputs"]["text"] = neg

    response = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph}, timeout=10)
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
        response.raise_for_status()
        record = response.json().get(prompt_id)
        if record:
            images = record.get("outputs", {}).get("7", {}).get("images", [])
            if not images:
                raise RuntimeError("ComfyUI finished without a saved image.")
            image = images[0]
            params = {key: image[key] for key in ("filename", "subfolder", "type")}
            return f"{COMFY_URL}/view?{urlencode(params)}"
        time.sleep(1)

    raise TimeoutError("No ComfyUI result within 120 seconds.")


@app.route("/", methods=["GET", "POST"])
def index():
    pos = request.form.get("pos", DEFAULT_POS).strip()
    neg = request.form.get("neg", DEFAULT_NEG).strip()
    image_url = error = None

    if request.method == "POST":
        if not pos:
            error = "Enter a positive prompt."
        else:
            try:
                image_url = generate_image(pos, neg)
            except (requests.RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
                error = str(exc)

    return render_template("image.html", pos=pos, neg=neg, image_url=image_url, error=error)
```

**`templates/image.html`** — 프롬프트 입력 폼과 생성 결과 표시.

```html
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Week 07 | ComfyUI</title>
<h1>Text to Image</h1>
<form method="post">
  <label for="pos">Positive prompt</label><br>
  <textarea id="pos" name="pos" rows="3" cols="60" required>{{ pos }}</textarea><br>
  <label for="neg">Negative prompt</label><br>
  <textarea id="neg" name="neg" rows="2" cols="60">{{ neg }}</textarea><br>
  <button type="submit">Generate</button>
</form>
{% if error %}<p role="alert">{{ error }}</p>{% endif %}
{% if image_url %}<img src="{{ image_url }}" alt="Generated image">{% endif %}
</html>
```

Flask 실습 폴더에서 개발 서버 실행 후 [http://127.0.0.1:5000](http://127.0.0.1:5000) 접속.

```bash
python -m pip install Flask requests
python -m flask --app app run
```

---

## 5. 프로세스 관리 및 실험 기록

컨테이너에 Node.js와 npm이 설치된 경우 PM2로 ComfyUI 프로세스 관리 가능. 기존 포그라운드 프로세스를 종료한 뒤 `/opt/ComfyUI`에서 실행.

```bash
npm install -g pm2
pm2 start main.py --name comfyui --interpreter python -- --listen 0.0.0.0 --port 8188
pm2 logs comfyui
```

- 체크포인트, 시드, 해상도를 고정하고 프롬프트나 CFG를 하나씩 바꿔 결과 비교.
- 생성 이미지와 함께 프롬프트 쌍, 체크포인트, 시드, 샘플러, 스케줄러, Steps, CFG, LoRA 적용 강도 기록.

</details>
