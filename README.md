<details>
<summary>ENG (English Version)</summary>

## Image Generation AI: ComfyUI

**Section 1: Image Generation AI Overview**
- Core Technologies: GANs, VAEs, Diffusion Models, Transformer+Denoising; key models include DALL-E 2/3, Midjourney, Imagen, Stable Diffusion, Adobe Firefly.
- Text-to-Image Pipeline: Text Encoder (CLIP converts prompts to vectors), Prior (maps text embeddings to image embeddings), Decoder (generates actual images from embeddings).

**Section 2: Stable Diffusion Usage**
- Prompt Engineering: Detailed positive prompts with subject+context+adjectives; negative prompts exclude unwanted elements ("low quality, worst quality").
- Generation Parameters: Sampling method (Euler, DPM++SDE), Steps (higher=better quality), CFG Scale (higher=prompt adherence), Seed (-1=reuse, -2=new).
- Model Usage: Combine checkpoint models (base) with LoRA (lightweight fine-tunes) using `<lora:name:weight>` syntax in prompts (weight=1 default).

**Section 3: ComfyUI Installation**
- Docker Setup: `docker run -d --gpus=all -p 8188:8188 python:3.12-slim`; exec into container, install PyTorch (CUDA), git, wget.
- Git Clone: `git clone https://github.com/comfyanonymous/ComfyUI.git`; `pip install -r requirements.txt`; run `python main.py --listen 0.0.0.0 --port 8188`.
- Access: http://127.0.0.1:8188; node-based workflow interface with template explorer.

**Section 4: ComfyUI Usage**
- Model Download: Stable Diffusion v1-5 from HuggingFace (`wget` URL); move to `models/checkpoints/`; restart ComfyUI.
- Workflow Execution: Load template, set prompts, click Execute; modify prompts with styles (8k, ultra-realistic, unreal engine).

**Section 5: Service Management & API**
- PM2 Management: Install Node.js, `npm i -g pm2`; `pm2 start main.py --name myapp --interpreter python3 -- --listen 0.0.0.0 --port 8188`.
- API Workflow: POST /prompt (workflow JSON), WebSocket progress, GET /history/{prompt_id} (results); endpoints for queue, history, uploads, monitoring.
- Flask Integration: Load workflow JSON, update prompts, submit to ComfyUI API, poll results, display generated images in web app.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## 이미지 생성 AI: ComfyUI

**이미지 생성 AI 개요**
- 핵심 기술: GAN, VAE, 확산모델(Diffusion), Transformer+Denoising; 주요 모델 DALL-E 2/3, Midjourney, Imagen, Stable Diffusion, Adobe Firefly.
- Text-to-Image 파이프라인: 텍스트 인코더(CLIP: 프롬프트→벡터), Prior(텍스트 임베딩→이미지 임베딩), 디코더(임베딩→실제 이미지).

**Stable Diffusion 활용법**
- 프롬프트 작성: 긍정 프롬프트(주제+상황+형용사로 구체화), 부정 프롬프트("low quality, worst quality" 등 원치 않는 요소 제외).
- 생성 파라미터: 샘플링(Euler, DPM++SDE), Steps(높을수록 품질↑), CFG Scale(높을수록 프롬프트 충실), Seed(-1:재생성, -2:신규).
- 모델 조합: 체크포인트(기본) + LoRA(경량 보조) 사용, 프롬프트에 `<lora:이름:가중치>` 형식(multiplier 기본 1).

**ComfyUI 설치**
- Docker 구축: `docker run -d --gpus=all -p 8188:8188 python:3.12-slim`; 컨테이너 접속 후 PyTorch(CUDA), git, wget 설치.
- Git 클론: `git clone https://github.com/comfyanonymous/ComfyUI.git`; `pip install -r requirements.txt`; `python main.py --listen 0.0.0.0 --port 8188` 실행.
- 접근: http://127.0.0.1:8188; 노드 기반 워크플로우 인터페이스, 템플릿 탐색 기능.

**ComfyUI 사용법**
- 모델 다운로드: HuggingFace Stable Diffusion v1-5 (`wget`), `models/checkpoints/` 이동 후 ComfyUI 재시작.
- 워크플로우 실행: 템플릿 로드, 프롬프트 설정 후 Execute 클릭; 스타일 추가(8k, ultra-realistic, unreal engine).

**서비스 관리 & API**
- PM2 관리: Node.js 설치, `npm i -g pm2`; `pm2 start main.py --name myapp --interpreter python3 -- --listen 0.0.0.0 --port 8188`.
- API 워크플로우: POST /prompt(워크플로우 JSON), WebSocket 진행상태, GET /history/{prompt_id}(결과); 큐/히스토리/업로드/모니터링 엔드포인트.
- Flask 연동: JSON 워크플로우 로드, 프롬프트 업데이트, ComfyUI API 제출, 결과 폴링 후 웹에서 이미지 표시.

</details>
