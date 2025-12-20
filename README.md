<details>
<summary>ENG (English Version)</summary>

## Local LLM Setup & Usage

**Section 1: LLM Fundamentals**
- Prompt Engineering: Crafting effective prompts for LLMs (input quality determines output); key traits—Clarity (specific), Context (background), Instruction (format/tone).
- LLM Operation: Trained on massive text to predict next words probabilistically; pattern-matching machine, not true reasoning.
- Major LLMs: ChatGPT (multimodal, creative), Claude (long context, ethical), Gemini (video analysis, cost-effective), Grok-3 (math/coding), DeepSeek/LLaMA (open-source alternatives).

**Section 2: Local LLM Overview**
- Local LLM: Runs on personal PC/edge devices (Ollama, LM Studio, llama.cpp); advantages—privacy, offline, cost-free, customizable; drawbacks—hardware demands, setup complexity.

**Section 3: Ollama Installation & Commands**
- Docker Setup: `docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama`; verify at http://127.0.0.1:11434.
- Commands: `docker exec -it ollama bash`; `ollama pull gemma3:1b` (download), `ollama list/show/rm/run <model>` (manage/execute).

**Section 4: Prompting Techniques**
- Zero-Shot: Direct classification ("Text: vacation seems fine. Sentiment:"); Few-Shot: Examples first ("Great movie! // positive"); Chain-of-Thought: "Think step-by-step" for reasoning (apple counting: 10-2-2+5-1=10).

**Section 5: Ollama API Integration**
- Endpoints: `/api/generate` (single prompt), `/api/chat` (conversational with context).
- Flask Proxy: Stream responses via POST `/api/generate` or `/api/chat`; forward to Ollama, return NDJSON streaming; templates for web UI.
- Project: Integrate LLM (Korean→English/image prompts) with ComfyUI for text-to-image pipeline.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## Local LLM 구축 및 활용

**LLM 기초**
- 프롬프트 엔지니어링: LLM에 효과적인 입력 설계(입력 품질=출력 품질); 명확성(구체적), 맥락(배경), 지시(형식/톤) 핵심.
- LLM 동작: 대량 텍스트로 다음 단어 예측 학습; 확률적 패턴 연결, 진짜 사고 아님.
- 주요 LLM: ChatGPT(멀티모달·창의), Claude(긴 맥락·윤리), Gemini(비디오·가성비), Grok-3(수학·코딩), DeepSeek/LLaMA(오픈소스).

**Local LLM 개요**
- Local LLM: 로컬 PC/엣지에서 실행(Ollama, LM Studio, llama.cpp); 장점—프라이버시·오프라인·무료·커스터마이징; 단점—하드웨어·설정 복잡.

**Ollama 설치 및 명령어**
- Docker 구축: `docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama`; http://127.0.0.1:11434 확인.
- 명령어: `docker exec -it ollama bash`; `ollama pull gemma3:1b`(다운로드), `ollama list/show/rm/run <model>`(관리·실행).

**프롬프트 기법**
- Zero-Shot: 직접 분류("휴가 괜찮을 것 같아요. 감정:"), Few-Shot: 예시 먼저("멋지다! // 긍정"), Chain-of-Thought: "단계별 생각"(사과 계산: 10-2-2+5-1=10).

**Ollama API 연동**
- 엔드포인트: `/api/generate`(단일), `/api/chat`(대화형 맥락).
- Flask 프록시: POST `/api/generate` 또는 `/api/chat` 스트리밍; Ollama로 전달 후 NDJSON 반환; 웹 UI 템플릿.
- 과제: LLM(한글→영문/이미지 프롬프트) + ComfyUI 통합 텍스트-이미지 파이프라인.

</details>
