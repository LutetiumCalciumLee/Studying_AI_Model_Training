<details>
<summary>ENG (English Version)</summary>

## Chapter 5 – LangChain Overview & Structure

**Section 1: LangChain Core Components**
- LangChain: Open-source framework connecting LLMs to external data/tools; enables complex LLM applications via chaining.
- Models: LLMs (text-in/text-out), Chat Models (role-based messages), Embeddings (text-to-vector for similarity search).
- Prompts: Managed via templates e.g., `"Suggest 3 creative advertising slogans for {product}."`.

**Section 2: Advanced Components**
- Indexes/Retrieval (RAG): Document Loaders (PDF/TXT/web), Text Splitters (chunking), Vector Stores (semantic search).
- Chains: Sequential pipelines—simple (prompt→LLM→output), complex (query→search→augmented prompt→LLM).
- Memory: Stores conversation context for chatbots; Agents: LLM as reasoning engine selecting Tools (search, calculator, APIs).

**Section 3: LangChain + Ollama Setup**
- Install: `pip install langchain==0.3.17 langchain-ollama langchain-text-splitters flask`.
- Hello Chain: `ChatOllama(model="gemma3:4b") | ChatPromptTemplate | StrOutputParser()` for basic Q&A.
- Summary Chain: Template `"Summarize the following text into 5 key bullet points in Korean.\n\nText:\n{content}"` → Ollama → parser.

**Section 4: Flask Integration**
- REST API: POST `/summarize` receives text, invokes chain, returns JSON summary.
- SSE Streaming: `/summarize/stream` yields tokens as generated via `chain.stream()` with Server-Sent Events.
- Structured JSON: `JsonOutputParser` enforces JSON output (`summary`, `keywords`, `oneline`).

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## 5장 – LangChain 개요와 구조

**LangChain 핵심 구성요소**
- LangChain: LLM 기반 애플리케이션 개발 오픈소스 프레임워크; 외부 데이터/도구와 LLM 연결(Chain).
- 모델: LLM(텍스트↔텍스트), Chat Models(역할 메시지), 임베딩(텍스트→벡터 유사도 검색).
- 프롬프트: 템플릿 관리 e.g., `"{product}에 대한 창의적인 광고문구 3가지를 제안해줘."`.

**고급 구성요소**
- 인덱스/검색(RAG): 문서 로더(PDF/TXT/웹), 텍스트 분할기(청크), 벡터 스토어(의미 검색).
- 체인: 순차 파이프라인—단순(프롬프트→LLM→출력), 복잡(질문→검색→증강 프롬프트→LLM).
- 메모리: 대화 맥락 저장(챗봇); 에이전트: LLM 추론엔진+도구(검색·계산기·API) 선택.

**LangChain + Ollama 구축**
- 설치: `pip install langchain==0.3.17 langchain-ollama langchain-text-splitters flask`.
- Hello 체인: `ChatOllama("gemma3:4b") | ChatPromptTemplate | StrOutputParser()` 기본 Q&A.
- 요약 체인: `"다음 텍스트를 핵심 bullet 5개로 한국어 요약해줘.\n\n{content}"` → Ollama → 파서.

**Flask 연동**
- REST API: POST `/summarize` 텍스트 수신→체인 호출→JSON 요약 반환.
- SSE 스트리밍: `/summarize/stream` `chain.stream()` 토큰 실시간 전송(Server-Sent Events).
- 구조화 JSON: `JsonOutputParser` 강제 JSON 출력(`summary`, `keywords`, `oneline`).

</details>
