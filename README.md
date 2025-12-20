<details>
<summary>ENG (English Version)</summary>

## LangChain Tools & Agents 

**Section 1: LangChain Tools**
- Tool Decorator: `@tool` creates functions for agents—`calc(expr)` (arithmetic calculator using `ast`), `web_search(query)` (DuckDuckGo via `ddgs` library).
- Calc Tool: Safe expression evaluation (`+,-,*,/,()`) with `ast.parse()` and operator mapping.
- Web Search Tool: Returns formatted results (`title | URL | snippet`) for agent readability.

**Section 2: ReAct Agent Implementation**
- ReAct Prompt: "Reasoning + Acting" format—`Question → Thought → Action → Action Input → Observation` loop until `Final Answer`.
- Agent Creation: `create_react_agent(llm, tools, react_prompt)` + `AgentExecutor(agent, tools, verbose=True)`.
- Example Query: "I'm curious about the latest Seoul population figures. Please search and give me a brief summary. Also calculate 1+2*3 at the end." → Agent searches population, summarizes, calculates `1+2*3=7`.

**Setup Requirements**
```
pip install ddgs
```
Uses `ChatOllama("gemma3:4b")` with Ollama backend.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## LangChain Tools & Agent

**LangChain Tools**
- Tool 데코레이터: `@tool`로 에이전트용 함수 생성—`calc(expr)`(산술 계산기 `ast` 사용), `web_search(query)`(DuckDuckGo `ddgs` 라이브러리).
- Calc Tool: 안전한 식 평가(`+,-,*,/,()`) `ast.parse()` + 연산자 매핑.
- Web Search Tool: 에이전트 읽기 쉬운 형식(`제목 | URL | 스니펫`) 반환.

**ReAct 에이전트 구현**
- ReAct 프롬프트: "추론+행동" 형식—`Question → Thought → Action → Action Input → Observation` 반복 → `Final Answer`.
- 에이전트 생성: `create_react_agent(llm, tools, react_prompt)` + `AgentExecutor(agent, tools, verbose=True)`.
- 예제 쿼리: "서울인구최신수치가궁금해. 검색하고간단히요약해줘. 마지막에1+2*3도계산해." → 인구 검색·요약·`1+2*3=7` 계산.

**설치 요구사항**
```
pip install ddgs
```
Ollama 백엔드 `ChatOllama("gemma3:4b")` 사용.

</details>
