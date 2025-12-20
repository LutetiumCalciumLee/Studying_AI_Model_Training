<details>
<summary>ENG (English Version)</summary>

## LangChain Chain Design & Memory

**Section 1: LangChain Chains**
- Chain Concept: Modular pipelines combining Prompt→LLM→Parser using LCEL (`chain = prompt | llm | StrOutputParser()`).
- Chain Types: LLMChain (Q&A/summary), SequentialChain (summary→translation), RouterChain (input-based branching).

**Section 2: Memory Types**
- Memory Role: Automatically stores input/output, injects conversation history into prompts for context-aware responses.
- Types: ConversationBufferMemory (full history), ConversationSummaryMemory (summarized for long chats), ConversationBufferWindowMemory (recent N turns).

**Section 3: Basic Chain Example**
- JSON Chain: `ChatPromptTemplate | ChatOllama("gemma3:4b") | JsonOutputParser()` for structured text summarization.

**Section 4: Sequential Chain (Summary→Translation)**
- LCEL Pipeline: `{"summary": summary_chain} | RunnablePassthrough.assign(translated=translate_chain)`; input `{"content": text}` → output `{"summary": "...", "translated": "..."}`.

**Section 5: Memory Implementation**
- Legacy: `ConversationChain(llm, ConversationBufferMemory())` for simple chatbots.
- Modern: `RunnableWithMessageHistory(base_chain, get_history, input_messages_key="input", history_messages_key="history")`; uses `MessagesPlaceholder("history")` and session ID config.
- Flask Chatbot: Cookie-based session recovery, chat deletion, SSE streaming for real-time responses.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## LangChain Chain 설계와 메모리

**LangChain 체인**
- 체인 개념: Prompt→LLM→Parser 모듈 파이프라인, LCEL 사용(`chain = prompt | llm | StrOutputParser()`).
- 체인 유형: LLMChain(Q&A/요약), SequentialChain(요약→번역), RouterChain(입력 분기).

**메모리 종류**
- 메모리 역할: input/output 자동 저장, 다음 프롬프트에 대화 히스토리 주입으로 문맥 유지.
- 종류: ConversationBufferMemory(전체), ConversationSummaryMemory(요약·긴대화), ConversationBufferWindowMemory(최근 N회).

**기본 체인 예제**
- JSON 체인: `ChatPromptTemplate | ChatOllama("gemma3:4b") | JsonOutputParser()` 구조화 요약.

**순차 체인 (요약→번역)**
- LCEL 파이프라인: `{"summary": summary_chain} | RunnablePassthrough.assign(translated=translate_chain)`; 입력 `{"content": text}` → 출력 `{"summary": "...", "translated": "..."}`.

**메모리 구현**
- 구버전: `ConversationChain(llm, ConversationBufferMemory())` 단순 챗봇.
- 신버전: `RunnableWithMessageHistory(base_chain, get_history, input_messages_key="input", history_messages_key="history")`; `MessagesPlaceholder("history")` + 세션 ID config.
- Flask 챗봇: 쿠키 세션 복구, 대화 삭제, SSE 스트리밍 실시간 응답.

</details>
