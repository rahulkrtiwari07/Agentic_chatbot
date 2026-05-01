Conversational Voice Agent:

Core Functional Architectures
The system utilizes a modular, multi-agent framework to ensure high-fidelity interactions:
•	Active Listening & Interruptibility: Unlike old "press 1 for sales" systems, this agent allows the user to speak naturally. If the user interrupts, the agent stops talking and listens—just like a human would.
•	Smart Intent Detection: The agent doesn't just look for keywords; it understands the purpose of what you say. It can tell the difference between a user giving an answer, asking for a repeat, or asking a completely different question.
•	Dynamic Tool Orchestration: Utilizing an agentic workflow, the system autonomously interfaces with various data environments:
o	PostgreSQL: For structured administrative data retrieval.
o	MongoDB: For persistent session management and historical user context.
o	Web-Search Integration: For real-time information synthesis.
•	Contextual Guardrails: The system includes a validation layer that assesses the compliance and relevance of user inputs, prompting for corrective information if the data provided is insufficient for the predefined schema.
•	Asynchronous State Persistence: User interactions are synchronized in real-time with a centralized database, ensuring session continuity and the ability to reference historical data across multiple touchpoints.

System requirements:
The program uses following models:
1)	Gemma-3 4B model
2)	STT (Speech-to-Text)
3)	TTS )Text-to-Speech)
4)	ASR (Automatic speech recognition)
*  The above models are hosted on the server.




