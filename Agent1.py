import os
import logging
import asyncio
import json
import time
import requests

import re
import ast

from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from Prompts1 import AgentConfig

from mongooperations import get_mongo_client, store_user_data
from extract_mongo import retrieve_data

from stream_lance import Retrieval  # Make sure this exists and is implemented

from uuid import uuid4
from dotenv import load_dotenv

from fastapi.responses import StreamingResponse

from questions import QUESTIONS

mongo_uri = os.getenv('MONGO_URL')
database_name = "chat_db"
collection_name = "user_data"
collection_name_1 = "user_data_1"

mongo_client = get_mongo_client(mongo_uri)
print(mongo_client)

load_dotenv('.env.example')

from uuid import uuid4

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, questions=None):
        session_id = str(uuid4())
        if questions:
            first_key, first_question = questions[0]
        else:
            first_key, first_question = None, None

        self.sessions[session_id] = {
            "answers": {},
            "current_key": first_key,
            "current_question": first_question,
            "welcomed": False,
            "negative_count": 0
        }
        return session_id

    def get_state(self, session_id):
        return self.sessions.get(
            session_id,
            {
                "answers": {},
                "current_key": None,
                "current_question": None,
                "welcomed": False
            }
        )

    def update_state(self, session_id, state):
        self.sessions[session_id] = state

    def has_session(self, session_id):
        return session_id in self.sessions




class IntentRouter:
    def __init__(self):

        self.session_manager = SessionManager()
        # Load config from environment
        self.es_pass = os.getenv('ELASTICSEARCH_KEY')
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://container1.openai.azure.com/")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")

        # Set up retrieval
        #self.retrieval = Retrieval(es_pass=self.es_pass)
        self.retrieval = Retrieval()
        self.retriever = self.retrieval.retrieve_data()

        self.questions = QUESTIONS


        self.log_path = "interaction.docx"

        # LLMs
        '''self.llm1 = AzureChatOpenAI(
            azure_deployment=self.azure_deployment,
            api_version="2023-06-01-preview",
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            temperature=0,
            streaming=False,
            max_retries=2,
        )'''
        self.url = "http://164.52.193.73:9000/v1/chat/completions"

        '''self.url = ChatOpenAI(
            model="/model_weights/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
            base_url="http://164.52.193.73:8070/v1",
            api_key="EMPTY"
        )'''

        '''self.llm2 = AzureChatOpenAI(
            azure_deployment=self.azure_deployment,
            api_version="2023-06-01-preview",
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            temperature=0.9,
            streaming=True,
            max_retries=2,
        )'''

        # Build graph
        self.graph = self._build_graph()

    def route_agent(self, state):
        input_text = state["input"]

        messages = [
            {"role": "system", "content": AgentConfig.DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": f"User query: {input_text}"}
        ]

        payload = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 500
        }

        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()

            # --- Extract model output ---
            content = data["choices"][0]["message"]["content"].strip()

            # --- Clean potential markdown wrappers ---
            # Removes leading/trailing backticks like ```json ... ```
            content = re.sub(r"^```(?:json)?|```$", "", content.strip())

            # --- Fix newline issues inside JSON ---
            # Replace unescaped newlines (common hallucination) with spaces
            content = re.sub(r'(?<!\\)\n', ' ', content)

            # Try JSON parsing
            try:
                decision = json.loads(content)
            except json.JSONDecodeError:
                logging.warning(f"Route agent failed to parse JSON.\nRaw content: {content}")
                decision = {
                    "agent": "parse_error_fallback_agent",
                    "reasoning": f"Invalid JSON from LLM: {content}",
                    "confidence": 0.0
                }

            # --- Confidence gating ---
            if decision.get("confidence", 0) < AgentConfig.CONFIDENCE_THRESHOLD:
                decision["agent"] = "low_confidence_fallback_agent"

        except Exception as e:
            logging.warning(f"Route agent request error: {e}")
            decision = {
                "agent": "parse_error_fallback_agent",
                "reasoning": f"Request failed: {e}",
                "confidence": 0.0
            }

        # Merge with state for LangGraph
        return {**state, **decision}

    
    def format_chat_log(self, chat_log):
        if not chat_log:
            return ""
        # If it's a dict (single message), format it accordingly
        if isinstance(chat_log, dict):
            # For example, if dict has keys 'role' and 'message' or 'content'
            role = chat_log.get('role', 'unknown').capitalize()
            content = chat_log.get('message') or chat_log.get('content') or ''
            return f"{role}: {content}"
        # If it's a list of dicts (a conversation history)
        if isinstance(chat_log, list):
            return "\n".join(
                f"{entry.get('role', 'unknown').capitalize()}: {entry.get('message') or entry.get('content', '')}" 
                for entry in chat_log
            )
        # Otherwise just convert to string
        return str(chat_log)

    


    def classify_intent(self, user_input, question=None, chat_log=None):
        if chat_log is None:
            formatted_log = ""
        else:
            formatted_log = self.format_chat_log(chat_log)

        system_prompt = AgentConfig.INTENT_CLASSIFIER_PROMPT.replace("{chat_log}", formatted_log)

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
            ],
            "temperature": 0,
            "max_tokens": 200
        }

        try:
            start_time = time.time()
            response = requests.post(self.url, json=payload)
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            #print("[DEBUG] Raw output:", response.text)

            response.raise_for_status()
            data = response.json()

            # Extract assistant content
            content = data["choices"][0]["message"]["content"]

            # Try to grab JSON block if wrapped in json ... 
            match = re.search(r"json\s*(\{.*?\})\s*", content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Otherwise, try to find the { ... } part manually
                match = re.search(r"\{.*\}", content, re.DOTALL)
                json_str = match.group(0) if match else ""

            intent = "unknown"
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    intent = parsed.get("intent", "unknown")
                except Exception as e:
                    print("[DEBUG] JSON parse failed, trying regex:", e)
                    # Fallback: regex extract "intent": "value"
                    match = re.search(r'"intent"\s*:\s*"([^"]+)"', json_str)
                    if match:
                        intent = match.group(1)

            # Absolute last fallback: regex on full content
            if intent == "unknown":
                match = re.search(r'"intent"\s*:\s*"([^"]+)"', content)
                if match:
                    intent = match.group(1)

        except Exception as e:
            logging.warning(f"Intent classification error: {e}")
            intent = "unknown"
        print(intent)
        return intent



    
    def classify_answer(self, user_input, question=None):
        print(question)
        messages = [
            {"role": "system", "content": AgentConfig.ANSWER_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        payload = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 200
        }

        value = "unknown"  # default fallback

        try:
            start_time = time.time()
            response = requests.post(self.url, json=payload)
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            #print("[DEBUG] Raw output:", response.text)

            response.raise_for_status()
            data = response.json()

            # Extract content string
            content = data["choices"][0]["message"]["content"]

            # Try to find a JSON block
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    value = parsed.get("intent") or parsed.get("clarification") or "unknown"
                except Exception as e:
                    logging.warning(f"JSON parse failed: {e}")
                    # fallback: regex search for intent value
                    intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', content)
                    clar_match = re.search(r'"clarification"\s*:\s*"([^"]+)"', content)
                    if intent_match:
                        value = intent_match.group(1)
                    elif clar_match:
                        value = clar_match.group(1)
            else:
                logging.warning("No JSON block found in model output")

        except Exception as e:
            logging.warning(f"Answer classification error: {e}")

        return value




    async def purpose_classifier(self, user_input, question):
        print("purpose_classifier")

        messages = [
            {"role": "system", "content": AgentConfig.PURPOSE_CLASSIFIER_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        payload = {
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": 500
        }

        try:
            start_time = time.time()
            response = requests.post(self.url, json=payload)
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            print("[DEBUG] Purpose classifier raw output:", response.text)

            response.raise_for_status()
            data = response.json()

            # Extract assistant response content
            content = data["choices"][0]["message"]["content"].strip()

            # Try parsing JSON block if present
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    return parsed.get("purpose", content)  # expect {"purpose": "..."}
                except Exception as e:
                    logging.warning(f"JSON parse failed, returning raw text: {e}")
                    return content
            else:
                return content  # fallback: plain text

        except Exception as e:
            logging.warning(f"Purpose classification error: {e}")
            return "unknown"

    async def interact(self, state):
        answers = state.get("answers", {})
        
        for key, question in self.questions:
            if key not in answers:
                state["current_key"] = key
                return {
                    "status": "asking",
                    "question": question,
                    "answers": answers,
                    "current_key": key
                }

        return {
            "status": "complete",
            "answers": answers
        }

    
    def greeting(self, user_input, question):
        print("greeting")

        messages = [
            {"role": "system", "content": AgentConfig.GREETING_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        payload = {
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": 300
        }

        try:
            start_time = time.time()
            response = requests.post(self.url, json=payload)
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            print("[DEBUG] Greeting raw output:", response.text)

            response.raise_for_status()
            data = response.json()

            # Extract assistant message content
            content = data["choices"][0]["message"]["content"].strip()

            # Try parsing JSON block if present
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    return parsed.get("greeting", content)  # expect {"greeting": "..."}
                except Exception as e:
                    logging.warning(f"JSON parse failed in greeting, returning raw text: {e}")
                    return content
            else:
                return content  # fallback: plain text

        except Exception as e:
            logging.warning(f"Greeting classification error: {e}")
            return "unknown"
    
    async def simple_llm(self, state):
        full_response = ""
        try:
            logging.info("simple_llm started")
            print("conversational_agent")

            # Use a prompt from the state or default to a basic instruction
            system_prompt = state.get("system_prompt", AgentConfig.CLARIFICATION_ASSISTANT_PROMPT)
            user_input = state["input"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]

            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 800,
                "stream": False  # set to True if your FastAPI proxy supports SSE streaming
            }

            start_time = time.time()
            response = requests.post(self.url, json=payload)
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            print("[DEBUG] Simple LLM raw output:", response.text)

            response.raise_for_status()
            data = response.json()

            # Extract assistant's response
            content = data["choices"][0]["message"]["content"].strip()

            # Simulate streaming by sending chunks of text
            for i in range(0, len(content), 50):  # adjust chunk size as needed
                chunk = content[i:i+50]
                full_response += chunk
                yield {"response": full_response}
                await asyncio.sleep(0.05)  # tiny delay to simulate stream updates

        except Exception as e:
            logging.exception("Error during streaming")
            yield {"response": f"Error processing response: {e}"}

    async def general_agent(self, state):
        full_response = ""
        try:
            logging.info("general_agent started")
            print("general_agent")

            # Use a prompt from the state or default to a basic instruction
            system_prompt = state.get("system_prompt", AgentConfig.QUERY_PROMPT)
            user_input = state["input"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]

            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 800,
                "stream": False  # if your FastAPI proxy supports SSE, set this True
            }

            start_time = time.time()
            response = requests.post(self.url, json=payload)
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            #print("[DEBUG] General agent raw output:", response.text)

            response.raise_for_status()
            data = response.json()

            # Extract assistant's message
            content = data["choices"][0]["message"]["content"].strip()

            # Simulate streaming by chunking output
            for i in range(0, len(content), 50):  # adjust chunk size if needed
                chunk = content[i:i+50]
                full_response += chunk
                yield {"response": full_response}
                await asyncio.sleep(0.05)  # small delay to mimic live streaming

        except Exception as e:
            logging.exception("Error during general_agent streaming")
            yield {"response": f"Error processing response: {e}"}

    async def RAG(self, state):
        query = state['input']
        email = state.get("email", "default@example.com")

        print("RAG")
        # Immediately yield initial message
        yield {"response": "Please wait while I fetch the information from the knowledge base..."}

        # Await the final full response
        final_response = await self.retrieval.response_llm(query, email)

        # Yield the actual result
        yield {"response": final_response}


    import json

    async def mongo_query(self, state):
        query = state.get("input")
        if not query:
            raise ValueError("query not provided in the state")

        print("DEBUG STATE in mongo_query:", state)

        # Inject user query into the MONGO_PROMPT
        prompt = AgentConfig.MONGO_PROMPT.replace("{user_query}", query)

        # Prepare messages
        messages = [
            {"role": "system", "content": AgentConfig.MONGO_SYSTEM_PROMPT if hasattr(AgentConfig, "MONGO_SYSTEM_PROMPT") else "You are an expert MongoDB query generator."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 400,
            "stream": False
        }

        try:
            # Run blocking requests.post in a background thread
            response = await asyncio.to_thread(requests.post, self.url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract LLM output
            response_text = data["choices"][0]["message"]["content"].strip()
            print("MongoDB query generated:")
            print(response_text)

            try:
                # Parse the JSON to get query_filter
                response_json = json.loads(response_text)
                query_filter = response_json.get("query_filter", {})
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from LLM response: {e}")

            # Access MongoDB (Motor async client)
            db = mongo_client[database_name]
            collection = db[collection_name]
            result = await collection.find(query_filter).to_list(length=100)

            # Format readable response
            if not result:
                response_text = "No data found for your query."
            else:
                formatted = []
                for doc in result:
                    for block in doc.get("chat_history", []):
                        for qa in block.get("chat_history", []):
                            q = qa.get("question", "")
                            a = qa.get("answer", "")
                            formatted.append(f"Q: {q}\nA: {a}")
                response_text = "\n\n".join(formatted)

            return {"response": response_text}

        except Exception as e:
            return {"response": f"Error in mongo_query: {e}"}


    def unknown_intent_handler(self, state):
        print("Handling unknown intent...")
        return {"response": "Sorry, I didn't understand that request."}

    def _build_graph(self):
        builder = StateGraph(dict)

        # Define nodes
        builder.add_node("route_agent", self.route_agent)
        builder.add_node("simple_llm", self.simple_llm)
        builder.add_node("RAG", self.RAG)
        builder.add_node("mongo_query", self.mongo_query)
        builder.add_node("general_agent", self.general_agent)
        builder.add_node("unknown_intent_handler", self.unknown_intent_handler)
        #builder.add_node("purpose_classifier", self.purpose_classifier)

        # Entry point
        builder.set_entry_point("route_agent")

        # Conditional routing based on agent decision
        def agent_selector(state):
            return state.get("agent", "unknown_intent_handler")

        builder.add_conditional_edges("route_agent", agent_selector, {
            "CONVERSATION_AGENT": "simple_llm",
            "RAG_AGENT": "RAG",
            "MONGO_QUERY": "mongo_query",
            "GENERAL_AGENT": "general_agent",
            "WEB_SEARCH_PROCESSOR_AGENT": "RAG",  # Or a separate web agent if available
            "low_confidence_fallback_agent": "unknown_intent_handler",
            "parse_error_fallback_agent": "unknown_intent_handler",
            "unknown_intent_handler": "unknown_intent_handler",
        })

        return builder.compile()
    
    def move_to_next_question(self, state):
        keys = [k for k, _ in self.questions]
        if state["current_key"] in keys:
            idx = keys.index(state["current_key"])
            if idx + 1 < len(keys):
                next_key, next_question = self.questions[idx + 1]
                state["current_key"] = next_key
                state["current_question"] = next_question
            else:
                # No more questions
                state["current_key"] = None
                state["current_question"] = None
        return state

    async def run(self, input_text=None, has_image=False, session_id=None):
        # Initialize session manager if not already done
        if not hasattr(self, "session_manager"):
            self.session_manager = SessionManager()

        # Create a new session if none provided
        if session_id is None or not self.session_manager.has_session(session_id):
            session_id = self.session_manager.create_session(questions=self.questions)

        # Load session state
        state = self.session_manager.get_state(session_id)
        answers = state.get("answers", {})
        state["answers"] = answers
        state["session_id"] = session_id

        # Handle initial welcome interaction
        if not state.get("welcomed", False):
            if input_text == "start123":
                return {
                    "status": "welcome",
                    "message": f"{AgentConfig.WELCOME_MESSAGE}",
                    "session_id": session_id
                }

            intent = self.classify_intent(input_text, AgentConfig.WELCOME_MESSAGE)
            state["negative_count"] = state.get("negative_count", 0)

            if intent == "denial":
                state["negative_count"] += 1
                if state["negative_count"] >= 2:
                    return {
                        "status": "ended",
                        "message": "I understand. Thank you for taking the time to talk to me. Goodbye",
                        "session_id": session_id
                    }
                explanation = await self.purpose_classifier(
                    input_text,
                    "May I ask you a few questions to better assist you?"
                )
                return {
                    "status": "denial",
                    "message": explanation + "\nCould you please reconsider starting with a few questions?",
                    "session_id": session_id
                }

            if intent == "query":
                email = answers.get("email", "default@example.com")
                self.session_manager.update_state(session_id, state)
                response = await self.graph.ainvoke({
                    "input": input_text,
                    "has_image": has_image,
                    "email": email
                })
                response_text = response.get("response", "")
                return {
                    "status": "questioning",
                    "message": response_text + "\nCan we start with the survey now?\n",
                    "session_id": session_id
                }

            if intent == "greeting":
                self.session_manager.update_state(session_id, state)
                response_text = self.greeting(input_text, chat_log)
                return {
                    "status": "questioning",
                    "message": response_text + " Should we start with the survey?",
                    "session_id": session_id
                }

            if intent == "answer":
                state["welcomed"] = True
                self.session_manager.update_state(session_id, state)
                response = await self.interact(state)
                response["session_id"] = session_id
                return response

            return {
                "status": "unclear_input",
                "message": "I couldn't understand that. Would you like to begin with a few questions to help guide our conversation?",
                "session_id": session_id
            }

        # Main survey flow
        current_key = state.get("current_key")
        current_question = state.get("current_question")

        if current_key is not None:
            print("Current question:", current_question)

            if input_text:
                intent = self.classify_intent(input_text, current_question, chat_log)

                if intent == "answer":
                    answer_validity = self.classify_answer(input_text, current_question)
                    print("answer validity:", answer_validity)
                    if answer_validity == "satisfactory":
                        # Save answer
                        answers[current_key] = input_text
                        state["answers"] = answers

                        # Move to next question
                        state = self.move_to_next_question(state)
                        self.session_manager.update_state(session_id, state)

                        next_question = state.get("current_question")
                        if next_question:
                            return {
                                "status": "awaiting_input",
                                "message": f"Next question:\n{next_question}",
                                "session_id": session_id
                            }
                        else:
                            return {
                                "status": "completed",
                                "message": "Thank you for completing the survey!",
                                "session_id": session_id
                            }

                    else:
                        # Repeat same question
                        self.session_manager.update_state(session_id, state)
                        return {
                            "status": "repeat",
                            "message": f"{answer_validity} Ok, let's try that again:\n{current_question}",
                            "session_id": session_id
                        }

                elif intent == "greeting":
                    self.session_manager.update_state(session_id, state)
                    response_text = self.greeting(input_text, chat_log)
                    return {
                        "status": "questioning",
                        "message": f"{response_text} \n  + {current_question}",
                        "session_id": session_id
                    }

                elif intent == "repeat":
                    return {
                        "status": "questioning",
                        "message": f"Here is the question again:\n{current_question}",
                        "session_id": session_id
                    }

                elif intent == "denial":
                    state["negative_count"] += 1
                    if state["negative_count"] >= 2:
                        return {
                            "status": "ended",
                            "message": "I understand. Thank you for taking the time to talk to me. Goodbye",
                            "session_id": session_id
                        }
                    explanation = await self.purpose_classifier(
                        input_text,
                        "May I ask you a few questions to better assist you?"
                    )
                    return {
                        "status": "denial",
                        "message": explanation + "\nCould you please reconsider answering this question?",
                        "session_id": session_id
                    }

                elif intent == "query":
                    email = answers.get("email", "default@example.com")
                    self.session_manager.update_state(session_id, state)
                    response = await self.graph.ainvoke({
                        "input": input_text,
                        "has_image": has_image,
                        "email": email,
                        "session_id": session_id
                    })
                    response_text = response.get("response", "")
                    return {
                        "status": "questioning",
                        "message": response_text + "\nPlease reconsider answering the current question:\n" + current_question,
                        "session_id": session_id
                    }

            # Awaiting input
            return {
                "status": "awaiting_input",
                "message": f"Could you please answer the following question?\n{current_question}",
                "session_id": session_id
            }



chat_log = {}


def log_message(role: str, message: str, session_id: str):
    if session_id not in chat_log:
        chat_log[session_id] = []
    chat_log[session_id].append({
        "role": role,
        "message": message
    })

def print_bot_message(message: str, session_id: str):
    print(f"Bot: {message}")
    log_message("bot", message, session_id)


def save_chat_log(filepath="chat_log.json"):
    with open(filepath, "w") as f:
        json.dump(chat_log, f, indent=2)
    print(f"[INFO] Chat log saved to {filepath}")


async def chat_loop():
    router = IntentRouter()
    session_id = None

    # Start conversation
    response = await router.run(input_text="start123", session_id=session_id)
    session_id = response.get("session_id", session_id)

    # Print initial bot message
    bot_message = response.get("message") or response.get("question", "")
    if bot_message:
        print_bot_message(bot_message, session_id)

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print("\n[INFO] Input closed. Exiting chat.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        log_message("user", user_input, session_id)

        # Run router
        response = await router.run(user_input, session_id=session_id)
        session_id = response.get("session_id", session_id)
        status = response.get("status")

        # Generalized message handling
        bot_message = response.get("message") or response.get("question") or response.get("response") or "Something went wrong."
        if bot_message:
            print_bot_message(bot_message, session_id)

        # Handle chat termination
        if status == "ended":
            break

        # If status is 'repeat', ensure the same question is shown again
        if status == "repeat":
            continue

        # If awaiting input or questioning, loop naturally for next user input
        # All other statuses are handled by printing the message above

    save_chat_log()

if __name__ == "__main__":
    asyncio.run(chat_loop())