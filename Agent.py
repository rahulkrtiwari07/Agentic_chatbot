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
from Prompts import AgentConfig

from mongooperations import get_mongo_client, store_user_data
from extract_mongo import retrieve_data

from stream import Retrieval  # Make sure this exists and is implemented

from uuid import uuid4
from dotenv import load_dotenv

from fastapi.responses import StreamingResponse

mongo_uri = os.getenv('MONGO_URL')
database_name = "chat_db"
collection_name = "user_data"
collection_name_1 = "user_data_1"

mongo_client = get_mongo_client(mongo_uri)
print(mongo_client)

load_dotenv('.env.example')

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self):
        session_id = str(uuid4())
        self.sessions[session_id] = {"answers": {}, "current_key": None}
        return session_id

    def get_state(self, session_id):
        return self.sessions.get(session_id, {"answers": {}, "current_key": None, "welcomed": False})

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
        self.retrieval = Retrieval(self.es_pass)
        self.retriever = self.retrieval.retrieve_data()

        self.questions = [
            ("name", "Could you please state your full name?"),
            ("age", "What is your age?"),
            ("address", "What is your address?"),
            ("Confirm address", "Is this your permanent address?"),
            ("Permanent address", "Please let meknow your permanent address?"),
            ("covid", "Have you have covid in the past 5 years?"),
            ("Year", "In which year did you last have covid?"),
            ("vaccinated","Were you vaccinated at that time?"),
            ("No covid", "Have you ever shown symptons of covid?"),
            ("Vaccination before", "Have you ever been vaccinated for Covid?"),
            ("email", "What is your email address?"),
            ("Thanks", "Thank you! Feel free to ask if you have any questions.")
        ]


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
        self.url = "http://164.52.193.73:9200/v1/chat/completions"

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
        input_text = state.get("input", "")

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
            response = requests.post(self.url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]

            # ---------- FIX 1: Properly strip markdown fences ----------
            cleaned = re.sub(
                r'^```(?:json)?\s*|\s*```$',
                '',
                content.strip(),
                flags=re.IGNORECASE | re.MULTILINE
            ).strip()

            # ---------- FIX 2: Parse JSON safely ----------
            try:
                decision = json.loads(cleaned)
            except json.JSONDecodeError:
                # Attempt newline repair ONLY if parsing fails
                fixed = re.sub(r'(?<!\\)\n', ' ', cleaned)
                try:
                    decision = json.loads(fixed)
                except json.JSONDecodeError:
                    logging.warning(
                        f"Route agent JSON parse failed. Raw content: {content}"
                    )
                    decision = {
                        "agent": "parse_error_fallback_agent",
                        "reasoning": f"Invalid JSON from LLM: {cleaned}",
                        "confidence": 0.0
                    }

            # ---------- FIX 3: Validate agent name ----------
            valid_agents = {
                "CONVERSATION_AGENT",
                "RAG_AGENT",
                "MONGO_QUERY",
                "GENERAL_AGENT",
                "WEB_SEARCH_PROCESSOR_AGENT",
            }

            if decision.get("agent") not in valid_agents:
                decision["agent"] = "unknown_intent_handler"

            # ---------- FIX 4: Confidence gating ----------
            if decision.get("confidence", 0) < AgentConfig.CONFIDENCE_THRESHOLD:
                decision["agent"] = "low_confidence_fallback_agent"

        except Exception as e:
            logging.warning(f"Route agent error: {e}")
            decision = {
                "agent": "parse_error_fallback_agent",
                "reasoning": f"Request failed: {e}",
                "confidence": 0.0
            }

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
    


    def information_extract(self, chat_history):
        messages = [
            {"role": "system", "content": AgentConfig.INFORMATION_EXTRACT_PROMPT.format(chat_history=chat_history)}
        ]

        payload = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 500
        }

        try:
            start_time = time.time()
            response = requests.post(self.url, json=payload)
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            print("[DEBUG] Extracted information raw output:", response.text)

            response.raise_for_status()
            data = response.json()

            # Extract assistant content
            content = data["choices"][0]["message"]["content"]

            # Try to parse JSON block
            extracted_info = {}
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    extracted_info = json.loads(match.group(0))
                except Exception as e:
                    logging.warning(f"JSON parse failed, fallback to raw text: {e}")
                    extracted_info = {"raw_text": content}
            else:
                # No JSON found, just return raw text
                extracted_info = {"raw_text": content}

            return extracted_info

        except Exception as e:
            logging.warning(f"Information extraction error: {e}")
            return {}



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

    async def run(self, input_text=None, has_image=False, session_id=None):
        # Initialize session manager if not already done
        if not hasattr(self, "session_manager"):
            self.session_manager = SessionManager()

        # Create a new session if none provided
        if session_id is None or not self.session_manager.has_session(session_id):
            session_id = self.session_manager.create_session()

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

            # Classify intent of reply to welcome message
            intent = self.classify_intent(input_text, AgentConfig.WELCOME_MESSAGE)
            
            state["negative_count"] = state.get("negative_count", 0)

            if intent in ("denial"):
                state["negative_count"] += 1

                if state["negative_count"] >= 2:
                
                    return {
                        "status": "ended",
                        "message": "I understand. Thank you for taking the time to talk to me. Goodbye",
                        "session_id": session_id
                    }
                explanation = await self.purpose_classifier(input_text, "May I ask you a few questions to better assist you?")
                return {
                    "status": "denial",
                    "message": explanation + "\nCould you please reconsider starting with a few questions?",
                    "session_id": session_id
                }
            
            if intent in ("query"):
                email = answers.get("email", "default@example.com")
                self.session_manager.update_state(session_id, state)
                current_question = dict(self.questions).get("name", "")
                response = await self.graph.ainvoke({
                    "input": input_text,
                    "has_image": has_image,
                    "email": email
                })
                response_text = response.get("response", "")
                response["session_id"] = session_id
                return {
                    "status": "questioning",
                    "message": response_text + "\nCan we start with the survey now?\n",
                    "session_id": session_id
                }
            
            if intent in ("greeting"):
                self.session_manager.update_state(session_id, state)
                response_text = self.greeting(input_text, chat_log)
                return {
                    "status": "questioning",
                    "message": response_text + "should we start with the survey?",
                    "session_id": session_id
                }

            if intent in ("answer"):
                state["welcomed"] = True
                self.session_manager.update_state(session_id, state)
                response = await self.interact(state)
                response["session_id"] = session_id
                return response
            else:
                return {
                    "status": "unclear_input",
                    "message": "I couldn't understand that. Would you like to begin with a few questions to help guide our conversation?",
                    "session_id": session_id
                }


        # Handle personal information collection
        if len(answers) < len(self.questions):
            if input_text:
                current_key = state.get("current_key")
                print(current_key)
                current_question = dict(self.questions).get(current_key, "")
                intent = self.classify_intent(input_text, current_question, chat_log)

                if intent == "answer" and current_key in ["name", "age", "address", "email", "Permanent address"] :
                    intent1 = self.classify_answer(input_text, current_question)
                    #print("intent1 output:", intent1)
                    if intent1 == "satisfactory":
                        answers[current_key] = input_text
                        state["answers"] = answers
                        self.session_manager.update_state(session_id, state)

                        chat_entry = {
                            "question": current_question,
                            "answer": input_text
                        }
                        #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)

                        if current_key == "email":
                            state["current_key"] = "Thanks"
                            self.session_manager.update_state(session_id, state)
                            next_question = dict(self.questions).get("Thanks", "")
                            return {
                                "status": "ready_for_questions",
                                "message": next_question,
                                "session_id": session_id
                            }

                        response = await self.interact(state)
                        response["session_id"] = session_id
                        return response
                    else :
                        clarification = intent1
                        return {
                                "status": "ready_for_questions",
                                "message": intent1,
                                "session_id": session_id
                            }
                
                if intent in ("greeting"):
                    self.session_manager.update_state(session_id, state)
                    response_text = self.greeting(input_text, chat_log)
                    return {
                        "status": "Queries",
                        "message": response_text,
                        "session_id": session_id
                    }

                if current_key == "Confirm address":
                    intent1 = self.classify_answer(input_text, current_question) 
                    print(intent1)
                    if intent1 == "acceptance":
                        state["current_key"] = "covid"
                        self.session_manager.update_state(session_id, state)
                        next_question = dict(self.questions).get("covid", "")
                        chat_entry = {
                                "question": current_question,
                                "answer": input_text
                            }
                        #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                        return {
                            "status": "correct address",
                            "message": "Thanks for the clarification.\n" + next_question,
                            "session_id": session_id
                        }
                    
                    if intent1 == "denial":
                        state["current_key"] = "Permanent address"
                        self.session_manager.update_state(session_id, state)
                        next_question = dict(self.questions).get("Permanent address", "")
                        chat_entry = {
                                "question": current_question,
                                "answer": input_text
                            }
                        #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                        return {
                            "status": "correct address",
                            "message": "Thanks for the clarification.\n" + next_question,
                            "session_id": session_id
                        }

                if current_key == "covid":
                    intent1 = self.classify_answer(input_text, current_question) 
                    print(intent1)
                    if intent1 == "acceptance":
                        state["current_key"] = "Year"
                        self.session_manager.update_state(session_id, state)
                        next_question = dict(self.questions).get("Year", "")
                        chat_entry = {
                                "question": current_question,
                                "answer": input_text
                            }
                        #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                        return {
                            "status": "correct address",
                            "message": "Thanks for the clarification.\n" + next_question,
                            "session_id": session_id
                        }
                    
                    if intent1 == "denial":
                        state["current_key"] = "No covid"
                        self.session_manager.update_state(session_id, state)
                        next_question = dict(self.questions).get("No covid", "")
                        chat_entry = {
                                "question": current_question,
                                "answer": input_text
                            }
                        #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                        return {
                            "status": "correct address",
                            "message": "Thanks for the clarification.\n" + next_question,
                            "session_id": session_id
                        }

                if current_key == "Year" and intent == "answer":
                    state["current_key"] = "vaccinated"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("vaccinated", "")
                    chat_entry = {
                            "question": current_question,
                            "answer": input_text
                        }
                    #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                    return {
                        "status": "Covid positive",
                        "message": next_question,
                        "session_id": session_id
                    }

                if current_key == "vaccinated" and intent == "answer":
                    state["current_key"] = "email"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("email", "")
                    chat_entry = {
                            "question": current_question,
                            "answer": input_text
                        }
                    #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                    return {
                        "status": "Covid positive",
                        "message": next_question,
                        "session_id": session_id
                    }
                
                if current_key == "No covid" and intent == "answer":
                    state["current_key"] = "Vaccination before"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("Vaccination before", "")
                    chat_entry = {
                            "question": current_question,
                            "answer": input_text
                        }
                    #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                    return {
                        "status": "Covid negative",
                        "message": next_question,
                        "session_id": session_id
                    }

                if current_key == "Vaccination before" and intent == "answer":
                    state["current_key"] = "email"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("email", "")
                    chat_entry = {
                            "question": current_question,
                            "answer": input_text
                        }
                    #store_user_data(mongo_client, database_name, collection_name, session_id, chat_entry)
                    return {
                        "status": "Covid negative",
                        "message": next_question,
                        "session_id": session_id
                    }

                if intent == "denial" and current_key not in ["Thanks"]:
                    explanation = await self.purpose_classifier(input_text, current_question)
                    return {
                        "status": "denial",
                        "message": explanation + "\nCould you please reconsider answering this question?\n" + current_question,
                        "session_id": session_id
                    }

                if intent == "repeat":
                    return {
                        "status": "repeat",
                        "message": f"Ok, I understand, here is the question again:\n{current_question}",
                        "session_id": session_id
                    }
                
                if intent in ("denial"):
                    state["negative_count"] += 1

                    if state["negative_count"] >= 2:
                    
                        return {
                            "status": "ended",
                            "message": "I understand. Thank you for taking the time to talk to me. Goodbye",
                            "session_id": session_id
                        }
                    explanation = await self.purpose_classifier(input_text, "May I ask you a few questions to better assist you?")
                    return {
                        "status": "denial",
                        "message": explanation + "\nCould you please reconsider starting with a few questions?",
                        "session_id": session_id
                    }

                if current_key == "Thanks" and intent == "denial":
                    '''data = retrieve_data(mongo_client, database_name, session_id)

                   # Safely extract chat_history
                    try:
                        chat = data['user_data'][0]['chat_history']
                    except (KeyError, IndexError) as e:
                        logging.warning(f"Failed to retrieve chat_history: {e}")
                        chat = []

                    # Proceed only if chat is not empty
                    if chat:
                        extract = self.information_extract(chat)
                        #print(extract)
                    else:
                        print("No chat history found for this session.")

                    chat_entry = {"chat_history": extract
                            } 
                    store_user_data(mongo_client, database_name, collection_name_1, session_id, chat_entry) '''

                    return {
                        "status": "ended",
                        "message": "No problem. Feel free to return anytime. Goodbye!",
                        "session_id": session_id
                    }

                
                if current_key == "Thanks" and intent == "acceptance":
                    state["current_key"] = "Query"
                    return{
                        "status": "Query",
                        "message": "Please ask your questions",
                        "session_id": session_id
                    }
                
                if current_key == "Thanks" and intent == "query":
                    state["current_key"] = "Query"
                    email = answers.get("email", "default@example.com")
                    self.session_manager.update_state(session_id, state)
                    response = await self.graph.ainvoke({
                        "input": input_text,
                        "has_image": has_image,
                        "email": email
                    })
                    response_text = response.get("response", "")

                    response["session_id"] = session_id
                    return {
                        "status": "Queries",
                        "message": response_text + "\n Any more questions?",
                        "session_id": session_id
                    }
                
                '''if current_key == "Query" and intent == "query":
                    state["current_key"] = "Query"
                    email = answers.get("email", "default@example.com")
                    self.session_manager.update_state(session_id, state)
                    response = await self.graph.ainvoke({
                        "input": input_text,
                        "has_image": has_image,
                        "email": email
                    })
                    response_text = response.get("response", "")
                    response["session_id"] = session_id
                    return {
                        "status": "Queries",
                        "message": response_text + "\n Any more questions?",
                        "session_id": session_id
                    }'''

                '''if current_key == "Query" and intent == "negative":
                    return {
                        "status": "ended",
                        "message": "No problem. Feel free to return anytime. Goodbye!",
                        "session_id": session_id
                    }'''

                if intent == "query":
                    email = answers.get("email", "default@example.com")
                    self.session_manager.update_state(session_id, state)
                    response = await self.graph.ainvoke({
                        "input": input_text,
                        "has_image": has_image,
                        "email": email, 
                        "session_id": session_id
                    })
                    response_text = response.get("response", "")
                    response["session_id"] = session_id
                    return {
                        "status": "Queries",
                        "message": response_text + "\nCould you please reconsider answering this question?\n" + current_question,
                        "session_id": session_id
                    }



            # If no input provided, or couldn't interpret intent
            return {
                "status": "awaiting_input",
                "message": f"Could you please answer the following question?\n{dict(self.questions).get(state.get('current_key'), '')}",
                "session_id": session_id
            }

        # All personal questions completed
        if input_text:
            # Route to appropriate agent
            updated_state = {
                "input": input_text,
                "answers": answers,
                "email": answers.get("email", "default@example.com")
            }
            async for output in self.graph.stream(updated_state):
                return {**output, "session_id": session_id}

        return {
            "status": "awaiting_query",
            "message": "Do you have any questions you'd like to ask?",
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

    if response.get("status") == "welcome":
        print_bot_message(response["message"], session_id)
    elif response.get("status") == "asking":
        print_bot_message(response["question"], session_id)

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

        response = await router.run(user_input, session_id=session_id)
        session_id = response.get("session_id", session_id)

        status = response.get("status")
        bot_message = ""

        if status in {"welcome", "asking", "denial", "unclear_input", "ready_for_questions", "confirm_permanent_address"}:
            bot_message = response.get("message") or response.get("question", "")
            print_bot_message(bot_message, session_id)
        elif status == "ended":
            bot_message = "That concludes my survey. Thank you for taking my call. Goodbye"
            print_bot_message(bot_message, session_id)
            break
        elif response.get("response"):
            bot_message = response["response"]
            print_bot_message(bot_message, session_id)
        else:
            bot_message = response.get("message", "Something went wrong.")
            print_bot_message(bot_message, session_id)

    save_chat_log()

if __name__ == "__main__":
    asyncio.run(chat_loop())