import os
import logging
import asyncio
import json

from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from Prompts import AgentConfig

from stream import Retrieval  # Make sure this exists and is implemented

from uuid import uuid4

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
        self.retrieval = Retrieval(es_pass=self.es_pass)
        self.retriever = self.retrieval.retrieve_data()

        self.questions = [
            ("name", "Could you please state your full name?"),
            ("age", "What is your age?"),
            ("address", "Can you please provide your address?"),
            ("Confirm address", "Is this your permanent address"),
            ("Permanent address", "Please let meknow your permanent address"),
            ("covid", "Have you have covid in the past 5 years?"),
            ("Year", "In which year did you last have covid"),
            ("vaccinated","Were you avccinated at that time"),
            ("No covid", "Have you ever shown symptons of covid"),
            ("Vaccination before", "Have you ever been vaccinated for Covid?"),
            ("email", "What is your email address?"),
            ("Thanks", "Thank you! Feel free to ask if you have any questions.")
        ]


        self.log_path = "interaction.docx"

        # LLMs
        self.llm1 = AzureChatOpenAI(
            azure_deployment=self.azure_deployment,
            api_version="2023-06-01-preview",
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            temperature=0,
            streaming=False,
            max_retries=2,
        )

        self.llm2 = AzureChatOpenAI(
            azure_deployment=self.azure_deployment,
            api_version="2023-06-01-preview",
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            temperature=0.9,
            streaming=True,
            max_retries=2,
        )

        # Build graph
        self.graph = self._build_graph()

    def route_agent(self, state):
        input_text = state["input"]
        messages = [
            {"role": "system", "content": AgentConfig.DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": f"User query: {input_text}"}
        ]
        try:
            result = self.llm1.invoke(messages)
            decision = json.loads(result.content)
            if decision.get("confidence", 0) < AgentConfig.CONFIDENCE_THRESHOLD:
                decision["agent"] = "low_confidence_fallback_agent"
        except Exception as e:
            decision = {
                "agent": "parse_error_fallback_agent",
                "reasoning": f"Parsing failed: {e}",
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        try:
            result = self.llm1.invoke(messages)
            print("[DEBUG] Intent classification raw output:", result.content)
            intent = json.loads(result.content).get("intent", "unknown")
        except Exception as e:
            logging.warning(f"Intent classification error: {e}")
            intent = "unknown"

        return intent

    
    def classify_answer(self, user_input, question=None):
        messages = [
            {"role": "system", "content": AgentConfig.ANSWER_PROMPT },
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        try:
            result = self.llm2.invoke(messages)
            print("[DEBUG] Intent classification raw output:", result.content)
            intent = json.loads(result.content).get("intent", "unknown")
        except Exception as e:
            logging.warning(f"Intent classification error: {e}")
            intent = "unknown"

        return intent

    def clarification(self, user_input, question=None):
        messages = [
            {"role": "system", "content": AgentConfig.CLARIFICATION_PROMPT },
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        try:
            result = self.llm2.invoke(messages)
            print("[DEBUG] Intent classification raw output:", result.content)
            intent = json.loads(result.content).get("intent", "unknown")
        except Exception as e:
            logging.warning(f"Intent classification error: {e}")
            intent = "unknown"

        return intent

    

    async def purpose_classifier(self, user_input, question):
        print("purpose_classifier")
        messages = [
            {"role": "system", "content": AgentConfig.PURPOSE_CLASSIFIER_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        try: 
            self.llm1.temperature = 0.7
            response = self.llm1.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    async def serious_classifier(self, user_input, question):
        print("serious_classifier")
        messages = [
            {"role": "system", "content": AgentConfig.SERIOUS_ClASSIFIER_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        try: 
            self.llm1.temperature = 0.7
            response = self.llm1.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"


        

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

        
    async def simple_llm(self, state):
        full_response = ""
        try:
            logging.info("simple_llm started")
            print("simple_llm")
            # Use a prompt from the state or default to a basic instruction
            system_prompt = state.get("system_prompt", AgentConfig.QUERY_PROMPT)
            user_input = state['input']

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]

            async for chunk in self.llm2.astream(
                messages,
                config={"configurable": {"session_id": state.get('email', 'default_session')}},
            ):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield {"response": full_response}
                    
        except Exception as e:
            logging.exception("Error during streaming")
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


    def unknown_intent_handler(self, state):
        print("Handling unknown intent...")
        return {"response": "Sorry, I didn't understand that request."}

    def _build_graph(self):
        builder = StateGraph(dict)

        # Define nodes
        builder.add_node("route_agent", self.route_agent)
        builder.add_node("simple_llm", self.simple_llm)
        builder.add_node("RAG", self.RAG)
        builder.add_node("unknown_intent_handler", self.unknown_intent_handler)
        builder.add_node("purpose_classifier", self.purpose_classifier)

        # Entry point
        builder.set_entry_point("route_agent")

        # Conditional routing based on agent decision
        def agent_selector(state):
            return state.get("agent", "unknown_intent_handler")

        builder.add_conditional_edges("route_agent", agent_selector, {
            "CONVERSATION_AGENT": "simple_llm",
            "RAG_AGENT": "RAG",
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

        # Handle initial welcome interaction
        if not state.get("welcomed", False):
            if not input_text:
                return {
                    "status": "welcome",
                    "message": f"Welcome! {AgentConfig.WELCOME_MESSAGE}",
                    "session_id": session_id
                }

            # Classify intent of reply to welcome message
            intent = self.classify_intent(input_text, AgentConfig.WELCOME_MESSAGE)
            

            if intent == "denial":
                explanation = await self.purpose_classifier(input_text, "May I ask you a few questions to better assist you?")
                return {
                    "status": "denial",
                    "message": explanation + "\nCould you please reconsider starting with a few questions?",
                    "session_id": session_id
                }
            elif intent in ("answer", "query", "acceptance"):
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

                if intent == "clarification":
                    clarification = self.clarification(input_text, current_question)
                    return {
                                "status": "clarification",
                                "message": clarification + "Please answer the question accordingly \n" + current_question,
                                "session_id": session_id
                            }

                if intent == "answer" and current_key in ["name", "age", "address", "email"] :
                    intent1 = self.classify_answer(input_text, current_question)
                    print(intent1)
                    if intent1 == "satisfactory":
                        answers[current_key] = input_text
                        state["answers"] = answers
                        self.session_manager.update_state(session_id, state)

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
                        return {
                                "status": "ready_for_questions",
                                "message": intent1,
                                "session_id": session_id
                            }

                if current_key == "address" and intent=="answer":
                    state["current_key"] = "Confirm address"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get(current_key, "")
                    return {
                        "status": "Permanent_address",
                        "message": next_question,
                        "session_id": session_id
                    } 

                if current_key == "Confirm address" and intent == "acceptance":
                    state["current_key"] = "covid"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("covid", "")
                    return {
                        "status": "correct address",
                        "message": "Thanks for the clarification.\n" + next_question,
                        "session_id": session_id
                    }

                if current_key == "Confirm address" and intent == "Different address":
                    state["current_key"] = "Permanent address"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("Permanent address", "")
                    return {
                        "status": "correct address",
                        "message": "Thanks for the clarification.\n" + next_question,
                        "session_id": session_id
                    }

                if current_key == "Permanent address" and intent == "answer":
                    state["current_key"] = "covid"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("covid", "")
                    return {
                        "status": "correct address",
                        "message": "Thanks for the clarification.\n" + next_question,
                        "session_id": session_id
                    }

                if current_key == "covid" and intent == "acceptance":
                    state["current_key"] = "Year"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("Year", "")
                    return {
                        "status": "Covid positive",
                        "message": next_question,
                        "session_id": session_id
                    }

                if current_key == "Year" and intent == "answer":
                    state["current_key"] = "vaccinated"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("vaccinated", "")
                    return {
                        "status": "Covid positive",
                        "message": next_question,
                        "session_id": session_id
                    }

                if current_key == "vaccinated" and intent == "acceptance":
                    state["current_key"] = "email"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("email", "")
                    return {
                        "status": "Covid positive",
                        "message": next_question,
                        "session_id": session_id
                    }

                if current_key == "vaccinated" and intent == "no vaccine":
                    state["current_key"] = "email"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("email", "")
                    return {
                        "status": "Covid positive",
                        "message": next_question,
                        "session_id": session_id
                    }

                if current_key == "covid" and intent == "no covid":
                    state["current_key"] = "No covid"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("No covid", "")
                    return {
                        "status": "Covid negative",
                        "message": next_question,
                        "session_id": session_id
                    }

                if current_key == "No covid":
                    state["current_key"] = "email"
                    self.session_manager.update_state(session_id, state)
                    next_question = dict(self.questions).get("email", "")
                    return {
                        "status": "Covid negative",
                        "message": next_question,
                        "session_id": session_id
                    }


                if intent == "denial":
                    explanation = await self.purpose_classifier(input_text, current_question)
                    return {
                        "status": "denial",
                        "message": explanation + "\nCould you please reconsider answering this question?\n" + current_question,
                        "session_id": session_id
                    }

                if intent == "repeat":
                    return {
                        "status": "repeat",
                        "message": f"Sure, here is the question again:\n{current_question}",
                        "session_id": session_id
                    }

                if current_key == "Thanks" and intent == "negative":
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
                    response_text = response.get("response", "")
                    response = await self.graph.ainvoke({
                        "input": input_text,
                        "has_image": has_image,
                        "email": email
                    })
                    response["session_id"] = session_id
                    return {
                        "status": "Queries",
                        "message": response_text + "\n Any more questions?",
                        "session_id": session_id
                    }
                
                if current_key == "Query" and intent == "query":
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

                if current_key == "Query" and intent == "negative":
                    return {
                        "status": "ended",
                        "message": "No problem. Feel free to return anytime. Goodbye!",
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
                    response["session_id"] = session_id
                    return response


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
    response = await router.run(input_text=None, session_id=session_id)
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
            bot_message = "The session has ended. Thank you!"
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