import os
import logging
import asyncio
import json

from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage

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
        return self.sessions.get(session_id, {"answers": {}, "current_key": None})

    def update_state(self, session_id, state):
        self.sessions[session_id] = state

    def has_session(self, session_id):
        return session_id in self.sessions


class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gpt-35-turbo"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-35-turbo"
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and any questions **not** falling into the other categories, including general questions about India and its districts that are not related to health or population.
    2. RAG_AGENT - For specific knowledge about **population dynamics and health indicators in India and its districts**, including:
   - Fertility
   - Infant and child mortality
   - Family planning practices
   - Maternal and child health
   - Reproductive health
   - Nutrition
   - Emerging health and family welfare issues 

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    INTENT_CLASSIFIER_PROMPT = """You are an intent classification assistant.

    Determine whether the user’s message is:

    1. An **answer** to a personal information question (like name, age, or email).
    Examples of answers:
    - "Rahul"
    - "25"
    - "rahul@example.com"
    - "My name is Priya"
    - "I’m 30 years old"

    2. Or a **query** asking for general or medical information, or starting a new conversation.
    Examples of queries:
    - "Tell me about population growth"
    - "What is the weather?"
    - "Hi, how are you?"

    3. **denial** — the user refuses to provide information or objects to answering.
    Examples:
    - "I don't want to tell you that"
    - "Why do you need my email?"
    - "I prefer not to share my age"
    - "That's personal"
    - "None of your business"

    Respond ONLY in the following JSON format:
    {
    "intent": "answer" | "query" | "denial"
    }
    """

    PURPOSE_CLASSIFIER_PROMPT = """You are a purpose classifier assistant.
    
    The user tries to object to the question being asked to them for data collection and ypur sole purpose is to make them understand about the purpose of data collection.
    The purpose of data collection is to store user data in to database so that it can help in better governance to the government."""

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
            ("name", "What is your name?"),
            ("age", "What is your age?"),
            ("email", "What is your email address?")
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
            temperature=0,
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
    
    def classify_intent(self, user_input, question=None):
        messages = [
            {"role": "system", "content": AgentConfig.INTENT_CLASSIFIER_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: {user_input}" if question else user_input}
        ]

        try:
            result = self.llm1.invoke(messages)
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
            print("simple_llm")
            async for chunk in self.llm2.astream(
                [HumanMessage(content=state['input'])],
                config={"configurable": {"session_id": state.get('email', 'default_session')}},
            ):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield {"response": full_response}
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
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

        # Handle personal information questions
        if len(answers) < len(self.questions):
            if input_text:
                current_key = state.get("current_key")
                current_question = dict(self.questions).get(current_key, "")

                intent = self.classify_intent(input_text, current_question)

                if intent == "answer":
                    if current_key:
                        answers[current_key] = input_text
                        state["answers"] = answers
                    self.session_manager.update_state(session_id, state)

                    if current_key == "email":
                        return {
                            "status": "ready_for_questions",
                            "message": "Thank you! Would you like to ask any questions now?",
                            "session_id": session_id
                        }
                    response = await self.interact(state)
                    response["session_id"] = session_id
                    return response

                elif intent == "query":
                    email = answers.get("email", "default@example.com")
                    self.session_manager.update_state(session_id, state)
                    response = await self.graph.ainvoke({
                        "input": input_text,
                        "has_image": has_image,
                        "email": email
                    })
                    response["session_id"] = session_id
                    return response

                elif intent == "denial":
                    print("denial")
                    current_question = dict(self.questions).get(state.get("current_key"), "")
                    explanation = await self.purpose_classifier(input_text, current_question)

                    state["status"] = "denial"
                    self.session_manager.update_state(session_id, state)

                    return {
                        "status": "denial",
                        "message": explanation + "\nCould you please reconsider answering the following?",
                        "question": current_question,
                        "current_key": state.get("current_key"),
                        "answers": answers,
                        "session_id": session_id
                    }

                else:
                    return {
                        "status": "unclear_input",
                        "message": "I couldn't tell if you're answering the question or asking something new. Could you clarify?",
                        "session_id": session_id
                    }
            else:
                response = await self.interact(state)
                self.session_manager.update_state(session_id, state)
                response["session_id"] = session_id
                return response

        # If all Q&A complete, continue with main processing
        email = answers.get("email", "default@example.com")
        response = await self.graph.ainvoke({
            "input": input_text or "",
            "has_image": has_image,
            "email": email
        })

        self.session_manager.update_state(session_id, state)

        # Attach session_id for tracking
        if isinstance(response, dict):
            response["session_id"] = session_id

        return response



# Example usage
async def main():
    router = IntentRouter()

    out1 = await router.run("Rahul")
    print(out1)

async def chat_loop():
    router = IntentRouter()
    session_id = None

    # Start the conversation with initial question
    response = await router.run(input_text=None, session_id=session_id)
    session_id = response.get("session_id", session_id)
    if response.get("status") == "asking":
        print(f"Bot: {response['question']}")

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print("\n[INFO] Input closed. Exiting chat.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        response = await router.run(user_input, session_id=session_id)
        session_id = response.get("session_id", session_id)

        if response.get("status") == "asking":
            print(f"Bot: {response['question']}")
        elif response.get("status") == "denial":
            print(f"Bot: {response['message']}")
        elif response.get("status") == "unclear_input":
            print(f"Bot: {response['message']}")
        elif response.get("response"):
            print(f"Bot: {response['response']}")
        elif response.get("status") == "ready_for_questions":
            print(f"Bot: {response['message']}")


if __name__ == "__main__":
    asyncio.run(chat_loop())
