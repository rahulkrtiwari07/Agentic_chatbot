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
        return self.sessions.get(session_id, {"answers": {}, "current_key": None, "welcomed": False})

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

    WELCOME_MESSAGE = "Hello! My name is Luna. I am calling on behalf of Microware. May I ask you a few questions?"
    
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

    Determine the user's intent based on the question they were asked and their response.

    User was asked the following question:
    "{question}"

    User replied with:
    "{user_input}"

    Possible intent categories:

    1. An **answer** to a personal information question (like name, age, address or email).
    Examples of answers:
    - "Rahul"
    - "25"
    - "rahul@example.com"
    - "My name is Priya"
    - "I’m 30 years old"
    - "I live in Ghaziabad district of Uttar Pradesh"
    - "House number 04 Near post office district Nainital Uttarakhand"
    - "Mumbai"
    - "Delhi"

    2. Or a **query** asking for general or medical information, or starting a new conversation.
    Examples of queries:
    - "Tell me about population growth"
    - "What is the weather?"
    - "Hi, how are you?"

    “IMPORTANT: If the user was asked ‘Is this your permanent address?’ and replies ‘No’, the intent should be classified as ‘Different address’, not ‘denial’.”

    3. **denial** — the user refuses to provide personal information or objects to question.
    Examples:
    - "I don't want to tell you that"
    - "Why do you need my email?"
    - "I prefer not to share my age"
    - "That's personal"
    - "None of your business"

    4. **acceptance** - The user agrees to provide information to the questions when the user is welcomed and asked whether he is comfortable sharing his personal information or 
    when the user is asked whether the address provided by him is his permanent address and he replies positively or when the user is asked whether he has covid in the last five years and he replies positively.
    Examples:
    - "Hii"
    - "Hello"
    - "Yes"
    - "Yes i can give my information"
    - "You can procedd further"
    - "Ask the question and then i will decide whether i want to answer or not"
    - "Yes"
    - "Yes this is my permanent address"

    

    5. **repeat** - The user asks for the question to be repeated or clarified.
    Examples:
    - "Can you please repeat the question"
    - "Pardon"
    - "I didn't get the question"

    6. **negative** - If the user replies in negative for the question "Would you like to ask any questions now?""
    Examples:
    - "No"
    - "No I don't want to"
    - "No thanks"
    - "No I am done"

    7. **Different address** - If the user is asked "Is this your permanent address?") and he replies in following manner.
    Examples:
    - "No"
    - "No This is not my permanent address"
    - "This is the address where I live and not my permanent address."
    - "This is not the address mentioned on my government ID"     

    8. **no vaccine** -  If the user replied negatively when asked whether he was vaccinated at that time.
    Examples:
    - "No"
    - "No I wasn't vaccinated at that time"

    9. **no covid** - When the user is asked "Have you have covid in the past 5 years" and he replies in follwing way.
    Example:
    - "No"
    - "No I didn't had covid"


    Respond ONLY in the following JSON format:
    {
    "intent": "answer" | "query" | "denial" | "acceptance" | "repeat"| "negative" | "Different address | "no vaccine" | "no covid"
    }
    """

    PURPOSE_CLASSIFIER_PROMPT = """
    You are a purpose classification assistant.

    Context:
    - The user was asked the following question:
    "{question}"
    - The user responded with:
    "{user_input}"

    Instructions:
    1. If the user objects to the question or refuses to provide information:
    - Your sole role is to politely explain the **purpose of data collection**.
    - Clarify that the information is collected to **store user data in a database** to support **better governance by the government**.

    Be empathetic, informative, and clear in your response.
    """

    SERIOUS_ClASSIFIER_PROMPT = """
    You are a purpose classifier assistant.
    
    Context:
    - The user was asked the following question:
    "{question}"
    - The user responded with:
    "{user_input}"

    Instruction:
    The user is not serious about providing the answers to the question or it seems that the input provided are not satisafactory so you need to make him understand that this data collection process
    is for government record and that proper data enhances governance.
    """

    ANSWER_PROMPT = """
        You are a judgment assistant. Your task is to determine whether the input provided by the user is satisfactory or not.

        Context:
        - The user was asked the following question:
        "{question}"
        - The user responded with:
        "{user_input}"

        Criteria for judging the response:

        1. If the question is: "Can you please state your full name?"
        - Responses like "123", "xyz", or anything that clearly cannot be interpreted as a real human name are **not satisfactory**.

        2. If the question is: "What is your age?"
        - The age must be a numeric value between **13 and 105** (inclusive). Any value outside this range is **not satisfactory**.

        3. If the question is: "Can you please provide your address?"
        - The address should refer to a real, habitable location. Answers like "sun", "Mars", or any imaginary or uninhabitable places are **not satisfactory**.

        4. If the question is: "What is your email address?"
        - The email address should be valid and should end with a domain like **@gmail.com**, **@yahoo.com**, **@outlook.com**, etc. Random strings or missing domains are **not satisfactory**.

        Instructions:
        - If the user input is satisfactory, return **"satisfactory"**.
        - If the user input is not satisfactory, return a friendly message encouraging the user to be serious and provide a proper answer according to the question.

        Respond ONLY in the following JSON format:
            {
            "intent": "satisfactory" | {freindly message}
            }
            
        """

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
            ("Thanks", "Thank you! Would you like to ask any questions now?")
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
            result = self.llm1.invoke(messages)
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
                intent = self.classify_intent(input_text, current_question)

                '''if intent == "impossible":
                    explanation = await self.serious_classifier(input_text, current_question)
                    return {
                        "status": "impossible",
                        "message": explanation + "\nCould you please reconsider the question?\n" + current_question,
                        "session_id": session_id
                    }'''

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

                if current_key == "address":
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
                        "message": explanation + "\nCould you please reconsider answering this question?",
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





async def chat_loop():
    router = IntentRouter()
    session_id = None

    # Start the conversation with initial welcome message
    response = await router.run(input_text=None, session_id=session_id)
    session_id = response.get("session_id", session_id)

    # Show welcome or first question
    if response.get("status") == "welcome":
        print(f"Bot: {response['message']}")
    elif response.get("status") == "asking":
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

        if response.get("status") == "welcome":
            print(f"Bot: {response['message']}")
        elif response.get("status") == "asking":
            print(f"Bot: {response['question']}")
        elif response.get("status") == "denial":
            print(f"Bot: {response['message']}")
        elif response.get("status") == "unclear_input":
            print(f"Bot: {response['message']}")
        elif response.get("status") == "ready_for_questions":
            print(f"Bot: {response['message']}")
        elif response.get("response"):
            print(f"Bot: {response['response']}")
        elif response.get("status") == "ended":
            print("Thanks for your support")
            break
        elif response.get("status") == "confirm_permanent_address":
            print(f"Bot: {response['message']}")
        else:
            print(f"Bot: {response.get('message', 'Something went wrong.')}")

if __name__ == "__main__":
    asyncio.run(chat_loop())
