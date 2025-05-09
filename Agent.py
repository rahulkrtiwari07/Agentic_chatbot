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

class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat and greetings.
    2. RAG_AGENT - For specific knowledge about population dynamics and health indicators in India and its districts, along with emerging health and family welfare issues or  fertility, infant and child mortality, family planning practices, maternal and child health, reproductive health, nutrition in Indiand and its districts'.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    Make your decision based on these guidelines:
    - If the user has not uploaded any image, always route to the conversation agent.
    - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    - If the user asks specific knowledge questions, use the RAG agent.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. 

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

    Respond ONLY in the following JSON format:
    {
    "intent": "answer" | "query"
    }
    """

class IntentRouter:
    def __init__(self):
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
        input_text = state['input']

        messages = [
            {"role": "system", "content": AgentConfig.DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": f"User query: {input_text}"}
        ]

        result = self.llm1.invoke(messages)

        try:
            decision = json.loads(result.content)
            confidence = decision.get("confidence", 0)
            if confidence < AgentConfig.CONFIDENCE_THRESHOLD:
                decision["agent"] = "low_confidence_fallback_agent"
        except Exception as e:
            decision = {
                "agent": "parse_error_fallback_agent",
                "reasoning": f"Could not parse response: {e}",
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

    async def run(self, input_text=None, has_image=False, email=None, state=None):
        if state is None:
            state = {}

        answers = state.get("answers", {})
        state["answers"] = answers  # Ensure it's set

        # If not all questions answered
        if len(answers) < len(self.questions):
            if input_text:
                current_key = state.get("current_key")
                current_question = dict(self.questions).get(current_key, "")

                intent = self.classify_intent(input_text, current_question)

                if intent == "answer":
                    if current_key:
                        answers[current_key] = input_text
                        state["answers"] = answers
                    return await self.interact(state)

                elif intent == "query":
                    return await self.graph.ainvoke({
                        "input": input_text,
                        "has_image": has_image,
                        "email": email or answers.get("email", "default@example.com")
                    })

                else:
                    return {
                        "status": "unclear_input",
                        "message": "I couldn't tell if you're answering the question or asking something new. Could you clarify?"
                    }
            else:
                return await self.interact(state)

        # If Q&A complete, proceed normally
        return await self.graph.ainvoke({
            "input": input_text or "",
            "has_image": has_image,
            "email": email or answers.get("email", "default@example.com")
        })


# Example usage
async def main():
    router = IntentRouter()

    out1 = await router.run("Rahul")
    print(out1)

async def simulate_user_interaction():
    router = IntentRouter()
    state = {}

    out = await router.run(state=state)
    print(out)

    out1 = await router.run("Rahul", state=state)
    print(out1)

    out2 = await router.run("25", state=state)
    print(out2)

    out3 = await router.run("rahul@example.com", state=state)
    print(out3)

    out4 = await router.run("Tell me about fertility rates in India", state=state)
    print(out4)

if __name__ == "__main__":
    asyncio.run(simulate_user_interaction())
