import json
import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- Business Logic & Tools ---
def mock_lead_capture(name, email, platform):
    """Call this tool only when name, email, and platform are all collected."""
    print(f"\n[SYSTEM] Lead captured successfully: {name}, {email}, {platform}")
    return "Lead successfully saved to CRM."

# --- State Management ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: Optional[str]
    lead_data: dict # name, email, platform
    lead_captured: bool

# --- Nodes ---

def intent_detector(state: AgentState):
    """Classifies the user's latest message intent."""
    last_message = state['messages'][-1].content
    
    # Prompt for intent detection
    prompt = ChatPromptTemplate.from_template(
        "Classify the following user message into ONE of these categories: "
        "['greeting', 'pricing', 'high_intent', 'other'].\n\n"
        "Message: {message}\n\nCategory:"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    intent = chain.invoke({"message": last_message}).content.lower().strip()
    
    return {"intent": intent if intent in ['greeting', 'pricing', 'high_intent'] else 'other'}

def retriever(state: AgentState):
    """Retrieves pricing and policy info if needed."""
    if state['intent'] == 'pricing':
        with open('knowledge.json', 'r') as f:
            kb = json.load(f)
        context = f"Pricing Info: {json.dumps(kb['pricing'])}\nPolicies: {json.dumps(kb['policies'])}"
        return {"messages": [AIMessage(content=f"Context for pricing query: {context}")]}
    return {}

def lead_collector(state: AgentState):
    """Handles logic for asking lead capture questions one by one."""
    data = state.get('lead_data', {})
    name = data.get('name')
    email = data.get('email')
    platform = data.get('platform')
    
    if not name:
        response = "I'd love to help you get started with AutoStream! First, what's your name?"
    elif not email:
        response = f"Nice to meet you, {name}! What's the best email address to reach you at?"
    elif not platform:
        response = "Great! And finally, which platform do you primarily create for (YouTube, Instagram, TikTok, etc.)?"
    else:
        # All info collected
        status = mock_lead_capture(name, email, platform)
        return {
            "messages": [AIMessage(content=f"Thanks! {status} One of our experts will reach out to you shortly.")],
            "lead_captured": True
        }
    
    return {"messages": [AIMessage(content=response)]}

def responder(state: AgentState):
    """Generates the main conversational response."""
    intent = state['intent']
    messages = state['messages']
    
    # If high intent and not yet captured, go to lead collector
    if intent == 'high_intent' and not state.get('lead_captured'):
        # This will be handled by the router
        return {}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly, helpful, and slightly persuasive AI agent for AutoStream, "
                   "an AI video platform. Keep responses concise and guide the user toward subscribing."),
        ("placeholder", "{messages}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm
    
    # Use only the last real user message + context if pricing was added
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}

# --- Router ---
def route_after_intent(state: AgentState):
    if state['intent'] == 'high_intent' or state.get('lead_data'):
        # If we are in high intent or in the middle of lead capture
        if state.get('lead_captured'):
            return "responder"
        return "lead_collector"
    
    if state['intent'] == 'pricing':
        return "retriever"
    
    return "responder"

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("intent_detector", intent_detector)
workflow.add_node("retriever", retriever)
workflow.add_node("lead_collector", lead_collector)
workflow.add_node("responder", responder)

workflow.set_entry_point("intent_detector")

workflow.add_conditional_edges(
    "intent_detector",
    route_after_intent,
    {
        "lead_collector": "lead_collector",
        "retriever": "retriever",
        "responder": "responder"
    }
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("lead_collector", END)
workflow.add_edge("responder", END)

# Compile
app = workflow.compile()

# --- Entry Point ---
if __name__ == "__main__":
    print("Welcome to AutoStream AI. Type 'exit' to quit.")
    state = {
        "messages": [],
        "intent": None,
        "lead_data": {},
        "lead_captured": False
    }
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        
        # In a real scenario, we might need to update lead_data based on user_input here
        # For simplicity, we assume the lead_collector logic handles the sequences.
        # To actually parse the answers, you'd add a node that extracts entities.
        
        result = app.invoke(state)
        state = result
        print(f"Agent: {state['messages'][-1].content}")
