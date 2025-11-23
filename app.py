"""
Wine Sommelier Agent - Streamlit Web Application
A conversational AI sommelier powered by LangGraph with RAG capabilities.
"""

import json
import streamlit as st
from typing import Annotated, Optional, List, Literal
from dotenv import load_dotenv
import os

from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.tools import tool

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

# Load environment variables
load_dotenv()

# Helper function to get secrets (supports both .env and Streamlit Cloud)
def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets or environment variables"""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except:
        return os.getenv(key, default)

# Page configuration
st.set_page_config(
    page_title="Wine Sommelier AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #722F37;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChat message {
        padding: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .wine-tip {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_models():
    """Initialize embeddings and LLM (cached for performance)"""
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = get_secret("EMBEDDING_DEPLOYMENT_NAME")

    # Set OpenAI API key for Streamlit Cloud
    openai_key = get_secret("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    embeddings = AzureOpenAIEmbeddings(
        deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        base_url=None,
        azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT"),
    )

    llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)

    return embeddings, llm


@st.cache_resource
def load_vector_store(_embeddings):
    """Load the wine vector store"""
    PERSIST_DIR = "./faiss_index_wine"

    if os.path.exists(PERSIST_DIR):
        vector_store = FAISS.load_local(PERSIST_DIR, _embeddings, allow_dangerous_deserialization=True)
    else:
        PERSIST_DIR = "./faiss_index"
        vector_store = FAISS.load_local(PERSIST_DIR, _embeddings, allow_dangerous_deserialization=True)

    return vector_store


def format_docs(docs):
    return "\n\n".join(f"{json.dumps(d.metadata)}: {d.page_content}" for d in docs)


# Initialize models
embeddings, llm = initialize_models()
vector_store = load_vector_store(embeddings)

# Create RAG chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable wine sommelier. Use the provided wine database context to answer questions about wines, provide recommendations, and suggest food pairings. If you don't find relevant information, say so."),
    ("human", "Question: {question}\n\nWine Database Context:\n{context}\n\nAnswer:")
])

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


# Define Tools
@tool(description="Search wine database for recommendations, information about wines, or pairing suggestions. Use this for any wine-related queries.")
def wine_search(query: Annotated[str, "Search query about wines, preferences, regions, or pairings"]) -> str:
    """Uses RAG chain to find relevant wines from the database"""
    result = rag_chain.invoke(query)
    return result


@tool(description="Get weather information for wine serving temperature recommendations")
def get_weather(city: Annotated[str, "City name for weather lookup"]) -> str:
    """Returns weather info to help with serving temperature suggestions"""
    import random
    temp = random.randint(15, 35)
    conditions = random.choice(["Sunny", "Cloudy", "Warm", "Cool"])

    suggestion = ""
    if temp > 25:
        suggestion = "Serve white wines well-chilled (7-10°C) and reds slightly cool (14-16°C)"
    else:
        suggestion = "Standard serving temperatures work well today"

    return f"Weather in {city}: {temp}°C, {conditions}. {suggestion}"


@tool(description="Calculate wine and food pairing compatibility score")
def pairing_score(
    wine_type: Annotated[str, "Type of wine: red, white, rosé, sparkling, or dessert"],
    food: Annotated[str, "Food dish to pair with the wine"]
) -> str:
    """Returns a compatibility score for wine-food pairing"""
    pairings = {
        ("red", "steak"): (95, "Excellent! Tannins complement the rich meat"),
        ("red", "beef"): (95, "Excellent! Classic pairing"),
        ("red", "lamb"): (90, "Great match with the gamey flavors"),
        ("red", "cheese"): (85, "Good pairing, especially aged cheeses"),
        ("red", "pasta"): (80, "Works well with tomato-based sauces"),
        ("white", "fish"): (92, "Perfect! Acidity complements seafood"),
        ("white", "seafood"): (92, "Excellent pairing"),
        ("white", "chicken"): (85, "Good match for lighter preparations"),
        ("white", "salad"): (80, "Refreshing combination"),
        ("sparkling", "appetizers"): (90, "Great for starting a meal"),
        ("sparkling", "oysters"): (95, "Classic luxurious pairing"),
        ("rosé", "salad"): (85, "Light and refreshing"),
        ("dessert", "chocolate"): (88, "Sweet wines complement desserts"),
    }

    key = (wine_type.lower(), food.lower())
    if key in pairings:
        score, reason = pairings[key]
    else:
        score = 70
        reason = "Moderate pairing - can work but not ideal"

    return f"Pairing Score: {score}/100\nWine: {wine_type}\nFood: {food}\nAnalysis: {reason}"


@tool(description="Save a wine recommendation to user's favorites list")
def save_favorite(
    wine_name: Annotated[str, "Name of the wine to save"],
    notes: Annotated[str, "User's notes about why they liked this wine"]
) -> str:
    """Saves wine to favorites for future reference"""
    if "favorites" not in st.session_state:
        st.session_state.favorites = []

    st.session_state.favorites.append({
        "wine": wine_name,
        "notes": notes
    })

    return f"Successfully saved '{wine_name}' to your favorites!\nNotes: {notes}"


# Define Custom State
class WineSommelierState(BaseModel):
    """Custom state for Wine Sommelier workflow"""
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    conversation_summary: Optional[str] = None
    final_answer: Optional[str] = None
    user_intent: Optional[str] = None
    wine_context: Optional[str] = None
    recommendation_draft: Optional[str] = None
    quality_score: Optional[float] = None
    remaining_steps: int = 10
    needs_improvement: bool = False
    optimization_attempts: int = 0


# Node implementations
def intent_classifier(state: WineSommelierState) -> dict:
    last_message = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break

    classification_prompt = f"""Classify the user's intent into ONE of these categories:
    - RECOMMENDATION: User wants wine recommendations
    - PAIRING: User wants food-wine pairing advice
    - INFORMATION: User wants to learn about wines, regions, or varieties
    - GENERAL: General conversation or greeting

    User message: {last_message}

    Return ONLY the category name, nothing else."""

    response = llm.invoke([SystemMessage(classification_prompt)])
    intent = response.content.strip().upper()

    valid_intents = ["RECOMMENDATION", "PAIRING", "INFORMATION", "GENERAL"]
    if intent not in valid_intents:
        intent = "GENERAL"

    return {"user_intent": intent}


def rag_retriever(state: WineSommelierState) -> dict:
    query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    docs = retriever.invoke(query)
    context = format_docs(docs)

    return {"wine_context": context}


def evaluator(state: WineSommelierState) -> dict:
    draft = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and msg.content:
            draft = msg.content
            break

    if not draft:
        return {
            "quality_score": 0.0,
            "needs_improvement": True,
            "recommendation_draft": ""
        }

    eval_prompt = f"""Evaluate this wine recommendation on a scale of 0-100.

Recommendation:
{draft}

Scoring criteria:
- Relevance to user query (0-25 points)
- Specificity of wine details (grape, region, vintage) (0-25 points)
- Helpfulness of serving/pairing suggestions (0-25 points)
- Clarity and engaging presentation (0-25 points)

Return ONLY a single number (the total score), nothing else."""

    response = llm.invoke([SystemMessage(eval_prompt)])

    try:
        score = float(response.content.strip())
        score = max(0, min(100, score))
    except:
        score = 70.0

    needs_improvement = score < 75 and state.optimization_attempts < 2

    return {
        "quality_score": score,
        "needs_improvement": needs_improvement,
        "recommendation_draft": draft
    }


def optimizer(state: WineSommelierState) -> dict:
    optimize_prompt = f"""Improve this wine recommendation to score higher.

Current recommendation (scored {state.quality_score}/100):
{state.recommendation_draft}

Improvements needed:
- Add more specific wine details (grape varieties, regions, producers)
- Include serving temperature and glass recommendations
- Add food pairing suggestions
- Make it more engaging and personalized

Provide the improved recommendation:"""

    response = llm.invoke([
        SystemMessage("You are a master sommelier improving wine recommendations to be more helpful and specific."),
        HumanMessage(optimize_prompt)
    ])

    improved = response.content

    return {
        "recommendation_draft": improved,
        "optimization_attempts": state.optimization_attempts + 1,
        "messages": [AIMessage(content=improved)]
    }


def answer_saver(state: WineSommelierState) -> dict:
    final = state.recommendation_draft
    if not final:
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage) and msg.content:
                final = msg.content
                break

    return {"final_answer": final}


def summarizer(state: WineSommelierState) -> dict:
    if len(state.messages) < 3:
        return {}

    recent_messages = state.messages[-6:] if len(state.messages) > 6 else state.messages

    summary_prompt = """Summarize this wine consultation conversation in 2-3 sentences:
    - What the user was looking for
    - What recommendations were made
    - Any specific preferences mentioned"""

    response = llm.invoke([
        SystemMessage(summary_prompt),
        *recent_messages
    ])

    summary = response.content
    return {"conversation_summary": summary}


# Conditional edge functions
def route_by_intent(state: WineSommelierState) -> Literal["rag_retriever", "agent"]:
    if state.user_intent in ["RECOMMENDATION", "PAIRING", "INFORMATION"]:
        return "rag_retriever"
    return "agent"


def should_continue_agent(state: WineSommelierState) -> Literal["tools", "evaluator"]:
    last_message = state.messages[-1] if state.messages else None

    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    return "evaluator"


def route_after_evaluation(state: WineSommelierState) -> Literal["optimizer", "answer_saver"]:
    if state.needs_improvement:
        return "optimizer"
    return "answer_saver"


@st.cache_resource
def build_wine_sommelier_graph():
    """Builds the complete Wine Sommelier workflow"""
    tools = [wine_search, get_weather, pairing_score, save_favorite]
    tool_node = ToolNode(tools)

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    workflow = StateGraph(WineSommelierState)

    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("rag_retriever", rag_retriever)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("optimizer", optimizer)
    workflow.add_node("answer_saver", answer_saver)
    workflow.add_node("summarizer", summarizer)

    workflow.add_edge(START, "intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "rag_retriever": "rag_retriever",
            "agent": "agent"
        }
    )

    workflow.add_edge("rag_retriever", "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue_agent,
        {
            "tools": "tools",
            "evaluator": "evaluator"
        }
    )

    workflow.add_edge("tools", "agent")

    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "optimizer": "optimizer",
            "answer_saver": "answer_saver"
        }
    )

    workflow.add_edge("optimizer", "agent")
    workflow.add_edge("answer_saver", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow.compile()


# Build the graph
graph = build_wine_sommelier_graph()


def chat_with_sommelier(user_message: str) -> tuple[str, dict]:
    """Chat with the Wine Sommelier and return response with metadata"""

    result = graph.invoke({
        "messages": [
            SystemMessage("You are a knowledgeable and friendly wine sommelier. Help users discover wines they'll love."),
            HumanMessage(user_message)
        ]
    })

    final_state = WineSommelierState(**result)

    metadata = {
        "intent": final_state.user_intent,
        "quality_score": final_state.quality_score,
        "optimization_attempts": final_state.optimization_attempts,
        "summary": final_state.conversation_summary
    }

    return final_state.final_answer or "I couldn't generate a response. Please try again.", metadata


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "favorites" not in st.session_state:
    st.session_state.favorites = []


# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This AI Sommelier uses advanced language models and a wine database
    to help you discover perfect wines for any occasion.
    """)

    st.markdown("---")

    st.markdown("### Capabilities")
    st.markdown("""
    - Wine recommendations
    - Food pairing suggestions
    - Wine region information
    - Serving temperature advice
    - Save favorites
    """)

    st.markdown("---")

    st.markdown("### Example Questions")
    examples = [
        "Recommend a red wine for grilled steak",
        "What wine pairs well with salmon?",
        "Tell me about wines from Burgundy",
        "I need a wine for a romantic dinner, budget $40"
    ]

    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.example_query = example

    st.markdown("---")

    # Favorites section
    if st.session_state.favorites:
        st.markdown("### Your Favorites")
        for i, fav in enumerate(st.session_state.favorites):
            with st.expander(f"{fav['wine'][:30]}..."):
                st.write(f"**Wine:** {fav['wine']}")
                st.write(f"**Notes:** {fav['notes']}")

        if st.button("Clear Favorites"):
            st.session_state.favorites = []
            st.rerun()

    st.markdown("---")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main content
st.markdown('<h1 class="main-header">Wine Sommelier AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your personal AI wine expert - ask me anything about wines!</p>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message and message["metadata"]:
            with st.expander("Details"):
                meta = message["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intent", meta.get("intent", "N/A"))
                with col2:
                    score = meta.get("quality_score", 0)
                    st.metric("Quality Score", f"{score:.0f}/100" if score else "N/A")

# Handle example query from sidebar
if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, metadata = chat_with_sommelier(prompt)
            st.markdown(response)

            with st.expander("Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intent", metadata.get("intent", "N/A"))
                with col2:
                    score = metadata.get("quality_score", 0)
                    st.metric("Quality Score", f"{score:.0f}/100" if score else "N/A")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "metadata": metadata
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask me about wines..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the wine cellar..."):
            response, metadata = chat_with_sommelier(prompt)
            st.markdown(response)

            with st.expander("Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intent", metadata.get("intent", "N/A"))
                with col2:
                    score = metadata.get("quality_score", 0)
                    st.metric("Quality Score", f"{score:.0f}/100" if score else "N/A")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "metadata": metadata
    })


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "Powered by LangGraph & OpenAI | Wine database with 5700+ wines"
    "</div>",
    unsafe_allow_html=True
)
