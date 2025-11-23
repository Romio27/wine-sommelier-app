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

    # Set all API keys for Streamlit Cloud
    openai_key = get_secret("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    azure_key = get_secret("AZURE_OPENAI_API_KEY")
    if azure_key:
        os.environ["AZURE_OPENAI_API_KEY"] = azure_key

    azure_endpoint = get_secret("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint

    # Clear conflicting env vars for Azure
    if "OPENAI_API_BASE" in os.environ:
        del os.environ["OPENAI_API_BASE"]

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        api_key=azure_key,
        azure_endpoint=azure_endpoint,
    )

    # Configure ChatOpenAI - use OpenRouter if that's the API key type
    openai_base_url = get_secret("OPENAI_API_BASE")
    if openai_key and openai_key.startswith("sk-or-"):
        # OpenRouter key detected
        llm = ChatOpenAI(
            model="openai/gpt-4.1",  # OpenRouter model format
            temperature=0.0,
            base_url=openai_base_url or "https://openrouter.ai/api/v1",
            api_key=openai_key,
        )
    else:
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


@tool(description="Search online wine stores to find where to buy a specific wine and compare prices")
def search_wine_stores(
    wine_name: Annotated[str, "Name of the wine to search for"],
    country: Annotated[str, "Country for store search (e.g., 'US', 'UA', 'EU')"] = "US"
) -> str:
    """Searches online wine retailers for availability and prices"""
    import random

    # Simulated store data (in production: integrate with Vivino API, Wine.com API)
    base_price = random.randint(20, 80)

    stores = [
        {
            "store": "Vivino",
            "price": f"${base_price + random.randint(-5, 10):.2f}",
            "rating": f"{random.uniform(3.8, 4.8):.1f}/5",
            "availability": "In Stock",
            "url": "vivino.com"
        },
        {
            "store": "Wine.com",
            "price": f"${base_price + random.randint(-3, 15):.2f}",
            "rating": f"{random.uniform(3.5, 4.5):.1f}/5",
            "availability": "In Stock",
            "url": "wine.com"
        },
        {
            "store": "Total Wine",
            "price": f"${base_price + random.randint(-8, 5):.2f}",
            "rating": "N/A",
            "availability": "Limited Stock",
            "url": "totalwine.com"
        },
    ]

    # Sort by price
    stores_sorted = sorted(stores, key=lambda x: float(x['price'].replace('$', '')))

    result = f"🛒 **Where to buy '{wine_name}':**\n\n"
    for store in stores_sorted:
        result += f"**{store['store']}** - {store['price']}\n"
        result += f"   Rating: {store['rating']} | {store['availability']}\n"
        result += f"   🔗 {store['url']}\n\n"

    best = stores_sorted[0]
    result += f"💡 **Best price:** {best['store']} at {best['price']}"

    return result


@tool(description="Get aggregated reviews and ratings for a wine from multiple sources like Vivino, Wine Spectator")
def get_wine_reviews(
    wine_name: Annotated[str, "Name of the wine to get reviews for"],
    vintage: Annotated[str, "Wine vintage year (optional)"] = "recent"
) -> str:
    """Fetches wine reviews from multiple rating sources"""
    import random

    # Simulated review data (in production: integrate with wine rating APIs)
    vivino_score = round(random.uniform(3.8, 4.7), 1)
    ws_score = random.randint(85, 96)
    ct_score = random.randint(84, 94)

    tasting_notes = [
        "Smooth tannins with cherry and blackberry notes",
        "Elegant structure with hints of oak and vanilla",
        "Rich and full-bodied with dark fruit flavors",
        "Balanced acidity with a long, pleasant finish",
        "Complex aromas of spice and ripe fruit"
    ]

    result = f"⭐ **Reviews for {wine_name}**\n"
    if vintage != "recent":
        result += f"Vintage: {vintage}\n"
    result += "\n"

    result += f"🍷 **Vivino**: {vivino_score}/5 ({random.randint(500, 3000)} reviews)\n"
    result += f"   _{random.choice(tasting_notes)}_\n\n"

    result += f"📰 **Wine Spectator**: {ws_score}/100\n"
    result += f"   _{random.choice(tasting_notes)}_\n\n"

    result += f"📊 **CellarTracker**: {ct_score}/100 ({random.randint(30, 200)} reviews)\n"
    result += f"   _{random.choice(tasting_notes)}_\n\n"

    # Calculate average
    avg_score = (vivino_score * 20 + ws_score + ct_score) / 3
    result += f"**Overall Score: {avg_score:.0f}/100** "

    if avg_score >= 90:
        result += "🏆 Excellent!"
    elif avg_score >= 85:
        result += "✨ Very Good"
    else:
        result += "👍 Good"

    return result


@tool(description="Suggest food recipes that pair perfectly with a specific wine type")
def suggest_recipes(
    wine_type: Annotated[str, "Type of wine (e.g., 'Cabernet Sauvignon', 'Chardonnay', 'Pinot Noir')"],
    cuisine: Annotated[str, "Preferred cuisine type (e.g., 'Italian', 'French', 'any')"] = "any",
    difficulty: Annotated[str, "Recipe difficulty: easy, medium, hard"] = "any"
) -> str:
    """Suggests recipes that pair well with the specified wine"""

    # Comprehensive pairing database
    pairings = {
        "cabernet sauvignon": [
            {"name": "Grilled Ribeye Steak", "time": "25 min", "difficulty": "easy", "cuisine": "American"},
            {"name": "Braised Short Ribs", "time": "3 hours", "difficulty": "medium", "cuisine": "French"},
            {"name": "Lamb Chops with Rosemary", "time": "30 min", "difficulty": "easy", "cuisine": "Mediterranean"},
            {"name": "Beef Bourguignon", "time": "2.5 hours", "difficulty": "medium", "cuisine": "French"},
        ],
        "pinot noir": [
            {"name": "Roasted Duck Breast", "time": "40 min", "difficulty": "medium", "cuisine": "French"},
            {"name": "Salmon with Herb Crust", "time": "25 min", "difficulty": "easy", "cuisine": "any"},
            {"name": "Mushroom Risotto", "time": "45 min", "difficulty": "medium", "cuisine": "Italian"},
            {"name": "Coq au Vin", "time": "2 hours", "difficulty": "medium", "cuisine": "French"},
        ],
        "chardonnay": [
            {"name": "Lobster with Butter Sauce", "time": "30 min", "difficulty": "medium", "cuisine": "French"},
            {"name": "Creamy Chicken Alfredo", "time": "35 min", "difficulty": "easy", "cuisine": "Italian"},
            {"name": "Baked Brie with Honey & Nuts", "time": "15 min", "difficulty": "easy", "cuisine": "French"},
            {"name": "Shrimp Scampi", "time": "20 min", "difficulty": "easy", "cuisine": "Italian"},
        ],
        "sauvignon blanc": [
            {"name": "Goat Cheese Salad", "time": "15 min", "difficulty": "easy", "cuisine": "French"},
            {"name": "Grilled Sea Bass", "time": "20 min", "difficulty": "easy", "cuisine": "Mediterranean"},
            {"name": "Asparagus Quiche", "time": "50 min", "difficulty": "medium", "cuisine": "French"},
            {"name": "Ceviche", "time": "30 min", "difficulty": "easy", "cuisine": "Latin"},
        ],
        "merlot": [
            {"name": "Roasted Pork Tenderloin", "time": "45 min", "difficulty": "easy", "cuisine": "any"},
            {"name": "Pasta Bolognese", "time": "1.5 hours", "difficulty": "medium", "cuisine": "Italian"},
            {"name": "Stuffed Bell Peppers", "time": "1 hour", "difficulty": "easy", "cuisine": "Mediterranean"},
            {"name": "Beef Tacos", "time": "30 min", "difficulty": "easy", "cuisine": "Mexican"},
        ],
        "riesling": [
            {"name": "Thai Green Curry", "time": "35 min", "difficulty": "medium", "cuisine": "Thai"},
            {"name": "Pork Schnitzel", "time": "25 min", "difficulty": "easy", "cuisine": "German"},
            {"name": "Spicy Shrimp Stir-Fry", "time": "20 min", "difficulty": "easy", "cuisine": "Asian"},
            {"name": "Apple Tart", "time": "1 hour", "difficulty": "medium", "cuisine": "French"},
        ],
    }

    # Find matching wine type
    wine_key = wine_type.lower()
    recipes = None
    for key in pairings:
        if key in wine_key or wine_key in key:
            recipes = pairings[key]
            break

    if not recipes:
        # Default to cabernet pairings for red, chardonnay for white
        if any(word in wine_key for word in ["red", "noir", "merlot", "shiraz", "syrah"]):
            recipes = pairings["cabernet sauvignon"]
        else:
            recipes = pairings["chardonnay"]

    # Filter by cuisine and difficulty if specified
    if cuisine != "any":
        filtered = [r for r in recipes if cuisine.lower() in r['cuisine'].lower() or r['cuisine'] == "any"]
        if filtered:
            recipes = filtered

    if difficulty != "any":
        filtered = [r for r in recipes if r['difficulty'] == difficulty.lower()]
        if filtered:
            recipes = filtered

    result = f"🍳 **Recipes for {wine_type}**\n\n"

    for i, recipe in enumerate(recipes[:4], 1):
        result += f"{i}. **{recipe['name']}**\n"
        result += f"   ⏱️ {recipe['time']} | 📊 {recipe['difficulty'].capitalize()} | 🌍 {recipe['cuisine']}\n\n"

    result += "💡 _These recipes are selected to complement the wine's flavor profile_"

    return result


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
    tools = [wine_search, get_weather, pairing_score, save_favorite, search_wine_stores, get_wine_reviews, suggest_recipes]
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
