from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
import chromadb
from dataset import docs
from datetime import datetime

# Embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
texts = [d["text"] for d in docs]
embeddings = embedder.encode(texts).tolist()

client = chromadb.Client()
collection = client.create_collection(name="studybuddy")

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[d["id"] for d in docs],
    metadatas=[{"topic": d["topic"]} for d in docs]
)

# State
class State(TypedDict):
    question: str
    messages: List
    route: str
    retrieved: str
    sources: List
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int

# Nodes
def memory_node(state):
    msgs = state.get("messages", [])
    msgs.append(("user", state["question"]))
    return {"messages": msgs[-6:]}

def router_node(state):
    q = state["question"].lower()
    if "time" in q or "date" in q:
        return {"route": "tool"}
    return {"route": "retrieve"}

def retrieval_node(state):
    q_embed = embedder.encode([state["question"]]).tolist()
    res = collection.query(query_embeddings=q_embed, n_results=3)

    context = ""
    for i, doc in enumerate(res["documents"][0]):
        topic = res["metadatas"][0][i]["topic"]
        context += f"[{topic}] {doc}\n"

    return {"retrieved": context, "sources": res["metadatas"][0]}

def tool_node(state):
    return {"tool_result": str(datetime.now())}

def answer_node(state):
    context = state.get("retrieved", "")
    tool = state.get("tool_result", "")
    answer = f"{context}\nTool:{tool}\nAnswer: {state['question']} explained simply."
    return {"answer": answer}

def eval_node(state):
    return {"faithfulness": 0.9, "eval_retries": state.get("eval_retries", 0) + 1}

def save_node(state):
    msgs = state["messages"]
    msgs.append(("assistant", state["answer"]))
    return {"messages": msgs}

# Graph
graph = StateGraph(State)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")

graph.add_edge("memory", "router")
graph.add_edge("retrieve", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("answer", "eval")
graph.add_edge("save", END)

def route_decision(state):
    return state["route"]

graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "tool": "tool"
})

def eval_decision(state):
    if state["faithfulness"] < 0.7 and state["eval_retries"] < 2:
        return "answer"
    return "save"

graph.add_conditional_edges("eval", eval_decision, {
    "answer": "answer",
    "save": "save"
})

app = graph.compile(checkpointer=MemorySaver())