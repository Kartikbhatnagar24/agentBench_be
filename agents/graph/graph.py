from langgraph.graph import StateGraph, END
from agents.fact_check import factcheck_agent
from agents.retrieval import make_retrieval_agent
from agents.synthesis import synthesis_agent
from agents.models.pipeline import PipelineState

from dotenv import load_dotenv




def build_graph(vectorstore):
    """Build and compile the RAG pipeline graph, binding the vectorstore."""
    load_dotenv()

    graph = StateGraph(PipelineState)

    graph.add_node("retrieval_agent", make_retrieval_agent(vectorstore))
    graph.add_node("synthesis_agent", synthesis_agent)
    graph.add_node("factcheck_agent", factcheck_agent)

    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", "factcheck_agent")

    # Phase 3: conditional edge
    # if confidence < 0.6 and retry_count < 2 → back to retrieval
    # else → END
    graph.add_conditional_edges(
        "factcheck_agent",
        lambda state: (
            "retrieval_agent"
            if state.confidence_score < 0.6 and state.retry_count < 2
            else END
        ),
    )

    return graph.compile()