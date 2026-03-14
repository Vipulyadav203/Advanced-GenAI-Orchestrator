import logging
from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from src.agents.researcher import ResearcherAgent
from src.agents.writer import WriterAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """
    Represents the state of the agentic workflow.
    """
    query: str
    research_findings: str
    final_report: str
    errors: List[str]

class WorkflowOrchestrator:
    """
    Coordinates multi-agent interaction using LangGraph.
    Follows SOLID: Open/Closed Principle.
    """
    def __init__(self):
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.workflow = self._build_graph()
        logger.info("Initialized WorkflowOrchestrator with StateGraph.")

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph workflow.
        """
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("research", self._research_step)
        workflow.add_node("write", self._write_step)

        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "write")
        workflow.add_edge("write", END)

        return workflow.compile()

    async def _research_step(self, state: AgentState) -> Dict[str, Any]:
        """
        Research node in the graph.
        """
        logger.info(f"Starting research step for query: {state['query']}")
        result = await self.researcher.run(state["query"])
        
        if result["status"] == "success":
            return {"research_findings": result["findings"]}
        else:
            return {"errors": [result["error"]]}

    async def _write_step(self, state: AgentState) -> Dict[str, Any]:
        """
        Writing node in the graph.
        """
        logger.info("Starting synthesis step.")
        result = await self.writer.run(state["research_findings"])
        
        if result["status"] == "success":
            return {"final_report": result["report"]}
        else:
            return {"errors": [result["error"]]}

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the orchestrated workflow.
        """
        logger.info(f"Running workflow for query: {query}")
        initial_state = {
            "query": query,
            "research_findings": "",
            "final_report": "",
            "errors": []
        }
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            logger.info("Workflow execution completed successfully.")
            return final_state
        except Exception as e:
            logger.error(f"Error during workflow execution: {str(e)}")
            return {"errors": [str(e)]}
