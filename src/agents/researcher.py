import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearcherAgent:
    """
    Researcher Agent specialized in gathering and extracting information.
    Following SOLID: Single Responsibility Principle.
    """
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        logger.info(f"Initialized ResearcherAgent with model {model}")

    async def run(self, query: str, context: List[str] = None) -> Dict[str, Any]:
        """
        Simulates information gathering and extraction based on the query.
        In a real-world scenario, this would include web search tool calls.
        """
        logger.info(f"ResearcherAgent processing query: {query}")
        
        system_prompt = (
            "You are an expert researcher. Your task is to gather detailed information, "
            "extract key insights, and identify relevant facts for the given topic. "
            "Provide high-quality data that the Writer agent can use."
        )
        
        input_content = f"Topic to research: {query}\n"
        if context:
            input_content += f"Additional Context: {' '.join(context)}"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_content)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            logger.info("ResearcherAgent successfully gathered information.")
            return {
                "role": "researcher",
                "findings": response.content,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in ResearcherAgent: {str(e)}")
            return {
                "role": "researcher",
                "findings": "",
                "status": "error",
                "error": str(e)
            }
