import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WriterAgent:
    """
    Writer Agent specialized in synthesizing research into structured reports.
    Following SOLID: Single Responsibility Principle.
    """
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        logger.info(f"Initialized WriterAgent with model {model}")

    async def run(self, research_findings: str) -> Dict[str, Any]:
        """
        Synthesizes research findings into a professional, structured report.
        """
        logger.info("WriterAgent starting to synthesize research findings.")
        
        system_prompt = (
            "You are a professional technical writer. Your task is to transform "
            "the provided research findings into a comprehensive, well-structured, "
            "and engaging report. Use Markdown for formatting. Ensure clarity and professional tone."
        )
        
        input_content = f"Research findings to synthesize:\n\n{research_findings}"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_content)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            logger.info("WriterAgent successfully generated the report.")
            return {
                "role": "writer",
                "report": response.content,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in WriterAgent: {str(e)}")
            return {
                "role": "writer",
                "report": "",
                "status": "error",
                "error": str(e)
            }
