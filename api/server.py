import logging
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from src.graph.workflow import WorkflowOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced GenAI Orchestrator API")
orchestrator = WorkflowOrchestrator()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Advanced GenAI Orchestrator API is running."}

@app.post("/generate-report")
async def generate_report(request: QueryRequest):
    """
    Standard HTTP POST endpoint for report generation.
    """
    logger.info(f"Received report generation request: {request.query}")
    try:
        result = await orchestrator.run(request.query)
        if result["errors"]:
            raise HTTPException(status_code=500, detail=result["errors"])
        return result
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time interaction and status updates.
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            # Simple interaction logic
            await websocket.send_json({"status": "processing", "message": f"Processing query: {data}"})
            
            # Execute workflow and stream final result
            result = await orchestrator.run(data)
            await websocket.send_json({"status": "completed", "result": result})
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
