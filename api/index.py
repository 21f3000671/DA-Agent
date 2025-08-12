from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import logging
import io
import json

# Try to import the agent, but provide fallback if it fails
try:
    from agent.main import run_agent
    AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Agent import failed: {e}. Running in minimal mode.")
    AGENT_AVAILABLE = False
    
    async def run_agent(questions, files):
        return {
            "error": "Agent functionality not available in minimal deployment",
            "questions_received": questions,
            "files_count": len(files) if files else 0,
            "message": "This is a minimal deployment. Heavy dependencies are disabled to fit Vercel's 250MB limit."
        }

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    This endpoint is the main entry point for the data analyst agent.
    It accepts a 'questions.txt' file along with optional data files
    and passes them to the agent for processing.
    """
    logger.info(f"Received {len(files)} files: {[f.filename for f in files]}")

    # The prompt guarantees that 'questions.txt' will always be sent.
    # We find it in the list of uploaded files.
    questions_file = next((f for f in files if f.filename == 'questions.txt'), None)

    if not questions_file:
        logger.error("'questions.txt' not found in the uploaded files.")
        raise HTTPException(status_code=400, detail="Missing required file: questions.txt")

    try:
        # Read the content of the questions file to pass to the agent.
        question_content = (await questions_file.read()).decode('utf-8')
        logger.info(f"Questions received:\n-----\n{question_content}\n-----")

        # The file has been read, so reset the pointer to the beginning.
        # This allows the agent's context loader to read it again if needed (e.g., if it's a CSV named questions.txt).
        await questions_file.seek(0)

        # Call the agent orchestrator with the questions and the full list of files.
        result = await run_agent(questions=question_content, files=files)

        if result and "error" in result:
             logger.error(f"Agent returned an error: {result['error']}")
             # Return a 500 error if the agent failed
             raise HTTPException(status_code=500, detail=result['error'])

        logger.info("Agent processing complete, returning result.")
        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred in the API endpoint: {e}", exc_info=True)
        # exc_info=True logs the full stack trace
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def home():
    """
    A simple GET endpoint to confirm that the service is up and running.
    """
    return {"message": "Data Analyst Agent is running. Use the /api/ endpoint to post tasks."}

@app.get("/health")
async def health():
    """
    Health check endpoint for debugging Vercel deployment issues.
    """
    return {
        "status": "healthy",
        "agent_available": AGENT_AVAILABLE,
        "deployment_mode": "minimal" if not AGENT_AVAILABLE else "full",
        "message": "API is operational"
    }
