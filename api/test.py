from fastapi import FastAPI, File, UploadFile
from typing import List
import os

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "FastAPI with file upload capability", "status": "ok"}

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "requirements": "essential",
        "env_test": "LLM_API_KEY" in os.environ
    }

@app.post("/api/test")
async def test_upload(files: List[UploadFile] = File(...)):
    """Test file upload functionality"""
    return {
        "files_received": len(files),
        "filenames": [f.filename for f in files],
        "status": "upload_test_successful"
    }