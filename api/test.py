from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Minimal test is working"}

@app.get("/health")
async def health():
    return {"status": "healthy", "test": "minimal"}