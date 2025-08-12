from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "FastAPI minimal test working", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "requirements": "minimal"}