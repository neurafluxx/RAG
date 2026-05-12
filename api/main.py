import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from api.engine import generate_response

# Load environment variables
load_dotenv()

app = FastAPI(title="Neuraflux Chatbot API")

# Load allowed origins from environment variable
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# allow_credentials cannot be True with wildcard "*"
_allow_credentials = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Thread pool for async execution
executor = ThreadPoolExecutor(max_workers=10)

# Request model
class ChatRequest(BaseModel):
    query: str

# Health check endpoint
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "NeuraFlux Chatbot API is running."
    }

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        executor,
        generate_response,
        request.query
    )

    return {"response": response}

# Run app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

    