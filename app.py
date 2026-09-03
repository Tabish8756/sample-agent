from fastapi import FastAPI
from pydantic import BaseModel

from agent import chat_agent

app = FastAPI(title="Chat Agent API")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    result = chat_agent.invoke({"messages": [{"role": "user", "content": request.message}]})
    print("Result:", result)
    last_message = result["messages"][-1]
    response_content = last_message.content if hasattr(last_message, "content") else last_message["content"]
    return {"response": response_content}
