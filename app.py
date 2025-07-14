from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from Agent import IntentRouter, chat_loop, log_message, print_bot_message, save_chat_log  # Use your actual imports
import json

app = FastAPI()

# Manage sessions
class ChatSession:
    def __init__(self):
        self.router = IntentRouter()
        self.session_id = None

    async def handle_message(self, message: str) -> dict:
        # If first message is empty, start the conversation
        if self.session_id is None and not message:
            response = await self.router.run(input_text=None, session_id=None)
        else:
            log_message("user", message, self.session_id)
            response = await self.router.run(input_text=message, session_id=self.session_id)

        self.session_id = response.get("session_id", self.session_id)
        return response

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    session = ChatSession()

    try:
        # Wait for the initial client message (or auto-start)
        while True:
            user_input = await websocket.receive_text()

            response = await session.handle_message(user_input)

            # Extract bot response
            status = response.get("status")
            bot_message = (
                response.get("message") or
                response.get("question") or
                response.get("response") or
                "Something went wrong."
            )

            # Send message to client
            await websocket.send_text(json.dumps({
                "status": status,
                "message": bot_message,
                "session_id": session.session_id
            }))

            if status == "ended":
                await websocket.close()
                save_chat_log()
                break

    except WebSocketDisconnect:
        print(f"[INFO] Client disconnected")
        save_chat_log()

