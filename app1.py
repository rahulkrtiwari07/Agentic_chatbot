import base64
import json
import os

from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
#from fastapi_websocket_rpc import RpcMethods, WebSocketRpcServer
from pyngrok import ngrok
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv('.env.example')

from twilio_transcriber import TwilioTranscriber  # Assuming you have this

# FastAPI settings
PORT = 8000  # Changed to 8000, common for FastAPI
DEBUG = False
INCOMING_CALL_ROUTE = "/"
WEBSOCKET_ROUTE = "/realtime"

# Twilio authentication
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
api_key = os.environ["TWILIO_API_KEY_SID"]
api_secret = os.environ["TWILIO_API_SECRET"]
client = Client(api_key, api_secret, account_sid)

# Twilio phone number to call
TWILIO_NUMBER = os.environ["TWILIO_NUMBER"]

# ngrok authentication
ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

'''class RpcApi(RpcMethods):
    async def get_transcription(self, data):
        #This is just a placeholder, you can add more functionality here.
        print(f"Received data via RPC: {data}")
        return {"result": "Data received"}'''

@app.get("/")  # Add this route
async def read_root():
    return {"message": "Voice Bot is running"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post(INCOMING_CALL_ROUTE)
async def receive_call(request: Request):
    xml = f"""
<Response>
    <Say>
        Speak to see your speech transcribed in the console
    </Say>
    <Connect>
        <Stream url='wss://{request.headers["host"]}{WEBSOCKET_ROUTE}' />
    </Connect>
</Response>
""".strip()
    return Response(content=xml, media_type="application/xml")

@app.websocket(WEBSOCKET_ROUTE)
async def transcription_websocket(websocket: WebSocket):
    await websocket.accept()
    transcriber = TwilioTranscriber()
    transcriber.connect()
    print('transcriber connected')

    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            match data["event"]:
                case "connected":
                    print("twilio connected")
                case "start":
                    print("twilio started")
                case "media":
                    payload_b64 = data["media"]["payload"]
                    payload_mulaw = base64.b64decode(payload_b64)
                    transcriber.stream(payload_mulaw)
                case "stop":
                    print("twilio stopped")
                    transcriber.close()
                    print("transcriber closed")
                    break

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        transcriber.close()
        print("transcriber closed (finally)")

if __name__ == "__main__":
    import uvicorn

    try:
        listener = ngrok.forward(f"http://localhost:{PORT}")
        print(f"Ngrok tunnel opened at {listener.url()} for port {PORT}")
        NGROK_URL = listener.url()

        twilio_numbers = client.incoming_phone_numbers.list()
        twilio_number_sid = [
            num.sid for num in twilio_numbers if num.phone_number == TWILIO_NUMBER
        ][0]
        client.incoming_phone_numbers(twilio_number_sid).update(
            account_sid, voice_url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}"
        )

        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

    finally:
        ngrok.disconnect()