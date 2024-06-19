from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Annotated

from db.db_actions import *

CURRENT_USERS = dict()

app = FastAPI()


def login():
    pass


def register():
    pass


def logout():
    pass


def is_logged_in():
    pass


def home():
    pass


# Chat stuff
# Everyone connected to the websocket will get the messages
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(user_id: str, websocket: WebSocket):
    await websocket.accept()
    print("client: ", user_id)

    CURRENT_USERS[user_id] = websocket
    print(CURRENT_USERS)

    try:
        while True:
            data = await websocket.receive_text()

            # Send data to every other user
            for user, user_ws in CURRENT_USERS.items():
                await user_ws.send_text(f"{user_id}: {data}")
    except WebSocketDisconnect:
        del CURRENT_USERS[user_id]
        await websocket.close()
        print(f"{user_id} has disconnected")
