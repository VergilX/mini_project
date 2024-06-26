from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, WebSocket, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Annotated

import jwt
from jwt.exceptions import InvalidTokenError

from passlib.context import CryptContext

from db.db_actions import *


# token
# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "103446f45914523b72b1caefb899a9aa9271e9c25bd63eb05c751aa1fd63e901"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    },

    "abhinand": {
        "username": "abhinand",
        "full_name": "Abhinand D Manoj",
        "email": "abhinand.dmanoj@gmail.com",
        "hashed_password": "blah",
        "disabled": False,
    }

}


# schemas (must import from db/models.py)
class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


CURRENT_USERS = dict()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def login():
    pass


@app.post("/register")
def register(
    name: str,
    username: str,
    password: str,
    email: EmailStr,
    district: str = None
    ):
    data = dict(
        username=username,
        full_name=name,
        email=email,
        hashed_password=get_password_hash(password),
        disabled=False
    )

    fake_users_db[username] = data
    print(fake_users_db)
    return "User registered successfully"


def logout():
    pass


def is_logged_in():
    pass


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


# Getting user from database
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        print("no username in db")
        return False
    if not verify_password(password, user.hashed_password):
        print("encrypted password verification failed")
        return False
    return user


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
    ):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token")
async def get_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    ) -> Token:
    print(fake_users_db)
    print(f"form username: {form_data.username}, form password: {form_data.password}")
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/home", response_model=User)
async def home(
        current_user: Annotated[User, Depends(get_current_active_user)],
    ):
    # Return user details
    return current_user


# Chat stuff
# Everyone connected to the websocket will get the messages
@app.websocket("/ws/{user_id}/{target_id}")
async def websocket_endpoint(user_id: str, target_id: str, websocket: WebSocket):
    await websocket.accept()
    print("client: ", user_id)

    CURRENT_USERS[user_id] = websocket
    print(CURRENT_USERS)

    try:
        while True:
            data = await websocket.receive_text()

            # Send data to target
            for user, user_ws in CURRENT_USERS.items():
                if user in [target_id, user_id]:
                    await user_ws.send_text(f"{user_id}: {data}")

            # write code to save to database
    except WebSocketDisconnect:
        # Exception already does websocket.close()
        del CURRENT_USERS[user_id]
        print(f"{user_id} has disconnected")
