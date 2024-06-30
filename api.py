from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, WebSocket, HTTPException, status, Request, Cookie,  Form, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from typing import Annotated

import jwt
from jwt.exceptions import InvalidTokenError

from passlib.context import CryptContext

from sqlalchemy import distinct
from sqlmodel import Session, select
import db.actions as database
import db.models as models

"""
Notes:

[ ] Use classes for requests; eg: user registration

"""

# token
# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "103446f45914523b72b1caefb899a9aa9271e9c25bd63eb05c751aa1fd63e901"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# schemas (must import from db/models.py)
class User(BaseModel):
    name: str | None = None
    username: str
    email: str | None = None
    district: str


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


CURRENT_USERS = dict()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

templates = Jinja2Templates(directory="ui")

app = FastAPI()

# Middleware

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSS and JS
app.mount("/css", StaticFiles(directory="ui/css"), name="css")
app.mount("/js", StaticFiles(directory="ui/js"), name="js")


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse(
        request=request, name="login.html"
    )


@app.post("/login")
def login():
    pass


@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse(
        request=request, name="register.html"
    )


@app.post("/register")
async def register(
    name: Annotated[str | None, Form()],
    username: Annotated[str | None, Form()],
    email: Annotated[EmailStr | None, Form()],
    password: Annotated[str | None, Form()],
    district: Annotated[str | None, Form()],
    ):
    data = dict(
        name=name,
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        district=district
    )

    # Check if present in db, if yes, error else create
    user = database.get_entity(database.USER, data['username'])
    if user is not None:
        print("User already present in db")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,  # change the code
            detail="Email already used",
        )

    database.create(database.USER, data)
    return {
        "detail": "User registered successfully",
    }


# You don't log out with jwt in backend
def logout():
    pass


def get_password_hash(password):
    return pwd_context.hash(password)


# Getting user from database
def check_header_token(Authorization: Annotated[str | None, Cookie()] = None):
    """ Function checks header for Bearer token """

    if Authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header not available",
            headers={"WWW-Authenticate": "Bearer"},
        )

    cookie_components = Authorization.split()
    if cookie_components[0] == "Bearer" and len(cookie_components) == 2:
        return cookie_components[1]
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid cookie format",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token: Annotated[str, Depends(check_header_token)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        print("means most likely Bearer cookie is present", token)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("Payload:", payload)
        username: str = payload.get("sub")
        if username is None:
            print("No username in decoded token")
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        print("InvalidTokenError")
        raise credentials_exception
    user = database.get_entity(database.USER, key=token_data.username)
    if user is None:
        raise credentials_exception
    return user


# idk if this is needed
"""
async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
    ):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
"""


@app.get("/home", response_class=HTMLResponse)
async def home(
        request: Request,
        current_user: Annotated[User, Depends(get_current_user)],
    ):
    # Return user home page
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "user": current_user,
        }
    )


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


def authenticate_user(username: str, password: str):
    user = database.get_entity(database.USER, key=username)
    if not user:
        print("no username in db")
        return False
    if not verify_password(password, user.hashed_password):
        print("encrypted password verification failed")
        return False
    return user


@app.post("/token")
async def get_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    ) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
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


# Chat stuff
@app.get("/chat-list")
async def chat_page(
        request: Request,
        user: Annotated[User, Depends(get_current_user)],
    ):

    # Get names of doctors with chats and last message
    with Session(models.engine) as session:
        query = select(distinct(models.Doctor.name)).where(models.Chat.doctor_id == models.Doctor.id).where(models.Chat.user_id == user.id)

        result = session.exec(query)

        return templates.TemplateResponse(
            request=request,
            name="listchat.html",
            context={
                "doctors": result
            }
        )


@app.get("/chat/{doctor_id}")
async def chat(
    request: Request,
    doctor_id: int,
    user: Annotated[User, Depends(get_current_user)],
    ):
    doctor = database.get_entity(database.DOCTOR, doctor_id)

    with Session(models.engine) as session:
        query = select(models.Chat.message).where(models.Chat.doctor_id == models.Doctor.id).where(models.Chat.user_id == user.id)

        chats = session.exec(query)

        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={
                "doctor": doctor,
                "chats": chats,
            }
        )


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
