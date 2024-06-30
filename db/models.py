from sqlmodel import Field, SQLModel, create_engine
from pydantic import EmailStr


class User(SQLModel, table=True):

    # Set name of table
    __tablename__ = "Users"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    username: str = Field(unique=True, index=True)
    hashed_password: str  # already hashed
    email: EmailStr
    district: str


class Doctor(SQLModel, table=True):

    __tablename__ = "Doctors"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    hosp_id: int | None = Field(default=None, foreign_key="Hospitals.id")
    specialisation_id: int | None = Field(default=None, foreign_key="Specialisation.id")
    successful: int = 0


class Hospital(SQLModel, table=True):

    __tablename__ = "Hospitals"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    district: str | None = Field(default=None, foreign_key="Districts.id")
    location: str = ""
    reputation: int | None = 0


class Chat(SQLModel, table=True):

    __tablename__ = "Chat"

    id: int | None = Field(default=None, index=True, primary_key=True)
    user_id: str | None = Field(default=None, foreign_key="Users.id")
    doctor_id: str | None = Field(default=None, foreign_key="Doctors.id")
    message: str


class Disease(SQLModel, table=True):

    __tablename__ = "Diseases"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    symptoms: str
    causes: str
    treatment: str


class Specialisation(SQLModel, table=True):

    __tablename__ = "Specialisation"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)


class Districts(SQLModel, table=True):

    __tablename__ = "Districts"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    code: str = Field(unique=True, index=True)


# Relative to root of project
sqlite_file_path = "db.db"
sqlite_url = f"sqlite:///{sqlite_file_path}"

engine = create_engine(sqlite_url, echo=True)


# Create tables
def generate_schema():
    SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    # Run file when you wanna create database
    generate_schema()
