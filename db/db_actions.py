from db_models import *
from sqlmodel import Session, select

# Encoding
USER = 1
DOCTOR = 2
HOSPITAL = 3
CHAT = 4
DISEASE = 5


def create(encoding, data):
    """ Function to add entry to right table """

    # No error checks as of now
    with Session(engine) as session:
        # create entity
        if encoding == USER:

            entity = User(**data)

        elif encoding == DOCTOR:
            entity = Doctor(**data)

        elif encoding == HOSPITAL:
            entity = Hospital(**data)

        elif encoding == CHAT:
            entity = Chat(**data)

        elif encoding == DISEASE:
            entity = Disease(**data)

        # add to session and commit
        session.add(entity)
        session.commit()
        print("Committed changes")


def delete(encoding, key):
    """ function to delete entry from right table """

    # no error checks as of now
    with session(engine) as session:
        # fetch entity
        if encoding == USER:
            statement = select(user).where(user.email == key)

        elif encoding == DOCTOR:
            statement = select(doctor).where(doctor.name == key)

        elif encoding == HOSPITAL:
            statement = select(Hospital).where(Hospital.name == key)

        elif encoding == CHAT:
            statement = select(Chat).where(Chat.id == key)

        elif encoding == DISEASE:
            statement = select(Disease).where(Disease.name == key)

        result = session.exec(statement)
        entity = result.first()

        # if found, delete
        if entity is not None:
            session.delete(entity)
            session.commit()
            print("Entity deleted successfully")
        else:
            print("No entry to delete")


if __name__ == "__main__":
    # Use only for testing purposes
    print("Testing...")

    data = dict(
            name="Abhinand",
            username="abhinand",
            password="hashed_pwd",
            email="luckyman@gmail.com",
            district="Ernakulam"
            )
    delete(USER, key="luckyman@gmail.com")
