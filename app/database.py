"""
Database connection setup using SQLAlchemy.

SQLAlchemy is an ORM (Object-Relational Mapper): it lets us work with Python
classes/objects instead of writing raw SQL, and it can talk to many databases
(SQLite, Postgres, MySQL...) through the same code. That is why we can develop
on a local SQLite file and deploy on Supabase Postgres by changing ONE line in
the .env file — the Python code does not change at all.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()  # read values from the .env file into environment variables

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./knowledge_tasks.db")

# SQLite needs one special flag when used with a multi-threaded web server.
# Postgres does not, so we add the flag only for SQLite.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# The "engine" is the actual connection pool to the database.
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# A session is one "conversation" with the database for a single request.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Every ORM model class will inherit from this Base.
Base = declarative_base()


def get_db():
    """
    FastAPI dependency: opens a DB session for a request and guarantees it is
    closed afterwards, even if the request errors. This prevents leaked
    connections.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
