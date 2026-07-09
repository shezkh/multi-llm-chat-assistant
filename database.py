import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Some hosts provide postgres:// but SQLAlchemy 2.x needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = None
SessionLocal = None

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(bind=engine)
        print("Connected to PostgreSQL")
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
        engine = None
        SessionLocal = None
else:
    print("No DATABASE_URL set, chat logging to DB is disabled")


def get_db():
    """Yield a DB session, or None if DB is unavailable."""
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
