from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)                        # UUID session ID
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_activity = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
    language = Column(String(10), nullable=True)                 # en, tr, de, etc.
    ip_address = Column(String(45), nullable=True)               # supports IPv6
    user_agent = Column(Text, nullable=True)
    country = Column(String(100), nullable=True)                 # from IP geo (optional)
    message_count = Column(Integer, default=0)

    # Review & analytics
    outcome = Column(String(30), default="in_progress")          # in_progress | goal_reached | dropped_off
    topics_covered = Column(JSON, default=list)                  # e.g. ["pricing", "onboarding"]
    flagged = Column(Boolean, default=False)                     # flag for manual review
    notes = Column(Text, nullable=True)                          # admin notes

    messages = relationship("ChatMessage", back_populates="session",
                            order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String(10), nullable=False)                    # "user" or "model"
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")


class AppSettings(Base):
    __tablename__ = "app_settings"

    key = Column(String(50), primary_key=True)
    value = Column(String(200), nullable=False)
