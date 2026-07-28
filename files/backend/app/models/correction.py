from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text

from app.database import Base


class TrainingCorrection(Base):
    """Audit log of every teach / revert action."""

    __tablename__ = "training_corrections"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("call_analyses.id"), nullable=True, index=True)
    phone_number = Column(String(32), default="", index=True)
    ai_status = Column(String(32), nullable=False, default="")
    corrected_status = Column(String(32), nullable=False)
    previous_taught_status = Column(String(32), default="")
    action = Column(String(32), default="teach")  # teach | revert | import | wipe
    corrected_by = Column(String(64), default="")
    notes = Column(Text, default="")
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class TrainingOverride(Base):
    """
    Active phone → taught status map used at analyze time.
    One row per normalized phone number.
    """

    __tablename__ = "training_overrides"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(32), unique=True, nullable=False, index=True)
    taught_status = Column(String(32), nullable=False, index=True)
    source_call_id = Column(Integer, ForeignKey("call_analyses.id"), nullable=True)
    last_correction_id = Column(Integer, nullable=True)
    taught_by = Column(String(64), default="")
    notes = Column(Text, default="")
    hit_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
