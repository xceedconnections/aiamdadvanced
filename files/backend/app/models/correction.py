from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.database import Base


class TrainingCorrection(Base):
    __tablename__ = "training_corrections"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("call_analyses.id"), nullable=False)
    ai_status = Column(String(32), nullable=False)
    corrected_status = Column(String(32), nullable=False)
    corrected_by = Column(String(64), default="")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
