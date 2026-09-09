from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import deferred, relationship

from app.database import Base


class ScamCall(Base):
    """Full agent-leg recording uploaded for SCAM / fraud review (not AMD)."""

    __tablename__ = "scam_calls"

    id = Column(Integer, primary_key=True, index=True)
    server_id = Column(Integer, ForeignKey("vicidial_servers.id"), nullable=True, index=True)
    call_id = Column(String(128), default="", index=True)
    campaign = Column(String(64), default="")
    agent_user = Column(String(64), default="", index=True)
    caller_id = Column(String(64), default="", index=True)
    called_number = Column(String(64), default="", index=True)
    status = Column(String(32), default="PENDING", index=True)  # PENDING|CLEAN|SCAM|SPAM|ERROR
    confidence = Column(Float, default=0.0)
    audio_seconds = Column(Float, default=0.0)
    audio_path = Column(String(512), default="")
    audio_saved = Column(Boolean, default=False)
    audio_blob = deferred(Column(LargeBinary, nullable=True))
    transcript = Column(Text, default="")
    match_terms = Column(Text, default="")  # comma-separated hits
    details_json = Column(Text, default="{}")
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed_at = Column(DateTime, nullable=True)

    server = relationship("VicidialServer", lazy="joined")
