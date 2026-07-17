from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import deferred, relationship

from app.database import Base


class CallAnalysis(Base):
    __tablename__ = "call_analyses"

    id = Column(Integer, primary_key=True, index=True)
    server_id = Column(Integer, ForeignKey("vicidial_servers.id"), nullable=True, index=True)
    call_id = Column(String(128), index=True, nullable=False)
    campaign = Column(String(128), default="", index=True)
    caller_id = Column(String(64), default="")
    called_number = Column(String(64), default="")
    ani = Column(String(64), default="")
    status = Column(String(32), nullable=False, index=True)  # final status sent to VICIdial
    raw_status = Column(String(32), default="", index=True)  # original engine status before gate
    confidence = Column(Float, default=0.0)
    processing_ms = Column(Integer, default=0)
    audio_seconds = Column(Float, default=0.0)
    audio_path = Column(String(512), default="")
    audio_saved = Column(Boolean, default=False, index=True)
    # Deferred so /api/live does not pull multi-KB blobs for every row
    audio_blob = deferred(Column(LargeBinary, nullable=True))
    features_json = Column(Text, default="{}")
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    server = relationship("VicidialServer", backref="calls")
