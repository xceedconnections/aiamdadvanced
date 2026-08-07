from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from app.database import Base


class VicidialServer(Base):
    __tablename__ = "vicidial_servers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    description = Column(Text, default="")
    timezone = Column(String(64), default="UTC")
    ip_whitelist = Column(Text, default="")  # comma-separated IPs, empty = allow all
    is_active = Column(Boolean, default=True)

    # Per-server classic AMD confidence gate (if disabled → global amd_settings.json)
    confidence_gate_enabled = Column(Boolean, default=False)
    min_human_confidence_percent = Column(Integer, default=70)
    below_threshold_action = Column(String(32), default="MACHINE")

    # Per-server ML pipeline override (if disabled → global ML settings)
    ml_pipeline_override_enabled = Column(Boolean, default=False)
    ml_pipeline_enabled = Column(Boolean, default=False)
    ml_whisper_enabled = Column(Boolean, default=True)
    ml_save_low_confidence = Column(Boolean, default=True)
    ml_min_human_confidence_percent = Column(Integer, default=85)
    ml_save_threshold_percent = Column(Integer, default=85)

    # Locale / market threshold pack (if disabled → default engine thresholds)
    locale_pack_enabled = Column(Boolean, default=False)
    locale_pack = Column(String(32), default="usa")  # usa | uk | multilingual

    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
