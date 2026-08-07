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

    # AMD mode for this dialer: global | classic | ml
    amd_mode = Column(String(16), default="global")

    # Custom Classic fields (when amd_mode=classic)
    confidence_gate_enabled = Column(Boolean, default=False)
    min_human_confidence_percent = Column(Integer, default=70)
    below_threshold_action = Column(String(32), default="MACHINE")

    # Custom ML fields (when amd_mode=ml) — legacy override flags kept in sync
    ml_pipeline_override_enabled = Column(Boolean, default=False)
    ml_pipeline_enabled = Column(Boolean, default=False)
    ml_whisper_enabled = Column(Boolean, default=True)
    ml_save_low_confidence = Column(Boolean, default=True)
    ml_min_human_confidence_percent = Column(Integer, default=85)
    ml_save_threshold_percent = Column(Integer, default=85)

    # Locale packs removed from UI; columns kept so old DBs don't break
    locale_pack_enabled = Column(Boolean, default=False)
    locale_pack = Column(String(32), default="usa")

    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
