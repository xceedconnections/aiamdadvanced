from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    server_id = Column(Integer, ForeignKey("vicidial_servers.id"), nullable=False)
    key_prefix = Column(String(16), nullable=False)
    key_hash = Column(String(128), nullable=False, unique=True, index=True)
    # Full secret for portal display/copy (auth still uses key_hash)
    key_value = Column(Text, default="")
    name = Column(String(128), default="default")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    notes = Column(Text, default="")

    server = relationship("VicidialServer", backref="api_keys")
