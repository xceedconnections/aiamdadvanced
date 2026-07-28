from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ServerCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    description: str = ""
    timezone: str = "UTC"
    ip_whitelist: str = ""
    confidence_gate_enabled: bool = False
    min_human_confidence_percent: int = Field(70, ge=0, le=100)
    below_threshold_action: str = "MACHINE"
    locale_pack_enabled: bool = False
    locale_pack: str = "usa"


class ServerUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    timezone: Optional[str] = None
    ip_whitelist: Optional[str] = None
    is_active: Optional[bool] = None
    confidence_gate_enabled: Optional[bool] = None
    min_human_confidence_percent: Optional[int] = Field(None, ge=0, le=100)
    below_threshold_action: Optional[str] = None
    locale_pack_enabled: Optional[bool] = None
    locale_pack: Optional[str] = None


class ServerOut(BaseModel):
    id: int
    name: str
    description: str
    timezone: str
    ip_whitelist: str
    is_active: bool
    confidence_gate_enabled: bool = False
    min_human_confidence_percent: int = 70
    below_threshold_action: str = "MACHINE"
    locale_pack_enabled: bool = False
    locale_pack: str = "usa"
    last_seen: Optional[datetime]
    created_at: datetime
    total_calls: int = 0
    calls_today: int = 0

    class Config:
        from_attributes = True


class ApiKeyCreate(BaseModel):
    server_id: int
    name: str = "default"
    notes: str = ""


class ApiKeyOut(BaseModel):
    id: int
    server_id: int
    key_prefix: str
    name: str
    is_active: bool
    created_at: datetime
    last_used: Optional[datetime]
    notes: str
    # Only returned once on create:
    api_key: Optional[str] = None

    class Config:
        from_attributes = True


class AnalyzeMeta(BaseModel):
    callid: str
    campaign: str = ""
    caller: str = ""
    ani: str = ""


class AnalyzeResponse(BaseModel):
    status: str
    confidence: float
    processing_ms: int
    callid: str
    analysis_id: Optional[int] = None
    details: dict = {}


class CallOut(BaseModel):
    id: int
    server_id: Optional[int]
    call_id: str
    campaign: str
    caller_id: str
    called_number: str = ""
    ani: str
    status: str
    raw_status: str = ""
    confidence: float
    processing_ms: int
    audio_seconds: float
    created_at: datetime
    server_name: Optional[str] = None
    has_recording: bool = False
    recording_filename: Optional[str] = None
    recording_bytes: int = 0
    audio_path: str = ""

    class Config:
        from_attributes = True


class CdrPageOut(BaseModel):
    total: int
    page: int
    page_size: int
    rows: List[CallOut]


class DashboardStats(BaseModel):
    total_calls_today: int
    human: int
    machine: int
    ivr: int
    fax: int
    sit: int
    errors: int
    avg_processing_ms: float
    avg_confidence: float
    active_servers: int
    total_servers: int


class ServerReport(BaseModel):
    server_id: int
    server_name: str
    total: int
    human: int
    machine: int
    ivr: int
    fax: int
    sit: int
    errors: int
    avg_processing_ms: float
    avg_confidence: float
    accuracy: Optional[float] = None


class CorrectionCreate(BaseModel):
    call_analysis_id: int
    corrected_status: str
    notes: str = ""


class HealthOut(BaseModel):
    status: str
    version: str
    database: str
    redis: str
