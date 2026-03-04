# schema.py - all the pydantic models for our ontology
# this is the single source of truth for entity types, claim types,
# evidence structure, etc. everything else imports from here
#
# schema version: v1.0

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
import uuid


# --- entity + claim enums ---

class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TOPIC = "topic"
    MEETING = "meeting"


class ClaimType(str, Enum):
    WORKS_AT = "works_at"
    REPORTS_TO = "reports_to"
    WORKS_ON = "works_on"
    PART_OF = "part_of"
    DISCUSSED_IN = "discussed_in"
    DECIDED = "decided"
    COMMUNICATED_WITH = "communicated_with"
    ATTENDED = "attended"
    ASSIGNED_TO = "assigned_to"
    MENTIONED_IN = "mentioned_in"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


# statuses - we track if a claim might be outdated or disputed
class ClaimStatus(str, Enum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    DISPUTED = "disputed"


# --- evidence: the proof behind every claim ---

class EvidenceLocation(BaseModel):
    """char offsets in the email body where we found the evidence"""
    start_offset: int
    end_offset: int


class Evidence(BaseModel):
    """one piece of evidence backing a claim - has the actual email text excerpt"""
    source_id: str  # email Message-ID header
    artifact_id: str  # our internal uuid for the artifact
    excerpt: str  # the exact text that supports the claim
    location: Optional[EvidenceLocation] = None
    timestamp: Optional[datetime] = None  # when email was sent
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(ge=0.0, le=1.0)


# --- entity model ---

class MergeRecord(BaseModel):
    """tracks when two entities got merged during dedup"""
    merged_from_id: str
    merged_into_id: str
    merge_reason: str
    merged_at: datetime = Field(default_factory=datetime.utcnow)
    reversible: bool = True


class Entity(BaseModel):
    """a distinct entity (person, org, project, etc) in our graph"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    canonical_name: str  # the primary name we display
    aliases: List[str] = Field(default_factory=list)  # all other names we've seen
    properties: dict = Field(default_factory=dict)  # extra metadata
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    merge_history: List[MergeRecord] = Field(default_factory=list)


# --- claim: connects two entities with evidence ---

class Claim(BaseModel):
    """a factual statement we extracted - always grounded in evidence"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_type: ClaimType
    subject_id: str  # source entity uuid
    subject_name: str  # for readability
    object_id: str   # target entity uuid
    object_name: str  # for readability
    detail: Optional[str] = None  # free-text extra info
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: ClaimStatus = ClaimStatus.ACTIVE
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    extraction_version: str = "v1.0"


# --- artifact: a source email ---

class Artifact(BaseModel):
    """represents one email from the corpus"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str  # email Message-ID header
    file_path: str   # path in original dataset
    sender: str
    sender_name: Optional[str] = None
    recipients: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: str = ""
    date: Optional[datetime] = None
    body: str = ""
    body_hash: str = ""  # sha256 of body for dedup
    is_forward: bool = False
    is_reply: bool = False
    parent_artifact_id: Optional[str] = None
    dedup_group: Optional[str] = None


# --- what the LLM gives us back per email ---

class ExtractedEntity(BaseModel):
    """entity as raw-extracted by the LLM from one email"""
    name: str
    entity_type: str  # person, organization, project, topic, meeting
    aliases: List[str] = Field(default_factory=list)
    properties: dict = Field(default_factory=dict)


class ExtractedClaim(BaseModel):
    """claim/relationship from one email, before dedup"""
    claim_type: str
    subject: str  # source entity name
    object: str = ""  # target entity name
    detail: Optional[str] = None
    excerpt: str = ""  # exact text from the email backing this
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """everything we got from one email"""
    entities: List[ExtractedEntity] = Field(default_factory=list)
    claims: List[ExtractedClaim] = Field(default_factory=list)


# -- schema version + convenience lists

SCHEMA_VERSION = "v1.0"
ENTITY_TYPES = [e.value for e in EntityType]
CLAIM_TYPES = [c.value for c in ClaimType]
