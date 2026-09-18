from dataclasses import dataclass, field
from typing import Any

from app.domain.documents.document import Document


@dataclass
class DocumentResponse:
    id: str
    source: str
    size: int | None = None


@dataclass
class DocumentCreate:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentListResponse:
    documents: list[DocumentResponse]
