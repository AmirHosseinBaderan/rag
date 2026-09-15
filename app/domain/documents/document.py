from dataclasses import dataclass,field
from typing import Any

@dataclass(frozen=True)
class Document:
    id:str
    content:str
    metadata: dict[str,Any] = field(default_factory=dict)