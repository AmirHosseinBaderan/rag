from pathlib import Path
from app.domain.documents.document import Document

class DocumentLoader:
    def load(self,path:Path) -> Document:
        content = path.read_text(encoding="utf-8")

        return Document(
            id=path.stem,
            content=content,
            metadata={
                "source":path.name
            }
        )