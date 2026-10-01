
from typing import Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    role: MessageRole
    content:str
    timestamp: datetime=Field(default_factory=datetime.now)
    metadata: Dict[str, Any]=Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"