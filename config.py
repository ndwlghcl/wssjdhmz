import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Config(BaseModel):
    default_model: str
    default_provider: str
    api_key: str = Field(default="", repr=False)   # 敏感信息：repr 时不显示
    base_url: str = ""
    temperature: float
    max_tokens: Optional[int]
    embedding_model: str
    max_obs_tokens: int

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置。
        环境变量只在 Config 里读一次，其他模块一律从 Config 取，不再自己 getenv"""
        return cls(
            default_model=os.getenv("LLM_MODEL_ID", "gpt-3.5-turbo"),
            default_provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", ""),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "100")),
            max_obs_tokens=int(os.getenv("MAX_OBS_TOKENS", "1000")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（排除 api_key，避免被日志打印出去）"""
        return self.model_dump(exclude={"api_key"})
