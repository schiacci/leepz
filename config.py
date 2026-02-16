"""
Configuration management for LEAP Strategic Asset Engine
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class OpenRouterConfig(BaseModel):
    """OpenRouter API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    grok_model: str = Field(default_factory=lambda: os.getenv("GROK_MODEL", "x-ai/grok-beta"))
    deepseek_r1_model: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_R1_MODEL", "deepseek/deepseek-r1"))
    deepseek_v3_model: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_V3_MODEL", "deepseek/deepseek-chat"))


class GoogleAIConfig(BaseModel):
    """Google AI (Gemini) configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_AI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))


class LEAPHeuristics(BaseModel):
    """Trading parameters for LEAP options strategy"""
    min_days_to_expiry: int = Field(default_factory=lambda: int(os.getenv("MIN_DAYS_TO_EXPIRY", "540")))
    target_delta: float = Field(default_factory=lambda: float(os.getenv("TARGET_DELTA", "0.80")))
    exit_day: int = Field(default_factory=lambda: int(os.getenv("EXIT_DAY", "365")))
    max_iv_percentile: float = Field(default_factory=lambda: float(os.getenv("MAX_IV_PERCENTILE", "40")))
    max_bid_ask_spread_pct: float = Field(default_factory=lambda: float(os.getenv("MAX_BID_ASK_SPREAD_PCT", "5.0")))


class DatabaseConfig(BaseModel):
    """Database configuration"""
    path: Path = Field(default_factory=lambda: Path(os.getenv("DATABASE_PATH", "./database/leap_engine.db")))


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: Path = Field(default_factory=lambda: Path(os.getenv("LOG_FILE", "./logs/leap_engine.log")))


class Config(BaseModel):
    """Master configuration class"""
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    google_ai: GoogleAIConfig = Field(default_factory=GoogleAIConfig)
    leap_heuristics: LEAPHeuristics = Field(default_factory=LEAPHeuristics)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    backtesting_mode: bool = Field(default=False)  # NEW: Enable backtesting mode
    backtesting_date: Optional[datetime] = Field(default=None)  # NEW: Historical date for temporal constraints

    def validate_api_keys(self) -> bool:
        """Validate that required API keys are present"""
        if self.openrouter.api_key:
            print("✅ Using OpenRouter API")
            return True
        elif self.google_ai.api_key:
            print("✅ Using Google AI (Gemini) - Free for development!")
            return True
        else:
            raise ValueError("No API key found. Set either OPENROUTER_API_KEY or GOOGLE_AI_API_KEY in environment variables")
        return True


# Global configuration instance
config = Config()
