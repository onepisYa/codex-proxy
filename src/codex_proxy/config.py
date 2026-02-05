import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --- Constants ---
DEFAULT_GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]

# Official gemini-cli credentials
# These are public-safe as per Google's own documentation for installed apps.
GEMINI_CLI_CLIENT_ID = (
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
)
GEMINI_CLI_CLIENT_SECRET = "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"


@dataclass
class Config:
    # Server
    host: str = "0.0.0.0"
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", 8765)))

    # Paths
    gemini_creds_path: str = os.path.expanduser("~/.gemini/oauth_creds.json")
    config_path: str = os.path.expanduser("~/.config/codex-proxy/config.json")

    # APIs
    z_ai_url: str = field(
        default_factory=lambda: os.environ.get(
            "Z_AI_URL", "https://api.z.ai/api/coding/paas/v4/chat/completions"
        )
    )
    gemini_api_internal: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_API_INTERNAL", "https://cloudcode-pa.googleapis.com"
        )
    )
    gemini_api_public: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_API_PUBLIC", "https://generativelanguage.googleapis.com"
        )
    )

    # Model Configs
    default_thinking_budget: int = 8192
    default_thinking_level: str = "HIGH"

    # Models
    gemini_models: List[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.environ.get(
                "GEMINI_MODELS", ",".join(DEFAULT_GEMINI_MODELS)
            ).split(",")
            if m.strip()
        ]
    )
    default_personality: str = "pragmatic"

    # Auth Defaults
    client_id: str = field(
        default_factory=lambda: os.environ.get("GEMINI_CLIENT_ID", GEMINI_CLI_CLIENT_ID)
    )
    client_secret: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_CLIENT_SECRET", GEMINI_CLI_CLIENT_SECRET
        )
    )
    z_ai_api_key: str = field(
        default_factory=lambda: os.environ.get("Z_AI_API_KEY", "")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "DEBUG").upper()
    )
    debug_mode: bool = field(
        default_factory=lambda: os.environ.get("DEBUG", "true").lower() == "true"
    )

    def __post_init__(self):
        self._load_from_file()

    def _load_from_file(self):
        """Override defaults from config file if it exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    file_config = json.load(f)

                    if not os.environ.get("GEMINI_CLIENT_ID"):
                        self.client_id = file_config.get("client_id", self.client_id)
                    if not os.environ.get("GEMINI_CLIENT_SECRET"):
                        self.client_secret = file_config.get(
                            "client_secret", self.client_secret
                        )
                    if not os.environ.get("Z_AI_API_KEY"):
                        self.z_ai_api_key = file_config.get(
                            "z_ai_api_key", self.z_ai_api_key
                        )
                    if not os.environ.get("GEMINI_API_KEY"):
                        self.gemini_api_key = file_config.get(
                            "gemini_api_key", self.gemini_api_key
                        )
                    if not os.environ.get("PORT"):
                        self.port = file_config.get("port", self.port)
                    if not os.environ.get("GEMINI_MODELS"):
                        self.gemini_models = file_config.get(
                            "gemini_models", self.gemini_models
                        )
                    if not os.environ.get("Z_AI_URL"):
                        self.z_ai_url = file_config.get("z_ai_url", self.z_ai_url)
                    if not os.environ.get("GEMINI_API_INTERNAL"):
                        self.gemini_api_internal = file_config.get(
                            "gemini_api_internal", self.gemini_api_internal
                        )
                    if not os.environ.get("GEMINI_API_PUBLIC"):
                        self.gemini_api_public = file_config.get(
                            "gemini_api_public", self.gemini_api_public
                        )
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")


# Global Config Instance
config = Config()
