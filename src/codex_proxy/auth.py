import json
import time
import logging
import os
import threading
from typing import Optional, Dict, Any, Tuple

from .config import config
from .utils import create_session

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Custom exception for authentication related errors."""

    pass


class GeminiAuth:
    """Manages Gemini OAuth2 credentials and Project ID mapping."""

    def __init__(self):
        self.session = create_session()
        self._cached_creds: Optional[str] = None
        self._creds_expiry: int = 0
        self._lock = threading.Lock()
        self._cached_project_id: Optional[str] = None

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Retrieve a valid access token, refreshing if necessary."""
        with self._lock:
            if not force_refresh and self._is_cached_token_valid():
                return self._cached_creds  # type: ignore

            creds_data = self._load_creds_from_file()

            # Check if file token is still valid
            file_token = creds_data.get("access_token")
            file_expiry = creds_data.get("expiry_date", 0)

            if not force_refresh and file_token and self._is_valid(file_expiry):
                self._cached_creds = file_token
                self._creds_expiry = file_expiry
                return file_token

            return self._refresh_token(creds_data)

    def _is_cached_token_valid(self) -> bool:
        return bool(self._cached_creds and self._is_valid(self._creds_expiry))

    def _is_valid(self, expiry_ms: int) -> bool:
        """Check if token is valid for at least another 5 minutes."""
        return (time.time() * 1000) < (expiry_ms - 300000)

    def _load_creds_from_file(self) -> Dict[str, Any]:
        if not os.path.exists(config.gemini_creds_path):
            raise AuthError(f"Credentials file missing: {config.gemini_creds_path}")

        try:
            with open(config.gemini_creds_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise AuthError(f"Failed to read credentials: {e}")

    def _refresh_token(self, creds_data: Dict[str, Any]) -> str:
        """Perform OAuth2 refresh flow."""
        logger.info("Refreshing Gemini Access Token...")

        client_id = config.client_id
        client_secret = config.client_secret
        refresh_token = creds_data.get("refresh_token")

        if not all([client_id, client_secret, refresh_token]):
            raise AuthError(
                "Missing OAuth2 credentials (client_id, secret, or refresh_token)."
            )

        try:
            resp = self.session.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )
            resp.raise_for_status()
            new_tokens = resp.json()

            access_token = new_tokens["access_token"]
            expiry_date = int((time.time() + new_tokens["expires_in"]) * 1000)

            # Update local data and save
            creds_data["access_token"] = access_token
            creds_data["expiry_date"] = expiry_date

            with open(config.gemini_creds_path, "w") as f:
                json.dump(creds_data, f, indent=2)

            self._cached_creds = access_token
            self._creds_expiry = expiry_date
            return access_token

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthError(f"OAuth2 refresh failed: {e}")

    def get_project_id(self, token: str) -> str:
        """Retrieve the associated Google Cloud Project ID."""
        if self._cached_project_id:
            return self._cached_project_id

        try:
            project_info = self._fetch_project_info(token)

            pid = project_info.get("cloudaicompanionProject")
            if isinstance(pid, dict):
                pid = pid.get("id")

            if pid:
                self._cached_project_id = pid
                return pid

            # Trigger onboarding if project is missing
            logger.info("Gemini Project ID missing. Attempting onboarding...")
            tier_id = self._determine_tier(project_info)
            return self._onboard_user(token, tier_id)

        except Exception as e:
            raise AuthError(f"Failed to resolve Project ID: {e}")

    def _fetch_project_info(self, token: str) -> Dict[str, Any]:
        resp = self.session.post(
            f"{config.gemini_api_base}/v1internal:loadCodeAssist",
            json={"metadata": self._get_default_metadata()},
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "GeminiCLI/0.26.0 (linux; x64)",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _determine_tier(self, project_info: Dict[str, Any]) -> str:
        allowed_tiers = project_info.get("allowedTiers", [])
        for tier in allowed_tiers:
            if tier.get("isDefault"):
                return tier.get("id", "free-tier")
        return "free-tier"

    def _get_default_metadata(self) -> Dict[str, str]:
        return {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }

    def _onboard_user(self, token: str, tier_id: str) -> str:
        """Execute onboarding Long-Running Operation (LRO)."""
        try:
            resp = self.session.post(
                f"{config.gemini_api_base}/v1internal:onboardUser",
                json={"tierId": tier_id, "metadata": self._get_default_metadata()},
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "GeminiCLI/0.26.0 (linux; x64)",
                },
                timeout=10,
            )
            resp.raise_for_status()
            operation = resp.json()

            op_name = operation.get("name")
            if not op_name:
                raise AuthError("Onboarding failed: No operation name returned.")

            # Poll for completion
            result = self._poll_operation(token, op_name)

            pid = (
                result.get("response", {}).get("cloudaicompanionProject", {}).get("id")
            )
            if not pid:
                raise AuthError("Onboarding finished but no Project ID found.")

            self._cached_project_id = pid
            return pid

        except Exception as e:
            raise AuthError(f"Onboarding failed: {e}")

    def _poll_operation(
        self, token: str, op_name: str, timeout: int = 60
    ) -> Dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            resp = self.session.get(
                f"{config.gemini_api_base}/v1internal/{op_name}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            resp.raise_for_status()
            op_data = resp.json()

            if op_data.get("done"):
                if "error" in op_data:
                    raise AuthError(f"Operation failed: {op_data['error']}")
                return op_data

            time.sleep(2)
        raise AuthError("Operation timed out.")
