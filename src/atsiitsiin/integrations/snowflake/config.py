"""Snowflake configuration management."""

import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class SnowflakeConfig(BaseModel):
    """Pydantic model for Snowflake configuration validation."""

    # Connection parameters
    account: str = Field(..., description="Snowflake account identifier")
    user: str = Field(..., description="Snowflake username")
    warehouse: str = Field(..., description="Snowflake warehouse name")
    database: str = Field(..., description="Snowflake database name")
    schema_name: str = Field(..., description="Snowflake schema name")
    role: str = Field(..., description="Snowflake role name")

    # Authentication - Key pair (preferred)
    private_key_path: str | None = Field(
        None, description="Path to private key file"
    )
    private_key_passphrase: str | None = Field(
        None, description="Private key passphrase"
    )

    # Authentication - Password (alternative)
    password: str | None = Field(None, description="Snowflake password")

    # Connection settings
    connection_timeout: int = Field(
        default=60, ge=1, le=300, description="Connection timeout in seconds"
    )
    query_timeout: int = Field(
        default=300, ge=1, le=3600, description="Query timeout in seconds"
    )
    max_connections: int = Field(
        default=10, ge=1, le=100, description="Maximum number of connections"
    )

    # Additional settings
    application_name: str = Field(
        default="atsiitsiin", description="Application name for Snowflake"
    )
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("private_key_path")
    @classmethod
    def validate_private_key_path(cls, v):
        """Validate that private key file exists if provided."""
        if v is not None and v.strip():
            key_path = Path(v)
            if not key_path.exists():
                raise ValueError(f"Private key file not found: {v}")
            if not key_path.is_file():
                raise ValueError(f"Private key path is not a file: {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log level: {v}. Must be one of {valid_levels}"
            )
        return v.upper()

    def validate_auth_method(self) -> str:
        """Validate that at least one authentication method is provided.

        Returns:
            Authentication method: 'key_pair' or 'password'

        Raises:
            ValueError: If no valid authentication method is provided
        """
        has_key_pair = self.private_key_path is not None
        has_password = self.password is not None

        if has_key_pair and has_password:
            raise ValueError(
                "Cannot specify both key pair and password authentication"
            )

        if not has_key_pair and not has_password:
            raise ValueError(
                "Must specify either private_key_path or password for authentication"
            )

        return "key_pair" if has_key_pair else "password"

    def get_connection_parameters(self) -> dict:
        """Get connection parameters for Snowflake connection.

        Returns:
            Dictionary of connection parameters
        """
        auth_method = self.validate_auth_method()

        params: dict[str, str | bytes] = {
            "account": self.account,
            "user": self.user,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema_name,
            "role": self.role,
            "application": self.application_name,
            "log_level": self.log_level,
        }

        if auth_method == "key_pair":
            # Load private key
            private_key = self._load_private_key()
            params["private_key"] = private_key
        else:
            if self.password is not None:
                params["password"] = self.password

        return params

    def _load_private_key(self) -> bytes:
        """Load and return the private key.

        Returns:
            Private key as bytes

        Raises:
            ValueError: If private key cannot be loaded
        """
        if self.private_key_path is None:
            raise ValueError(
                "Private key path is required for key-pair authentication"
            )
        try:
            with open(self.private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=(
                        self.private_key_passphrase.encode()
                        if self.private_key_passphrase
                        else None
                    ),
                )

            # Convert to DER format for Snowflake
            private_key_der = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            return private_key_der

        except Exception as e:
            raise ValueError(
                f"Failed to load private key from {self.private_key_path}: {e}"
            ) from e

    @classmethod
    def from_env_file(
        cls, env_path: str | Path, var_names: list[str] | None = None
    ) -> "SnowflakeConfig":
        """Load configuration from .env file.

        Args:
            env_path: Path to .env file
            var_names: Optional list of specific environment variables to load

        Returns:
            Validated SnowflakeConfig instance
        """
        env_path = Path(env_path)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")

        # Load .env file
        load_dotenv(env_path, override=False)

        if var_names is None:
            var_names = [
                "SNOWFLAKE_ACCOUNT",
                "SNOWFLAKE_USER",
                "SNOWFLAKE_WAREHOUSE",
                "SNOWFLAKE_DATABASE",
                "SNOWFLAKE_SCHEMA",
                "SNOWFLAKE_ROLE",
                "SNOWFLAKE_PRIVATE_KEY_PATH",
                "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
                "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_CONNECTION_TIMEOUT",
                "SNOWFLAKE_QUERY_TIMEOUT",
                "SNOWFLAKE_MAX_CONNECTIONS",
                "SNOWFLAKE_APPLICATION_NAME",
                "SNOWFLAKE_LOG_LEVEL",
            ]

        # Extract environment variables
        config_dict: dict[str, Any] = {}
        for var_name in var_names:
            value = os.getenv(var_name)
            if value is not None:
                config_dict[var_name] = value

        # Map environment variable names to config field names
        field_mapping = {
            "SNOWFLAKE_ACCOUNT": "account",
            "SNOWFLAKE_USER": "user",
            "SNOWFLAKE_WAREHOUSE": "warehouse",
            "SNOWFLAKE_DATABASE": "database",
            "SNOWFLAKE_SCHEMA": "schema_name",
            "SNOWFLAKE_ROLE": "role",
            "SNOWFLAKE_PRIVATE_KEY_PATH": "private_key_path",
            "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE": "private_key_passphrase",
            "SNOWFLAKE_PASSWORD": "password",
            "SNOWFLAKE_CONNECTION_TIMEOUT": "connection_timeout",
            "SNOWFLAKE_QUERY_TIMEOUT": "query_timeout",
            "SNOWFLAKE_MAX_CONNECTIONS": "max_connections",
            "SNOWFLAKE_APPLICATION_NAME": "application_name",
            "SNOWFLAKE_LOG_LEVEL": "log_level",
        }

        # Convert environment variables to config fields
        mapped_config: dict[str, Any] = {}
        for env_var, field_name in field_mapping.items():
            if env_var in config_dict:
                value = config_dict[env_var]
                # Convert numeric strings to integers
                if field_name in [
                    "connection_timeout",
                    "query_timeout",
                    "max_connections",
                ]:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        pass
                mapped_config[field_name] = value

        return cls(**mapped_config)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()

    def __repr__(self) -> str:
        """String representation of the configuration."""
        auth_method = "key_pair" if self.private_key_path else "password"
        return f"SnowflakeConfig(account='{self.account}', user='{self.user}', auth='{auth_method}')"
