"""Snowflake connection management with key-pair authentication and connection pooling."""

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any

# Disable cloud metadata probing before importing Snowflake/boto3 dependencies.
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
os.environ.setdefault("AWS_METADATA_SERVICE_TIMEOUT", "1")
os.environ.setdefault("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
os.environ.setdefault("BOTO_DISABLE_IMDS", "true")

from snowflake.snowpark import Session  # type: ignore
from snowflake.snowpark.exceptions import SnowparkSQLException  # type: ignore

from .config import SnowflakeConfig


class SnowflakeConnection:
    """Manages Snowflake connections with pooling and authentication.

    This class provides connection management for Snowflake using Snowpark,
    with support for both key-pair and password authentication, connection
    pooling, and automatic session management.
    """

    def __init__(self, config: SnowflakeConfig):
        """Initialize the connection manager.

        Args:
            config: Snowflake configuration
        """
        self.config = config
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

        # Validate configuration
        self.config.validate_auth_method()

    @classmethod
    def from_env_file(cls, env_path: str) -> "SnowflakeConnection":
        """Create connection from .env file.

        Args:
            env_path: Path to .env file

        Returns:
            SnowflakeConnection instance
        """
        config = SnowflakeConfig.from_env_file(env_path)
        return cls(config)

    @classmethod
    def from_environment(cls) -> "SnowflakeConnection":
        """Create connection from environment variables.

        Returns:
            SnowflakeConnection instance
        """
        config = SnowflakeConfig.from_env_file(".env")
        return cls(config)

    def get_session(self, session_name: str = "default") -> Session:
        """Get or create a Snowpark session.

        Args:
            session_name: Name for the session (used for pooling)

        Returns:
            Snowpark Session instance

        Raises:
            SnowparkSQLException: If connection fails
        """
        with self._lock:
            if session_name in self._sessions:
                session = self._sessions[session_name]
                # Check if session is still valid
                try:
                    session.sql("SELECT 1").collect()
                    return session
                except Exception:
                    # Session is invalid, remove it
                    self._logger.warning(
                        f"Session '{session_name}' is invalid, recreating"
                    )
                    del self._sessions[session_name]

            # Create new session
            try:
                connection_params = self.config.get_connection_parameters()
                session = Session.builder.configs(connection_params).create()

                self._sessions[session_name] = session
                self._logger.info(
                    f"Created new Snowpark session: {session_name}"
                )

                return session

            except Exception as e:
                self._logger.error(f"Failed to create Snowpark session: {e}")
                raise SnowparkSQLException(f"Connection failed: {e}") from e

    def close_session(self, session_name: str = "default") -> None:
        """Close a specific session.

        Args:
            session_name: Name of the session to close
        """
        with self._lock:
            if session_name in self._sessions:
                try:
                    self._sessions[session_name].close()
                    self._logger.info(f"Closed session: {session_name}")
                except Exception as e:
                    self._logger.warning(
                        f"Error closing session '{session_name}': {e}"
                    )
                finally:
                    del self._sessions[session_name]

    def close_all_sessions(self) -> None:
        """Close all active sessions."""
        with self._lock:
            for session_name in list(self._sessions.keys()):
                self.close_session(session_name)

    @contextmanager
    def session_context(self, session_name: str = "default"):
        """Context manager for automatic session cleanup.

        Args:
            session_name: Name for the session

        Yields:
            Snowpark Session instance
        """
        session = None
        try:
            session = self.get_session(session_name)
            yield session
        finally:
            if session_name in self._sessions:
                # Don't close the session, just return it to the pool
                pass

    def test_connection(self) -> bool:
        """Test the connection to Snowflake.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.session_context() as session:
                result = session.sql("SELECT 1 as test").collect()
                return len(result) > 0 and result[0]["TEST"] == 1
        except Exception as e:
            self._logger.error(f"Connection test failed: {e}")
            return False

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the current connection.

        Returns:
            Dictionary with connection information
        """
        try:
            with self.session_context() as session:
                # Get current session info
                session_info = session.sql(
                    "SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE()"
                ).collect()[0]

                return {
                    "account": session_info[0],
                    "user": session_info[1],
                    "warehouse": session_info[2],
                    "database": session_info[3],
                    "schema": session_info[4],
                    "role": session_info[5],
                    "session_count": len(self._sessions),
                    "active_sessions": list(self._sessions.keys()),
                }
        except Exception as e:
            self._logger.error(f"Failed to get connection info: {e}")
            return {"error": str(e)}

    def execute_sql(self, sql: str, session_name: str = "default") -> Any:
        """Execute SQL query using a session.

        Args:
            sql: SQL query to execute
            session_name: Name of the session to use

        Returns:
            Query result
        """
        with self.session_context(session_name) as session:
            return session.sql(sql).collect()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all sessions."""
        self.close_all_sessions()

    def __repr__(self) -> str:
        """String representation of the connection."""
        return f"SnowflakeConnection(config={self.config}, sessions={len(self._sessions)})"
