"""Database-specific exception hierarchy for Trade Intelligence."""


class DatabaseError(Exception):
    """Base exception for all database-related errors."""


class DatabaseInitError(DatabaseError):
    """Raised when database initialization or schema creation fails."""


class DatabaseIntegrityError(DatabaseError):
    """Raised when a database integrity constraint is violated (FK, PK, UNIQUE)."""
