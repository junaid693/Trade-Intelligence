"""Exception hierarchy for the Trade Intelligence feature engine."""


class FeatureError(Exception):
    """Base exception for all feature-engine errors."""


class FeatureValidationError(FeatureError):
    """Base exception for input validation errors."""


class EmptySequenceError(FeatureValidationError):
    """Raised when an empty candle sequence is provided."""


class UnsortedSequenceError(FeatureValidationError):
    """Raised when a candle sequence is not strictly monotonically increasing by open_time."""


class DuplicateTimestampError(FeatureValidationError):
    """Raised when duplicate open_time timestamps exist in the sequence."""


class MalformedIntervalSpacingError(FeatureValidationError):
    """Raised when consecutive candles have spacing less than the expected interval duration."""


class FormingCandleError(FeatureValidationError):
    """Raised when a candle is forming / unfinalized (close_time >= server_time)."""


class CorruptedCandleError(FeatureValidationError):
    """Raised when OHLC relationships are physically impossible (e.g. High < Low, High < Open)."""


class CorruptedCandleValueError(FeatureValidationError):
    """Raised when price/volume values are non-finite (NaN, Inf) or prices are non-positive."""


class InvalidIntervalError(FeatureValidationError):
    """Raised when an unsupported or invalid interval is supplied."""


class InvalidTimestampError(FeatureValidationError):
    """Raised when a timestamp is negative, non-integer, or boolean."""


class LookaheadViolationError(FeatureError):
    """Raised when an operation would violate point-in-time / no-lookahead invariants."""
