"""Technical feature extraction engine for Trade Intelligence (Phase 2)."""

from trade_intelligence.features.exceptions import (
    CorruptedCandleError,
    CorruptedCandleValueError,
    DuplicateTimestampError,
    EmptySequenceError,
    FeatureError,
    FeatureValidationError,
    FormingCandleError,
    InvalidIntervalError,
    InvalidTimestampError,
    LookaheadViolationError,
    MalformedIntervalSpacingError,
    UnsortedSequenceError,
)
from trade_intelligence.features.types import (
    CandleSegment,
    FeatureCandle,
    FeatureMatrix,
    FeatureMetadata,
    FeatureSpec,
    MomentumSpec,
    NumericalCandleArrays,
    StructureSpec,
    TrendSpec,
    VolatilitySpec,
    VolumeSpec,
)
from trade_intelligence.features.validation import (
    convert_candles_to_arrays,
    partition_contiguous_segments,
    validate_candle,
    validate_candle_sequence,
)

__all__ = [
    # Types & Models
    "FeatureCandle",
    "NumericalCandleArrays",
    "CandleSegment",
    "FeatureSpec",
    "StructureSpec",
    "TrendSpec",
    "MomentumSpec",
    "VolatilitySpec",
    "VolumeSpec",
    "FeatureMetadata",
    "FeatureMatrix",
    # Exceptions
    "FeatureError",
    "FeatureValidationError",
    "EmptySequenceError",
    "UnsortedSequenceError",
    "DuplicateTimestampError",
    "MalformedIntervalSpacingError",
    "FormingCandleError",
    "CorruptedCandleError",
    "CorruptedCandleValueError",
    "InvalidIntervalError",
    "InvalidTimestampError",
    "LookaheadViolationError",
    # Validation & Boundary Conversion
    "validate_candle",
    "validate_candle_sequence",
    "convert_candles_to_arrays",
    "partition_contiguous_segments",
]
