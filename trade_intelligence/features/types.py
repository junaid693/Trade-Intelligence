"""Data models, specifications, and typed containers for the feature engine."""

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _canonicalize_value(val: Any) -> Any:
    """Recursively normalize values for canonical dictionary serialization."""
    if isinstance(val, dict):
        return {str(k): _canonicalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [_canonicalize_value(item) for item in val]
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, str)):
        return val
    if isinstance(val, float):
        if not np.isfinite(val):
            raise ValueError(f"Non-finite float value in specification: {val}")
        return val
    if isinstance(val, Decimal):
        if not val.is_finite():
            raise ValueError(f"Non-finite Decimal value in specification: {val}")
        return str(val)
    if hasattr(val, "value"):  # Enum support
        return val.value
    return str(val)


@dataclass(frozen=True)
class FeatureCandle:
    """Structured representation of a single candlestick for feature extraction.

    Prices and volumes are preserved as exact finite ``Decimal`` instances.
    Timestamps are non-negative integer epoch milliseconds.
    """

    open_time: int
    close_time: int
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_asset_volume: Decimal
    number_of_trades: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    symbol: Optional[str] = None
    interval: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "FeatureCandle":
        """Construct a FeatureCandle from a database klines row dictionary."""
        return cls(
            open_time=int(row["open_time"]),
            close_time=int(row["close_time"]),
            open_price=Decimal(str(row["open_price"])),
            high_price=Decimal(str(row["high_price"])),
            low_price=Decimal(str(row["low_price"])),
            close_price=Decimal(str(row["close_price"])),
            volume=Decimal(str(row["volume"])),
            quote_asset_volume=Decimal(str(row["quote_asset_volume"])),
            number_of_trades=int(row["number_of_trades"]),
            taker_buy_base_volume=Decimal(str(row["taker_buy_base_volume"])),
            taker_buy_quote_volume=Decimal(str(row["taker_buy_quote_volume"])),
            symbol=row.get("symbol"),
            interval=row.get("interval"),
        )


@dataclass(frozen=True)
class NumericalCandleArrays:
    """Contiguous homogeneous 1D NumPy arrays at the numerical conversion boundary.

    Guarantees that all price and volume arrays are 1D ``np.float64`` arrays,
    and timestamps and trade counts are 1D ``np.int64`` arrays.
    """

    symbol: str
    interval: str
    length: int
    open_times: np.ndarray
    close_times: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    quote_volumes: np.ndarray
    trades: np.ndarray
    taker_buy_base: np.ndarray
    taker_buy_quote: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.length, int) or isinstance(self.length, bool) or self.length < 0:
            raise TypeError(f"length must be non-negative int, got {self.length!r}")

        for name, arr, expected_dtype in [
            ("open_times", self.open_times, np.int64),
            ("close_times", self.close_times, np.int64),
            ("opens", self.opens, np.float64),
            ("highs", self.highs, np.float64),
            ("lows", self.lows, np.float64),
            ("closes", self.closes, np.float64),
            ("volumes", self.volumes, np.float64),
            ("quote_volumes", self.quote_volumes, np.float64),
            ("trades", self.trades, np.int64),
            ("taker_buy_base", self.taker_buy_base, np.float64),
            ("taker_buy_quote", self.taker_buy_quote, np.float64),
        ]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-dimensional, got ndim={arr.ndim}")
            if arr.dtype != expected_dtype:
                raise TypeError(f"{name} dtype must be {expected_dtype}, got {arr.dtype}")
            if len(arr) != self.length:
                raise ValueError(f"{name} length {len(arr)} does not match container length {self.length}")


@dataclass(frozen=True)
class CandleSegment:
    """A contiguous slice of candles where each step is exactly interval_ms.

    Constructed by partition_contiguous_segments.
    """

    symbol: str
    interval: str
    candles: Tuple[FeatureCandle, ...]
    start_open_time: int
    end_open_time: int
    length: int

    def __post_init__(self) -> None:
        if len(self.candles) != self.length:
            raise ValueError(f"candles count {len(self.candles)} does not match length {self.length}")


@dataclass(frozen=True)
class FeatureSpec:
    """Canonical, immutable specification defining a feature calculation.

    Includes every calculation-affecting attribute in its canonical identity
    and SHA-256 spec_hash.
    """

    name: str
    version: str = "1.0.0"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("FeatureSpec name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("FeatureSpec version must be a non-empty string")
        if not isinstance(self.params, dict):
            raise TypeError(f"params must be a dict, got {type(self.params).__name__}")

    def canonical_dict(self) -> Dict[str, Any]:
        """Return a sorted, normalized dictionary representing the specification identity."""
        return {
            "name": self.name.strip(),
            "params": _canonicalize_value(self.params),
            "version": self.version.strip(),
        }

    @property
    def spec_hash(self) -> str:
        """Deterministic 64-character hexadecimal SHA-256 hash of canonical specification identity."""
        canonical = self.canonical_dict()
        serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructureSpec:
    """Specification for Price Structure features.

    Locked default: k = 2 (5-bar comparison window: 2 bars before, 2 bars after).
    Strict inequality: If an equal high/low exists in the comparison window, NO pivot is confirmed.
    """

    name: str = "price_structure"
    version: str = "1.0.0"
    pivot_k: int = 2
    strict_inequality: bool = True
    range_periods: Tuple[int, ...] = (20, 50)

    def __post_init__(self) -> None:
        if isinstance(self.pivot_k, bool) or not isinstance(self.pivot_k, int):
            raise TypeError(f"pivot_k must be integer, got {type(self.pivot_k).__name__}")
        if self.pivot_k < 1:
            raise ValueError(f"pivot_k must be >= 1, got {self.pivot_k}")
        if not isinstance(self.strict_inequality, bool):
            raise TypeError(f"strict_inequality must be bool, got {type(self.strict_inequality).__name__}")
        if not self.range_periods or not isinstance(self.range_periods, tuple):
            raise TypeError("range_periods must be a non-empty tuple of integers")
        for rp in self.range_periods:
            if isinstance(rp, bool) or not isinstance(rp, int) or rp < 1:
                raise ValueError(f"Each range_period must be a positive integer, got {rp!r}")

    def to_feature_spec(self) -> FeatureSpec:
        """Convert to canonical FeatureSpec."""
        return FeatureSpec(
            name=self.name,
            version=self.version,
            params={
                "pivot_k": self.pivot_k,
                "range_periods": list(self.range_periods),
                "strict_inequality": self.strict_inequality,
                "window_size": 2 * self.pivot_k + 1,
            },
        )


@dataclass(frozen=True)
class TrendSpec:
    """Specification for Trend features (EMAs, Spreads, Normalized Slope).

    Locked default: slope_k = 3 bars lookback delta.
    """

    name: str = "trend"
    version: str = "1.0.0"
    ema_periods: Tuple[int, ...] = (20, 50, 200)
    source: str = "close"
    smoothing: str = "exponential"
    seed_convention: str = "sma"
    slope_k: int = 3
    warmup_multiplier: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.slope_k, bool) or not isinstance(self.slope_k, int) or self.slope_k < 1:
            raise ValueError(f"slope_k must be positive int, got {self.slope_k!r}")
        if isinstance(self.warmup_multiplier, bool) or not isinstance(self.warmup_multiplier, int) or self.warmup_multiplier < 1:
            raise ValueError(f"warmup_multiplier must be positive int, got {self.warmup_multiplier!r}")
        for p in self.ema_periods:
            if isinstance(p, bool) or not isinstance(p, int) or p < 1:
                raise ValueError(f"Invalid EMA period: {p!r}")

    def to_feature_spec(self) -> FeatureSpec:
        """Convert to canonical FeatureSpec."""
        return FeatureSpec(
            name=self.name,
            version=self.version,
            params={
                "ema_periods": list(self.ema_periods),
                "seed_convention": self.seed_convention,
                "slope_k": self.slope_k,
                "smoothing": self.smoothing,
                "source": self.source,
                "warmup_multiplier": self.warmup_multiplier,
            },
        )


@dataclass(frozen=True)
class MomentumSpec:
    """Specification for Momentum features (RSI, MACD, ROC)."""

    name: str = "momentum"
    version: str = "1.0.0"
    rsi_period: int = 14
    source: str = "close"
    smoothing: str = "wilder"
    seed_convention: str = "sma"
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    roc_periods: Tuple[int, ...] = (10, 21)
    warmup_multiplier: int = 3

    def __post_init__(self) -> None:
        for p_name, p_val in [
            ("rsi_period", self.rsi_period),
            ("macd_fast", self.macd_fast),
            ("macd_slow", self.macd_slow),
            ("macd_signal", self.macd_signal),
        ]:
            if isinstance(p_val, bool) or not isinstance(p_val, int) or p_val < 1:
                raise ValueError(f"{p_name} must be a positive int, got {p_val!r}")
        if self.macd_fast >= self.macd_slow:
            raise ValueError(f"macd_fast ({self.macd_fast}) must be < macd_slow ({self.macd_slow})")

    def to_feature_spec(self) -> FeatureSpec:
        """Convert to canonical FeatureSpec."""
        return FeatureSpec(
            name=self.name,
            version=self.version,
            params={
                "macd_fast": self.macd_fast,
                "macd_signal": self.macd_signal,
                "macd_slow": self.macd_slow,
                "roc_periods": list(self.roc_periods),
                "rsi_period": self.rsi_period,
                "seed_convention": self.seed_convention,
                "smoothing": self.smoothing,
                "source": self.source,
                "warmup_multiplier": self.warmup_multiplier,
            },
        )


@dataclass(frozen=True)
class VolatilitySpec:
    """Specification for Volatility features (ATR, NATR, Bollinger Bands).

    Locked: Population standard deviation (ddof = 0).
    """

    name: str = "volatility"
    version: str = "1.0.0"
    atr_period: int = 14
    smoothing: str = "wilder"
    bb_period: int = 20
    bb_std_mult: float = 2.0
    ddof: int = 0
    warmup_multiplier: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.atr_period, bool) or not isinstance(self.atr_period, int) or self.atr_period < 1:
            raise ValueError(f"atr_period must be a positive int, got {self.atr_period!r}")
        if isinstance(self.bb_period, bool) or not isinstance(self.bb_period, int) or self.bb_period < 1:
            raise ValueError(f"bb_period must be a positive int, got {self.bb_period!r}")
        if isinstance(self.bb_std_mult, bool) or not isinstance(self.bb_std_mult, (int, float)) or self.bb_std_mult <= 0:
            raise ValueError(f"bb_std_mult must be positive float, got {self.bb_std_mult!r}")
        if isinstance(self.ddof, bool) or not isinstance(self.ddof, int) or self.ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {self.ddof!r}")

    def to_feature_spec(self) -> FeatureSpec:
        """Convert to canonical FeatureSpec."""
        return FeatureSpec(
            name=self.name,
            version=self.version,
            params={
                "atr_period": self.atr_period,
                "bb_period": self.bb_period,
                "bb_std_mult": float(self.bb_std_mult),
                "ddof": self.ddof,
                "smoothing": self.smoothing,
                "warmup_multiplier": self.warmup_multiplier,
            },
        )


@dataclass(frozen=True)
class VolumeSpec:
    """Specification for Volume features (Volume SMA, RVOL, Taker-Buy Volume Share)."""

    name: str = "volume"
    version: str = "1.0.0"
    volume_sma_period: int = 20
    rvol_period: int = 20
    seed_convention: str = "sma"
    neutral_taker_share: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.volume_sma_period, bool) or not isinstance(self.volume_sma_period, int) or self.volume_sma_period < 1:
            raise ValueError(f"volume_sma_period must be positive int, got {self.volume_sma_period!r}")
        if isinstance(self.rvol_period, bool) or not isinstance(self.rvol_period, int) or self.rvol_period < 1:
            raise ValueError(f"rvol_period must be positive int, got {self.rvol_period!r}")
        if not isinstance(self.neutral_taker_share, (int, float)) or not (0.0 <= self.neutral_taker_share <= 1.0):
            raise ValueError(f"neutral_taker_share must be in [0.0, 1.0], got {self.neutral_taker_share!r}")

    def to_feature_spec(self) -> FeatureSpec:
        """Convert to canonical FeatureSpec."""
        return FeatureSpec(
            name=self.name,
            version=self.version,
            params={
                "neutral_taker_share": float(self.neutral_taker_share),
                "rvol_period": self.rvol_period,
                "seed_convention": self.seed_convention,
                "volume_sma_period": self.volume_sma_period,
            },
        )


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata accompanying a calculated FeatureMatrix."""

    symbol: str
    interval: str
    start_time: int
    end_time: int
    candle_count: int
    segment_count: int
    is_warmed_up: np.ndarray
    is_gap_boundary: np.ndarray
    spec_version: str
    spec_hash: str
    generated_at_ms: int

    def __post_init__(self) -> None:
        for name, arr in [("is_warmed_up", self.is_warmed_up), ("is_gap_boundary", self.is_gap_boundary)]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
            if arr.ndim != 1 or arr.dtype != bool:
                raise TypeError(f"{name} must be a 1D boolean array, got ndim={arr.ndim}, dtype={arr.dtype}")
            if len(arr) != self.candle_count:
                raise ValueError(f"{name} length {len(arr)} does not match candle_count {self.candle_count}")


@dataclass(frozen=True)
class FeatureMatrix:
    """Dedicated typed container for extracted technical features.

    Stores named 1D numpy.float64 numerical arrays and optional integer/state
    arrays alongside metadata. No pandas or polars dependencies.
    """

    symbol: str
    interval: str
    open_times: np.ndarray
    close_times: np.ndarray
    features: Dict[str, np.ndarray]
    state_features: Dict[str, np.ndarray]
    metadata: FeatureMetadata

    def __post_init__(self) -> None:
        count = len(self.open_times)
        if len(self.close_times) != count:
            raise ValueError(f"close_times length {len(self.close_times)} does not match open_times length {count}")
        if self.open_times.ndim != 1 or self.open_times.dtype != np.int64:
            raise TypeError("open_times must be a 1D int64 array")
        if self.close_times.ndim != 1 or self.close_times.dtype != np.int64:
            raise TypeError("close_times must be a 1D int64 array")

        for fname, farr in self.features.items():
            if not isinstance(farr, np.ndarray):
                raise TypeError(f"feature {fname} must be a numpy.ndarray, got {type(farr).__name__}")
            if farr.ndim != 1:
                raise ValueError(f"feature {fname} must be 1D, got ndim={farr.ndim}")
            if farr.dtype != np.float64:
                raise TypeError(f"numerical feature {fname} dtype must be float64, got {farr.dtype}")
            if len(farr) != count:
                raise ValueError(f"feature {fname} length {len(farr)} does not match matrix length {count}")

        for sname, sarr in self.state_features.items():
            if not isinstance(sarr, np.ndarray):
                raise TypeError(f"state feature {sname} must be a numpy.ndarray, got {type(sarr).__name__}")
            if sarr.ndim != 1:
                raise ValueError(f"state feature {sname} must be 1D, got ndim={sarr.ndim}")
            if len(sarr) != count:
                raise ValueError(f"state feature {sname} length {len(sarr)} does not match matrix length {count}")

    def get_column(self, name: str) -> np.ndarray:
        """Retrieve a feature array by name."""
        if name in self.features:
            return self.features[name]
        if name in self.state_features:
            return self.state_features[name]
        if name == "open_times":
            return self.open_times
        if name == "close_times":
            return self.close_times
        raise KeyError(f"Feature column {name!r} not found in FeatureMatrix")

    def column_names(self) -> List[str]:
        """Return list of all available feature column names."""
        return ["open_times", "close_times"] + list(self.features.keys()) + list(self.state_features.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (row_count, total_column_count)."""
        return len(self.open_times), len(self.column_names())

    def __len__(self) -> int:
        return len(self.open_times)


# =============================================================================
# Unit 2.3 — Canonical Feature Constants & Assembly Contracts
# =============================================================================

CANONICAL_FEATURE_NAMES: Tuple[str, ...] = (
    # Trend (12)
    "ema20", "ema50", "ema200",
    "ratio_close_ema20", "ratio_close_ema50", "ratio_close_ema200",
    "spread_ema20_50", "spread_ema50_200", "spread_ema20_200",
    "slope_ema20", "slope_ema50", "slope_ema200",
    # Momentum (8)
    "rsi14",
    "macd_line", "macd_signal", "macd_hist",
    "macd_norm", "macd_hist_norm",
    "roc10", "roc21",
    # Volatility (7)
    "atr14", "natr14",
    "bb_upper", "bb_mid", "bb_lower", "bb_pct_b", "bb_bandwidth",
    # Volume (3)
    "vol_sma20", "rvol20", "taker_buy_share",
)

assert len(CANONICAL_FEATURE_NAMES) == 30, f"Expected 30 canonical features, got {len(CANONICAL_FEATURE_NAMES)}"

# Mathematical availability: the earliest 0-indexed row where a finite value
# can appear for each feature on a single contiguous segment.
FEATURE_FIRST_VALID_INDICES: Dict[str, int] = {
    # Trend — EMA(N): first valid at N-1
    "ema20": 19,
    "ema50": 49,
    "ema200": 199,
    # Ratios inherit the EMA availability
    "ratio_close_ema20": 19,
    "ratio_close_ema50": 49,
    "ratio_close_ema200": 199,
    # Spreads: max(fast, slow) availability
    "spread_ema20_50": 49,
    "spread_ema50_200": 199,
    "spread_ema20_200": 199,
    # Slopes: EMA availability + slope_k (3)
    "slope_ema20": 22,
    "slope_ema50": 52,
    "slope_ema200": 202,
    # RSI(14): first valid at index 14 (needs 15 candles / 14 deltas)
    "rsi14": 14,
    # MACD line: first valid at slow_period-1 = 25
    "macd_line": 25,
    # MACD signal: first valid at 25 + signal_period - 1 = 33
    "macd_signal": 33,
    "macd_hist": 33,
    # Normalized MACD: same as MACD line
    "macd_norm": 25,
    "macd_hist_norm": 33,
    # ROC(N): first valid at N
    "roc10": 10,
    "roc21": 21,
    # ATR(14): first valid at period = 14 (needs period+1 candles, seed uses TR[1:period+1])
    "atr14": 14,
    "natr14": 14,
    # Bollinger Bands(20): first valid at period-1 = 19
    "bb_upper": 19,
    "bb_mid": 19,
    "bb_lower": 19,
    "bb_pct_b": 19,
    "bb_bandwidth": 19,
    # Volume SMA(20): first valid at period-1 = 19
    "vol_sma20": 19,
    "rvol20": 19,
    # Taker buy share: valid from index 0 (point-in-time ratio)
    "taker_buy_share": 0,
}

# Stabilization / convergence warm-up threshold.
# For recursive (convergence-sensitive) indicators: 3 × N.
# For non-recursive (lookback-only) indicators: same as first_valid_index.
FEATURE_WARMUP_THRESHOLDS: Dict[str, int] = {
    # Recursive EMA-based (3 × period)
    "ema20": 60,
    "ema50": 150,
    "ema200": 600,
    # Ratios inherit EMA warm-up
    "ratio_close_ema20": 60,
    "ratio_close_ema50": 150,
    "ratio_close_ema200": 600,
    # Spreads: max warm-up of constituent EMAs
    "spread_ema20_50": 150,
    "spread_ema50_200": 600,
    "spread_ema20_200": 600,
    # Slopes: EMA warm-up + slope_k (3)
    "slope_ema20": 63,
    "slope_ema50": 153,
    "slope_ema200": 603,
    # RSI(14) recursive Wilder smoothing: 3 × 14 = 42
    "rsi14": 42,
    # MACD line: max of recursive EMAs = 3 × 26 = 78
    "macd_line": 78,
    # MACD signal: MACD warm-up + 3 × signal_period = 78 + 27 = 105
    "macd_signal": 105,
    "macd_hist": 105,
    # Normalized MACD: same as MACD line warm-up
    "macd_norm": 78,
    "macd_hist_norm": 105,
    # ROC is non-recursive: same as first_valid
    "roc10": 10,
    "roc21": 21,
    # ATR(14) recursive Wilder: 3 × 14 = 42
    "atr14": 42,
    "natr14": 42,
    # Bollinger Bands are non-recursive (rolling window): same as first_valid
    "bb_upper": 19,
    "bb_mid": 19,
    "bb_lower": 19,
    "bb_pct_b": 19,
    "bb_bandwidth": 19,
    # Volume SMA is non-recursive: same as first_valid
    "vol_sma20": 19,
    "rvol20": 19,
    # Taker buy share: non-recursive
    "taker_buy_share": 0,
}

# Ensure both dictionaries cover exactly the canonical set
assert set(FEATURE_FIRST_VALID_INDICES.keys()) == set(CANONICAL_FEATURE_NAMES)
assert set(FEATURE_WARMUP_THRESHOLDS.keys()) == set(CANONICAL_FEATURE_NAMES)


@dataclass(frozen=True)
class TechnicalFeaturesSpec:
    """Unified specification combining all four feature family specs.

    Provides a single ``spec_hash`` covering the entire 30-feature configuration,
    and per-family ``FeatureSpec`` objects for traceability.
    """

    trend: TrendSpec = field(default_factory=TrendSpec)
    momentum: MomentumSpec = field(default_factory=MomentumSpec)
    volatility: VolatilitySpec = field(default_factory=VolatilitySpec)
    volume: VolumeSpec = field(default_factory=VolumeSpec)
    version: str = "2.3.0"

    def family_specs(self) -> Dict[str, FeatureSpec]:
        """Return a dictionary of per-family FeatureSpec objects."""
        return {
            "trend": self.trend.to_feature_spec(),
            "momentum": self.momentum.to_feature_spec(),
            "volatility": self.volatility.to_feature_spec(),
            "volume": self.volume.to_feature_spec(),
        }

    def canonical_dict(self) -> Dict[str, Any]:
        """Return a sorted, normalized dictionary for deterministic hashing."""
        families = self.family_specs()
        return {
            "families": {k: _canonicalize_value(v.canonical_dict()) for k, v in sorted(families.items())},
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "version": self.version,
        }

    @property
    def spec_hash(self) -> str:
        """Deterministic 64-character hexadecimal SHA-256 hash."""
        canonical = self.canonical_dict()
        serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
