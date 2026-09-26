"""Feature assembly layer — Unit 2.3.

Assembles the 30 Unit 2.2 kernel outputs into the authoritative FeatureMatrix,
preserving canonical feature ordering, dual-tier warm-up metadata, deterministic
spec hashing, and segment/gap isolation.

No indicator mathematics is performed here — all calculations are delegated to
the Unit 2.2 kernel functions via ``compute_all_kernels``.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.features.exceptions import FeatureAssemblyError
from trade_intelligence.features.kernels import compute_all_kernels
from trade_intelligence.features.types import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_FIRST_VALID_INDICES,
    FEATURE_WARMUP_THRESHOLDS,
    CandleSegment,
    FeatureCandle,
    FeatureMatrix,
    FeatureMetadata,
    NumericalCandleArrays,
    TechnicalFeaturesSpec,
)
from trade_intelligence.features.validation import (
    convert_candles_to_arrays,
    partition_contiguous_segments,
)


def _build_gap_boundary_mask(
    segment_lengths: List[int],
    total_length: int,
) -> np.ndarray:
    """Construct a 1D boolean mask marking the first bar of each segment after the first.

    Gap boundaries mark where indicator state must NOT carry over because the
    preceding bar belongs to a different contiguous segment.

    Args:
        segment_lengths: Ordered list of per-segment bar counts.
        total_length: Total number of bars across all segments.

    Returns:
        1D bool array of length ``total_length``.
    """
    mask = np.zeros(total_length, dtype=bool)
    offset = 0
    for i, seg_len in enumerate(segment_lengths):
        if i > 0:
            mask[offset] = True
        offset += seg_len
    return mask


def _build_warmup_mask(
    segment_lengths: List[int],
    total_length: int,
) -> np.ndarray:
    """Determine per-bar warm-up status across concatenated segments.

    A bar is considered warmed up when ALL 30 features have reached their
    stabilization threshold within the bar's own contiguous segment.

    The most demanding feature is ``slope_ema200`` with warmup_threshold = 603.
    Within each segment, bar ``j`` (0-indexed within segment) is warmed up
    when ``j >= max(FEATURE_WARMUP_THRESHOLDS.values())``.

    Args:
        segment_lengths: Ordered list of per-segment bar counts.
        total_length: Total number of bars across all segments.

    Returns:
        1D bool array of length ``total_length``.
    """
    max_threshold = max(FEATURE_WARMUP_THRESHOLDS.values())
    mask = np.zeros(total_length, dtype=bool)
    offset = 0
    for seg_len in segment_lengths:
        for j in range(seg_len):
            if j >= max_threshold:
                mask[offset + j] = True
        offset += seg_len
    return mask


def assemble_features_for_segment(
    arrays: NumericalCandleArrays,
) -> Dict[str, np.ndarray]:
    """Compute all 30 features for a single contiguous segment.

    Delegates to ``compute_all_kernels`` and enforces canonical ordering.

    Args:
        arrays: A validated, contiguous ``NumericalCandleArrays``.

    Returns:
        Ordered dictionary mapping each of the 30 canonical feature names
        to its 1D ``np.float64`` array.

    Raises:
        FeatureAssemblyError: If kernel output is missing features or
            contains unexpected names.
    """
    raw = compute_all_kernels(arrays)

    # Validate completeness
    missing = set(CANONICAL_FEATURE_NAMES) - set(raw.keys())
    if missing:
        raise FeatureAssemblyError(
            f"Kernel output missing features: {sorted(missing)}"
        )
    extra = set(raw.keys()) - set(CANONICAL_FEATURE_NAMES)
    if extra:
        raise FeatureAssemblyError(
            f"Kernel output contains unexpected features: {sorted(extra)}"
        )

    # Enforce canonical ordering
    ordered: Dict[str, np.ndarray] = {}
    for name in CANONICAL_FEATURE_NAMES:
        arr = raw[name]
        if arr.dtype != np.float64:
            raise FeatureAssemblyError(
                f"Feature {name!r} dtype must be float64, got {arr.dtype}"
            )
        if arr.ndim != 1 or len(arr) != arrays.length:
            raise FeatureAssemblyError(
                f"Feature {name!r} shape mismatch: expected ({arrays.length},), "
                f"got {arr.shape}"
            )
        ordered[name] = arr

    return ordered


def assemble_features_from_segments(
    segments: List[CandleSegment],
    symbol: str,
    interval: str,
    spec: Optional[TechnicalFeaturesSpec] = None,
) -> FeatureMatrix:
    """Assemble a FeatureMatrix from pre-partitioned contiguous CandleSegments.

    Each segment is computed independently — no indicator state carries across
    segment boundaries. The outputs are concatenated in segment order.

    Args:
        segments: Non-empty list of contiguous ``CandleSegment`` objects.
        symbol: Trading pair symbol (e.g. ``"BTCUSDT"``).
        interval: Kline interval string (e.g. ``"1h"``).
        spec: Optional ``TechnicalFeaturesSpec``; defaults are used if None.

    Returns:
        Fully assembled ``FeatureMatrix`` with metadata.

    Raises:
        FeatureAssemblyError: On empty segments, shape mismatches, or
            kernel failures.
    """
    if not segments:
        raise FeatureAssemblyError("Cannot assemble features from empty segment list")

    if spec is None:
        spec = TechnicalFeaturesSpec()

    segment_lengths: List[int] = []
    all_open_times: List[np.ndarray] = []
    all_close_times: List[np.ndarray] = []
    all_features: Dict[str, List[np.ndarray]] = {
        name: [] for name in CANONICAL_FEATURE_NAMES
    }

    for seg_idx, seg in enumerate(segments):
        if seg.length == 0:
            raise FeatureAssemblyError(
                f"Segment {seg_idx} has zero length"
            )

        # Convert CandleSegment to NumericalCandleArrays
        arrays = convert_candles_to_arrays(
            list(seg.candles), symbol=symbol, interval=interval
        )

        # Compute features for this isolated segment
        try:
            seg_features = assemble_features_for_segment(arrays)
        except Exception as e:
            raise FeatureAssemblyError(
                f"Feature computation failed for segment {seg_idx}: {e}"
            ) from e

        segment_lengths.append(arrays.length)
        all_open_times.append(arrays.open_times)
        all_close_times.append(arrays.close_times)

        for name in CANONICAL_FEATURE_NAMES:
            all_features[name].append(seg_features[name])

    # Concatenate across segments
    total_length = sum(segment_lengths)
    open_times = np.concatenate(all_open_times)
    close_times = np.concatenate(all_close_times)

    features: Dict[str, np.ndarray] = {}
    for name in CANONICAL_FEATURE_NAMES:
        features[name] = np.concatenate(all_features[name])

    # Build metadata arrays
    is_gap_boundary = _build_gap_boundary_mask(segment_lengths, total_length)
    is_warmed_up = _build_warmup_mask(segment_lengths, total_length)

    metadata = FeatureMetadata(
        symbol=symbol,
        interval=interval,
        start_time=int(open_times[0]),
        end_time=int(open_times[-1]),
        candle_count=total_length,
        segment_count=len(segments),
        is_warmed_up=is_warmed_up,
        is_gap_boundary=is_gap_boundary,
        spec_version=spec.version,
        spec_hash=spec.spec_hash,
        generated_at_ms=int(time.time() * 1000),
    )

    return FeatureMatrix(
        symbol=symbol,
        interval=interval,
        open_times=open_times,
        close_times=close_times,
        features=features,
        state_features={},
        metadata=metadata,
    )


def assemble_features_from_candles(
    candles: Sequence[Union[FeatureCandle, Dict[str, Any]]],
    interval: Union[str, KlineInterval],
    symbol: str = "UNKNOWN",
    server_time_ms: Optional[int] = None,
    spec: Optional[TechnicalFeaturesSpec] = None,
) -> FeatureMatrix:
    """End-to-end assembly from raw candle data to FeatureMatrix.

    Validates the candle sequence, partitions into contiguous segments,
    computes all 30 features per segment in isolation, and assembles
    the final matrix with full metadata.

    Args:
        candles: Sequence of candle data (``FeatureCandle`` or dicts).
        interval: Kline interval (string or ``KlineInterval`` enum).
        symbol: Trading pair symbol.
        server_time_ms: Optional server timestamp for forming-candle filtering.
        spec: Optional ``TechnicalFeaturesSpec``; defaults used if None.

    Returns:
        Fully assembled ``FeatureMatrix``.

    Raises:
        FeatureAssemblyError: On validation failure or computation error.
    """
    iv_str = KlineInterval.from_value(interval).value if not isinstance(interval, str) else interval

    try:
        segments = partition_contiguous_segments(
            candles, interval=interval, symbol=symbol, server_time_ms=server_time_ms
        )
    except Exception as e:
        raise FeatureAssemblyError(
            f"Candle validation/partitioning failed: {e}"
        ) from e

    if not segments:
        raise FeatureAssemblyError(
            "Candle partitioning produced zero segments"
        )

    return assemble_features_from_segments(
        segments=segments,
        symbol=symbol,
        interval=iv_str,
        spec=spec,
    )


def assemble_feature_matrix(
    arrays: NumericalCandleArrays,
    spec: Optional[TechnicalFeaturesSpec] = None,
) -> FeatureMatrix:
    """Simplest entry point: assemble a FeatureMatrix from a single contiguous segment.

    This is the preferred path when the caller already holds a validated
    ``NumericalCandleArrays`` representing a single gap-free segment.

    Args:
        arrays: Validated contiguous ``NumericalCandleArrays``.
        spec: Optional ``TechnicalFeaturesSpec``; defaults used if None.

    Returns:
        Fully assembled ``FeatureMatrix``.

    Raises:
        FeatureAssemblyError: On computation or validation errors.
    """
    if not isinstance(arrays, NumericalCandleArrays):
        raise FeatureAssemblyError(
            f"Expected NumericalCandleArrays, got {type(arrays).__name__}"
        )

    if spec is None:
        spec = TechnicalFeaturesSpec()

    try:
        seg_features = assemble_features_for_segment(arrays)
    except Exception as e:
        raise FeatureAssemblyError(
            f"Feature computation failed: {e}"
        ) from e

    n = arrays.length
    is_gap_boundary = np.zeros(n, dtype=bool)
    is_warmed_up = _build_warmup_mask([n], n)

    metadata = FeatureMetadata(
        symbol=arrays.symbol,
        interval=arrays.interval,
        start_time=int(arrays.open_times[0]),
        end_time=int(arrays.open_times[-1]),
        candle_count=n,
        segment_count=1,
        is_warmed_up=is_warmed_up,
        is_gap_boundary=is_gap_boundary,
        spec_version=spec.version,
        spec_hash=spec.spec_hash,
        generated_at_ms=int(time.time() * 1000),
    )

    return FeatureMatrix(
        symbol=arrays.symbol,
        interval=arrays.interval,
        open_times=arrays.open_times.copy(),
        close_times=arrays.close_times.copy(),
        features=seg_features,
        state_features={},
        metadata=metadata,
    )
