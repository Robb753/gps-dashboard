from __future__ import annotations

import pandas as pd


def detect_stops(df: pd.DataFrame, speed_threshold_kmh: float = 1.0, min_stop_duration_s: int = 90) -> pd.DataFrame:
    """Detect stop windows where speed stays below a threshold for long enough."""
    if df.empty:
        return pd.DataFrame(columns=["start", "end", "duration_s", "lat", "lon"])

    work = df.copy()
    work["is_stop"] = work["speed_kmh"] <= speed_threshold_kmh

    groups = (work["is_stop"] != work["is_stop"].shift()).cumsum()
    stop_rows = []

    for _, grp in work.groupby(groups):
        if not grp["is_stop"].iloc[0]:
            continue

        duration_s = grp["delta_time_s"].sum()
        if duration_s < min_stop_duration_s:
            continue

        stop_rows.append(
            {
                "start": grp["timestamp"].iloc[0],
                "end": grp["timestamp"].iloc[-1],
                "duration_s": float(duration_s),
                "lat": float(grp["latitude"].median()),
                "lon": float(grp["longitude"].median()),
            }
        )

    return pd.DataFrame(stop_rows)


def classify_speed_segments(df: pd.DataFrame, slow_quantile: float = 0.25, fast_quantile: float = 0.75) -> pd.DataFrame:
    """Classify each point as slow/normal/fast based on trip-relative speed quantiles."""
    if df.empty:
        return df.assign(segment_type=pd.Series(dtype=str))

    work = df.copy()
    low = work["speed_kmh"].quantile(slow_quantile)
    high = work["speed_kmh"].quantile(fast_quantile)

    work["segment_type"] = "normal"
    work.loc[work["speed_kmh"] <= low, "segment_type"] = "slow"
    work.loc[work["speed_kmh"] >= high, "segment_type"] = "fast"
    return work


def detect_slow_zones(df: pd.DataFrame, min_points: int = 8) -> pd.DataFrame:
    """Aggregate consecutive slow points into coarse 'slow zones'."""
    if df.empty or "segment_type" not in df:
        return pd.DataFrame(columns=["lat", "lon", "avg_speed_kmh", "distance_km", "samples"])

    slow = df[df["segment_type"] == "slow"].copy()
    if slow.empty:
        return pd.DataFrame(columns=["lat", "lon", "avg_speed_kmh", "distance_km", "samples"])

    bin_size = 0.002  # approx 200m
    slow["lat_bin"] = (slow["latitude"] / bin_size).round().astype(int)
    slow["lon_bin"] = (slow["longitude"] / bin_size).round().astype(int)

    zones = (
        slow.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(
            lat=("latitude", "median"),
            lon=("longitude", "median"),
            avg_speed_kmh=("speed_kmh", "mean"),
            distance_km=("delta_distance_m", lambda s: s.sum() / 1000),
            samples=("speed_kmh", "count"),
        )
        .query("samples >= @min_points")
        .sort_values("samples", ascending=False)
    )
    return zones


def trip_efficiency(df: pd.DataFrame, stops: pd.DataFrame) -> dict:
    """Compute simple efficiency indicators for the trip."""
    if df.empty:
        return {"moving_ratio": 0.0, "stop_ratio": 0.0, "directness_ratio": 0.0}

    total_time_s = df["delta_time_s"].sum()
    stop_time_s = stops["duration_s"].sum() if not stops.empty else 0
    moving_ratio = float(max(total_time_s - stop_time_s, 0) / total_time_s) if total_time_s > 0 else 0.0
    stop_ratio = float(stop_time_s / total_time_s) if total_time_s > 0 else 0.0

    start = df.iloc[0]
    end = df.iloc[-1]
    crow_dist_km = (((start["latitude"] - end["latitude"]) ** 2 + (start["longitude"] - end["longitude"]) ** 2) ** 0.5) * 111
    path_dist_km = float(df["delta_distance_m"].sum() / 1000)
    directness_ratio = float(crow_dist_km / path_dist_km) if path_dist_km > 0 else 0.0

    return {
        "moving_ratio": moving_ratio,
        "stop_ratio": stop_ratio,
        "directness_ratio": min(max(directness_ratio, 0.0), 1.0),
    }
