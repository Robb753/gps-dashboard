from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import gpxpy
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000


@dataclass
class TripSummary:
    total_distance_km: float
    total_duration_h: float
    moving_duration_h: float
    average_speed_kmh: float
    moving_average_speed_kmh: float
    max_speed_kmh: float


def haversine_distance_m(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Compute Haversine distance in meters between two sets of coordinates."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def _extract_points(gpx_obj: gpxpy.gpx.GPX) -> list[dict]:
    points: list[dict] = []
    for track in gpx_obj.tracks:
        for segment in track.segments:
            for p in segment.points:
                if p.time is None:
                    continue
                points.append(
                    {
                        "latitude": p.latitude,
                        "longitude": p.longitude,
                        "timestamp": pd.to_datetime(p.time, utc=True),
                    }
                )
    return points


def parse_gpx_file(file: BinaryIO) -> pd.DataFrame:
    """Parse GPX and return a dataframe with cleaned trajectory metrics."""
    gpx_obj = gpxpy.parse(file)
    points = _extract_points(gpx_obj)

    if not points:
        return pd.DataFrame(columns=["latitude", "longitude", "timestamp", "delta_distance_m", "delta_time_s", "speed_kmh", "cum_distance_km"])

    df = pd.DataFrame(points).sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Basic GPS noise filtering using rolling median on coordinates.
    df["latitude"] = df["latitude"].rolling(window=3, center=True, min_periods=1).median()
    df["longitude"] = df["longitude"].rolling(window=3, center=True, min_periods=1).median()

    df["prev_lat"] = df["latitude"].shift(1)
    df["prev_lon"] = df["longitude"].shift(1)
    df["prev_ts"] = df["timestamp"].shift(1)

    df["delta_distance_m"] = haversine_distance_m(df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"]).fillna(0)
    df["delta_time_s"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds().fillna(0)

    # Robust speed computation with clipping for obvious outliers.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_speed = (df["delta_distance_m"] / df["delta_time_s"]) * 3.6
    df["speed_kmh"] = raw_speed.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["speed_kmh"] = df["speed_kmh"].clip(lower=0, upper=180)

    # Remove jitter: if almost no movement, treat speed as zero.
    df.loc[df["delta_distance_m"] < 3, "speed_kmh"] = 0

    df["cum_distance_km"] = df["delta_distance_m"].cumsum() / 1000
    return df.drop(columns=["prev_lat", "prev_lon", "prev_ts"])


def summarize_trip(df: pd.DataFrame, moving_threshold_kmh: float = 2.0) -> TripSummary:
    if df.empty:
        return TripSummary(0, 0, 0, 0, 0, 0)

    total_distance_km = float(df["delta_distance_m"].sum() / 1000)
    total_duration_h = float(df["delta_time_s"].sum() / 3600)

    moving_mask = df["speed_kmh"] >= moving_threshold_kmh
    moving_duration_h = float(df.loc[moving_mask, "delta_time_s"].sum() / 3600)

    average_speed_kmh = total_distance_km / total_duration_h if total_duration_h > 0 else 0.0
    moving_distance_km = float(df.loc[moving_mask, "delta_distance_m"].sum() / 1000)
    moving_average_speed_kmh = moving_distance_km / moving_duration_h if moving_duration_h > 0 else 0.0
    max_speed_kmh = float(df["speed_kmh"].max())

    return TripSummary(
        total_distance_km=total_distance_km,
        total_duration_h=total_duration_h,
        moving_duration_h=moving_duration_h,
        average_speed_kmh=average_speed_kmh,
        moving_average_speed_kmh=moving_average_speed_kmh,
        max_speed_kmh=max_speed_kmh,
    )
