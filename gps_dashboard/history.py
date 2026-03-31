"""Couche de persistance SQLite pour l'historique des trajets.

Le fichier de base de données est stocké dans ~/.gps_dashboard/history.db
(hors du dépôt git) pour ne jamais être commité par erreur.

Utilisation :
    from gps_dashboard.history import save_trip, load_all_trips, load_similar_trips

    save_trip(record)
    df = load_all_trips()
    df_route = load_similar_trips("a3f8b2c1d4")
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import TripRecord

# Emplacement de la base, hors repo.
DB_PATH = Path.home() / ".gps_dashboard" / "history.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trips (
    id                   TEXT PRIMARY KEY,
    filename             TEXT NOT NULL,
    trip_date            TEXT,
    import_date          TEXT NOT NULL,
    distance_km          REAL,
    duration_h           REAL,
    moving_duration_h    REAL,
    avg_speed_kmh        REAL,
    moving_avg_speed_kmh REAL,
    max_speed_kmh        REAL,
    score                REAL,
    stop_count           INTEGER,
    idle_time_s          REAL,
    moving_ratio         REAL,
    directness_ratio     REAL,
    start_lat            REAL,
    start_lon            REAL,
    end_lat              REAL,
    end_lon              REAL,
    route_id             TEXT,
    summary              TEXT
);
CREATE INDEX IF NOT EXISTS idx_route_id  ON trips(route_id);
CREATE INDEX IF NOT EXISTS idx_trip_date ON trips(trip_date DESC);
"""


def _connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crée la table et les index si nécessaire. Idempotent."""
    with _connection() as conn:
        conn.executescript(_CREATE_TABLE_SQL)


def save_trip(record: TripRecord) -> str:
    """Persiste un TripRecord. Remplace l'enregistrement si l'id existe déjà.

    Returns:
        L'id du trajet sauvegardé.
    """
    init_db()
    with _connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO trips (
                id, filename, trip_date, import_date,
                distance_km, duration_h, moving_duration_h,
                avg_speed_kmh, moving_avg_speed_kmh, max_speed_kmh,
                score, stop_count, idle_time_s,
                moving_ratio, directness_ratio,
                start_lat, start_lon, end_lat, end_lon,
                route_id, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.filename,
                record.trip_date.isoformat() if record.trip_date else None,
                record.import_date.isoformat(),
                record.distance_km,
                record.duration_h,
                record.moving_duration_h,
                record.avg_speed_kmh,
                record.moving_avg_speed_kmh,
                record.max_speed_kmh,
                record.score,
                record.stop_count,
                record.idle_time_s,
                record.moving_ratio,
                record.directness_ratio,
                record.start_lat,
                record.start_lon,
                record.end_lat,
                record.end_lon,
                record.route_id,
                record.summary,
            ),
        )
        conn.commit()
    return record.id


def load_all_trips() -> pd.DataFrame:
    """Charge tous les trajets, triés du plus récent au plus ancien."""
    init_db()
    with _connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM trips ORDER BY trip_date DESC",
            conn,
        )
    if not df.empty:
        df["trip_date"] = pd.to_datetime(df["trip_date"], utc=True, errors="coerce")
        df["import_date"] = pd.to_datetime(df["import_date"], utc=True, errors="coerce")
    return df


def load_similar_trips(route_id: str) -> pd.DataFrame:
    """Charge uniquement les trajets ayant le même route_id."""
    init_db()
    with _connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM trips WHERE route_id = ? ORDER BY trip_date DESC",
            conn,
            params=(route_id,),
        )
    if not df.empty:
        df["trip_date"] = pd.to_datetime(df["trip_date"], utc=True, errors="coerce")
        df["import_date"] = pd.to_datetime(df["import_date"], utc=True, errors="coerce")
    return df


def delete_trip(trip_id: str) -> None:
    """Supprime un trajet par son id."""
    init_db()
    with _connection() as conn:
        conn.execute("DELETE FROM trips WHERE id = ?", (trip_id,))
        conn.commit()


def trip_exists(trip_id: str) -> bool:
    """Vérifie si un trajet est déjà en base."""
    init_db()
    with _connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM trips WHERE id = ? LIMIT 1", (trip_id,)
        ).fetchone()
    return row is not None
