"""Pont entre les calculs de l'app existante et le modèle TripRecord persistable.

Assemble les résultats de gpx_processing.py, analysis.py et du score calculé
dans app.py en un TripRecord prêt à être sauvegardé.

Utilisation :
    record = build_trip_record(
        filename=uploaded_file.name,
        df=segmented,
        summary=summary,
        efficiency=eff,
        stops=stops,
        score=score,
    )
    save_trip(record)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .gpx_processing import TripSummary
from .models import TripRecord
from .similarity import compute_route_id


def build_trip_record(
    filename: str,
    df: pd.DataFrame,
    summary: TripSummary,
    efficiency: dict,
    stops: pd.DataFrame,
    score: float,
) -> TripRecord:
    """Construit un TripRecord à partir des résultats de l'analyse courante.

    Args:
        filename:   nom du fichier GPX importé.
        df:         DataFrame enrichi (issu de classify_speed_segments).
        summary:    TripSummary calculé par summarize_trip().
        efficiency: dict retourné par trip_efficiency().
        stops:      DataFrame retourné par detect_stops().
        score:      score 0-100 calculé dans app.py.

    Returns:
        TripRecord avec un id UUID généré automatiquement.
    """
    first = df.iloc[0]
    last = df.iloc[-1]

    start_lat = float(first["latitude"])
    start_lon = float(first["longitude"])
    end_lat = float(last["latitude"])
    end_lon = float(last["longitude"])

    # Date du trajet = premier timestamp GPS, normalisé en UTC aware.
    trip_date: datetime = first["timestamp"]
    if hasattr(trip_date, "to_pydatetime"):
        trip_date = trip_date.to_pydatetime()
    if trip_date.tzinfo is None:
        trip_date = trip_date.replace(tzinfo=timezone.utc)

    # Temps total immobilisé (somme des arrêts détectés).
    idle_time_s = float(stops["duration_s"].sum()) if not stops.empty else 0.0

    route_id = compute_route_id(start_lat, start_lon, end_lat, end_lon)

    # Résumé texte court généré automatiquement.
    duration_min = round(summary.total_duration_h * 60)
    summary_text = (
        f"{summary.total_distance_km:.1f} km en {duration_min} min — "
        f"vitesse roulante {round(summary.moving_average_speed_kmh)} km/h — "
        f"score {round(score)}/100"
    )

    return TripRecord(
        filename=filename,
        trip_date=trip_date,
        import_date=datetime.now(tz=timezone.utc),
        distance_km=summary.total_distance_km,
        duration_h=summary.total_duration_h,
        moving_duration_h=summary.moving_duration_h,
        avg_speed_kmh=summary.average_speed_kmh,
        moving_avg_speed_kmh=summary.moving_average_speed_kmh,
        max_speed_kmh=summary.max_speed_kmh,
        score=score,
        stop_count=len(stops),
        idle_time_s=idle_time_s,
        moving_ratio=efficiency.get("moving_ratio", 0.0),
        directness_ratio=efficiency.get("directness_ratio", 0.0),
        start_lat=start_lat,
        start_lon=start_lon,
        end_lat=end_lat,
        end_lon=end_lon,
        route_id=route_id,
        summary=summary_text,
    )
