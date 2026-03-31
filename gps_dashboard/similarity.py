"""Détection de trajets similaires et calcul d'indicateurs de comparaison.

Stratégie (sans ML) :
    1. route_id  → hash MD5 sur départ+arrivée arrondis à 2 décimales (~1.1 km)
                   Regroupe rapidement les trajets du même itinéraire.
    2. Endpoints → vérification géométrique Haversine à ±500 m (filet de sécurité).
    3. Distance  → compatibilité à ±15 % (évite de confondre des trajets différents
                   qui partagent un parking commun).

Choix des seuils
    - Précision 2 décimales ≈ 1.1 km : absorbe les variations de parking quotidiennes
      sans confondre des quartiers différents.
    - Tolérance endpoints 500 m : couvre le cas "je pars de la maison vs du bureau
      à 300 m de là".
    - Tolérance distance ±15 % : un trajet de 10 km reste compatible jusqu'à 8.5-11.5 km.
      Au-delà, c'est un itinéraire différent.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import pandas as pd

# --- Constantes ---
EARTH_RADIUS_KM = 6_371.0
ENDPOINT_TOLERANCE_KM: float = 0.5   # 500 m
DISTANCE_TOLERANCE_PCT: float = 0.15  # ±15 %
ROUTE_ID_PRECISION: int = 2           # décimales pour l'arrondi lat/lon


# ---------------------------------------------------------------------------
# Fonctions géographiques
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance Haversine entre deux points GPS, en kilomètres."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(max(a, 0)))


# ---------------------------------------------------------------------------
# Identification de l'itinéraire
# ---------------------------------------------------------------------------

def compute_route_id(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    precision: int = ROUTE_ID_PRECISION,
) -> str:
    """Génère un identifiant stable d'itinéraire.

    Arrondit les coordonnées à `precision` décimales avant de hacher.
    Avec precision=2, deux départs dans un rayon de ~1.1 km donnent le même id.

    Returns:
        Chaîne hexadécimale de 10 caractères (ex : "a3f8b2c1d4").
    """
    start_key = f"{round(start_lat, precision)},{round(start_lon, precision)}"
    end_key = f"{round(end_lat, precision)},{round(end_lon, precision)}"
    raw = f"{start_key}->{end_key}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Filtres de similarité
# ---------------------------------------------------------------------------

def _endpoints_match(
    lat1s: float, lon1s: float, lat1e: float, lon1e: float,
    lat2s: float, lon2s: float, lat2e: float, lon2e: float,
    tolerance_km: float,
) -> bool:
    """True si départ et arrivée sont tous deux dans le rayon de tolérance."""
    return (
        haversine_km(lat1s, lon1s, lat2s, lon2s) <= tolerance_km
        and haversine_km(lat1e, lon1e, lat2e, lon2e) <= tolerance_km
    )


def _distance_compatible(d1: float, d2: float, tolerance_pct: float) -> bool:
    """True si les deux distances sont dans le même ordre de grandeur."""
    if d1 <= 0 or d2 <= 0:
        return False
    ratio = abs(d1 - d2) / max(d1, d2)
    return ratio <= tolerance_pct


# ---------------------------------------------------------------------------
# Recherche de trajets similaires
# ---------------------------------------------------------------------------

def find_similar_trips(
    current: dict,
    history_df: pd.DataFrame,
    endpoint_tolerance_km: float = ENDPOINT_TOLERANCE_KM,
    distance_tolerance_pct: float = DISTANCE_TOLERANCE_PCT,
) -> pd.DataFrame:
    """Trouve les trajets historiques similaires au trajet courant.

    Args:
        current: dict avec les clés obligatoires :
            start_lat, start_lon, end_lat, end_lon, distance_km, route_id
        history_df: DataFrame retourné par load_all_trips().
        endpoint_tolerance_km: rayon de tolérance pour les endpoints (km).
        distance_tolerance_pct: tolérance relative sur la distance (0-1).

    Returns:
        Sous-DataFrame des trajets similaires, trié par trip_date décroissant.
        Retourne un DataFrame vide si history_df est vide.

    Algorithme :
        1. Filtre rapide par route_id (hash de cellules ~1.1 km).
        2. Fallback : vérification Haversine exacte à ±500 m si le filtre 1 donne rien.
        3. Dans les deux cas, filtre de distance à ±15 %.
    """
    if history_df.empty:
        return pd.DataFrame()

    # Filtre 1 — route_id identique (O(1) en SQL, O(n) ici sur le DataFrame en mémoire)
    candidates = history_df[history_df["route_id"] == current["route_id"]].copy()

    # Filtre 2 — fallback géométrique si le hash seul ne suffit pas
    if candidates.empty:
        mask = history_df.apply(
            lambda row: _endpoints_match(
                current["start_lat"], current["start_lon"],
                current["end_lat"], current["end_lon"],
                float(row["start_lat"]), float(row["start_lon"]),
                float(row["end_lat"]), float(row["end_lon"]),
                endpoint_tolerance_km,
            ),
            axis=1,
        )
        candidates = history_df[mask].copy()

    # Filtre 3 — compatibilité de distance (anti-faux-positifs)
    if not candidates.empty:
        candidates = candidates[
            candidates["distance_km"].apply(
                lambda d: _distance_compatible(
                    float(d), current["distance_km"], distance_tolerance_pct
                )
            )
        ]

    return candidates.sort_values("trip_date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calcul des indicateurs de comparaison
# ---------------------------------------------------------------------------

def compute_comparison_stats(current: dict, similar_df: pd.DataFrame) -> dict:
    """Calcule les deltas entre le trajet courant et les trajets similaires passés.

    Args:
        current: dict avec au moins {score, duration_h, moving_avg_speed_kmh, stop_count}.
        similar_df: résultat de find_similar_trips().

    Returns:
        Dictionnaire d'indicateurs prêts pour l'affichage Streamlit.
        Clé "has_comparison" = False si pas assez de données (< 2 trajets).

    Note :
        On exige au moins 2 trajets pour calculer une "moyenne" significative.
        Avec un seul trajet passé, les deltas seraient trompeurs.
    """
    if similar_df.empty or len(similar_df) < 2:
        return {
            "has_comparison": False,
            "count": len(similar_df) if not similar_df.empty else 0,
        }

    avg_score = float(similar_df["score"].mean())
    avg_duration_h = float(similar_df["duration_h"].mean())
    avg_moving_speed = float(similar_df["moving_avg_speed_kmh"].mean())
    avg_stops = float(similar_df["stop_count"].mean())
    best_score = float(similar_df["score"].max())
    worst_score = float(similar_df["score"].min())

    score_delta = current["score"] - avg_score
    duration_delta_min = (current["duration_h"] - avg_duration_h) * 60
    speed_delta = current["moving_avg_speed_kmh"] - avg_moving_speed
    stop_delta = current["stop_count"] - avg_stops

    # Tendance du score sur les 10 derniers trajets (chronologique → plus récent en dernier)
    trend_series = (
        similar_df.sort_values("trip_date")
        .tail(10)["score"]
        .tolist()
    )

    return {
        "has_comparison": True,
        "count": len(similar_df),
        # Moyennes de référence
        "avg_score": round(avg_score, 1),
        "best_score": round(best_score, 1),
        "worst_score": round(worst_score, 1),
        "avg_duration_h": round(avg_duration_h, 2),
        "avg_moving_speed_kmh": round(avg_moving_speed, 1),
        "avg_stop_count": round(avg_stops, 1),
        # Deltas (positif = meilleur que la moyenne, sauf durée et stops)
        "score_delta": round(score_delta, 1),
        "duration_delta_min": round(duration_delta_min, 1),
        "speed_delta": round(speed_delta, 1),
        "stop_delta": round(stop_delta, 1),
        # Tendance brute pour mini-graphe
        "score_trend": trend_series,
        # Méta
        "last_trip_date": similar_df["trip_date"].max(),
    }
