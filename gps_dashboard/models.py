from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TripRecord:
    """Enregistrement persistable d'un trajet analysé.

    Contient toutes les métriques calculées par l'app + les métadonnées
    nécessaires à la comparaison inter-trajets.
    """

    # --- Identité ---
    filename: str
    trip_date: datetime       # premier timestamp GPS du trajet
    import_date: datetime     # moment d'import dans l'app

    # --- Métriques distance / durée ---
    distance_km: float
    duration_h: float
    moving_duration_h: float

    # --- Métriques vitesse ---
    avg_speed_kmh: float
    moving_avg_speed_kmh: float
    max_speed_kmh: float

    # --- Score et qualité ---
    score: float
    stop_count: int
    idle_time_s: float        # somme des durées d'arrêts détectés (secondes)
    moving_ratio: float       # fraction du temps en mouvement (0-1)
    directness_ratio: float   # vol d'oiseau / distance réelle (0-1)

    # --- Géographie départ / arrivée ---
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

    # --- Regroupement par itinéraire ---
    route_id: str = ""        # hash court MD5 sur départ+arrivée arrondis

    # --- Résumé texte auto-généré ---
    summary: str = ""

    # --- Clé primaire générée automatiquement ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
