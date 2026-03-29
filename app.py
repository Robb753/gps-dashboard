from __future__ import annotations

import streamlit as st

from gps_dashboard.analysis import classify_speed_segments, detect_slow_zones, detect_stops, trip_efficiency
from gps_dashboard.gpx_processing import parse_gpx_file, summarize_trip
from gps_dashboard.visualization import heatmap_figure, histogram_speed, line_distance_vs_time, line_speed_vs_time, map_speed_trace

st.set_page_config(page_title="GPS Trip Analyzer", layout="wide")
st.title("🚗🚶 Dashboard d'analyse GPS (GPX)")
st.caption("Importez un fichier GPX exporté depuis Locus Map pour analyser un trajet voiture ou marche.")

with st.sidebar:
    st.header("Paramètres")
    stop_speed_threshold = st.slider("Seuil arrêt (km/h)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    min_stop_duration = st.slider("Durée minimale d'un arrêt (secondes)", min_value=30, max_value=600, value=90, step=30)

uploaded_file = st.file_uploader("1) Upload d'un fichier GPX", type=["gpx"])

if not uploaded_file:
    st.info("Chargez un fichier GPX pour démarrer l'analyse.")
    st.stop()

with st.spinner("Parsing GPX et calcul des métriques..."):
    df = parse_gpx_file(uploaded_file)

if df.empty:
    st.error("Aucun point valide trouvé dans le GPX (vérifiez timestamps et structure du fichier).")
    st.stop()

# Intelligent analysis
segmented = classify_speed_segments(df)
stops = detect_stops(segmented, speed_threshold_kmh=stop_speed_threshold, min_stop_duration_s=min_stop_duration)
slow_zones = detect_slow_zones(segmented)
eff = trip_efficiency(segmented, stops)
summary = summarize_trip(segmented)

# KPI cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Distance totale", f"{summary.total_distance_km:.2f} km")
col2.metric("Durée totale", f"{summary.total_duration_h:.2f} h")
col3.metric("Vitesse moyenne", f"{summary.average_speed_kmh:.1f} km/h")
col4.metric("Vitesse max", f"{summary.max_speed_kmh:.1f} km/h")
col5.metric("Arrêts détectés", f"{len(stops)}")

st.markdown("---")

left, right = st.columns([2, 1])
with left:
    st.subheader("2) Carte interactive (trace colorée par vitesse)")
    st.plotly_chart(map_speed_trace(segmented, stops), use_container_width=True)
with right:
    st.subheader("3) Efficacité du trajet")
    st.metric("Temps en mouvement", f"{eff['moving_ratio'] * 100:.1f}%")
    st.metric("Temps à l'arrêt", f"{eff['stop_ratio'] * 100:.1f}%")
    st.metric("Ratio de directivité", f"{eff['directness_ratio'] * 100:.1f}%")
    st.caption("Le ratio de directivité compare la distance à vol d'oiseau et la distance réellement parcourue.")

row1, row2 = st.columns(2)
with row1:
    st.plotly_chart(line_speed_vs_time(segmented), use_container_width=True)
    st.plotly_chart(histogram_speed(segmented), use_container_width=True)
with row2:
    st.plotly_chart(line_distance_vs_time(segmented), use_container_width=True)
    st.plotly_chart(heatmap_figure(segmented), use_container_width=True)

st.subheader("4) Zones lentes détectées")
if slow_zones.empty:
    st.success("Aucune zone lente significative détectée sur ce trajet.")
else:
    show = slow_zones.copy()
    show["avg_speed_kmh"] = show["avg_speed_kmh"].round(2)
    show["distance_km"] = show["distance_km"].round(3)
    st.dataframe(show, use_container_width=True)

st.subheader("5) Arrêts détectés")
if stops.empty:
    st.info("Aucun arrêt long détecté avec les paramètres actuels.")
else:
    display = stops.copy()
    display["duration_min"] = (display["duration_s"] / 60).round(1)
    st.dataframe(display[["start", "end", "duration_min", "lat", "lon"]], use_container_width=True)

st.subheader("6) Données calculées")
with st.expander("Voir les points enrichis"):
    st.dataframe(segmented.head(500), use_container_width=True)

st.caption("Conseil : ajustez les seuils dans la barre latérale selon le mode de transport.")
