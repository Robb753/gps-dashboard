from __future__ import annotations

import streamlit as st

from gps_dashboard.analysis import classify_speed_segments, detect_slow_zones, detect_stops, trip_efficiency
from gps_dashboard.gpx_processing import parse_gpx_file, summarize_trip
from gps_dashboard.visualization import heatmap_figure, histogram_speed, line_distance_vs_time, line_speed_vs_time, map_speed_trace

st.set_page_config(page_title="DriveSense — GPS Trip Intelligence", layout="wide")


def format_duration(hours: float) -> str:
    total_min = max(int(hours * 60), 0)
    h = total_min // 60
    m = total_min % 60
    return f"{h}h {m:02d}m"


def to_percent(value: float) -> float:
    return float(max(min(value * 100, 100), 0))


def trip_score(summary, eff: dict, stops_count: int) -> tuple[int, str, str]:
    moving_score = to_percent(eff["moving_ratio"])
    direct_score = to_percent(eff["directness_ratio"])

    speed_penalty = 0
    if summary.max_speed_kmh > 130:
        speed_penalty = min((summary.max_speed_kmh - 130) * 0.7, 12)

    stop_penalty = min(stops_count * 2.0, 12)

    score = int(round(0.45 * moving_score + 0.35 * direct_score + 0.20 * min(summary.moving_average_speed_kmh, 120) - speed_penalty - stop_penalty))
    score = max(min(score, 100), 0)

    if score >= 80:
        label = "Fluide"
        message = "Trajet globalement efficace et régulier."
    elif score >= 60:
        label = "Correct"
        message = "Bon trajet, mais quelques points de friction ralentissent la progression."
    else:
        label = "À optimiser"
        message = "Le trajet présente des ralentissements importants ou trop d'arrêts."

    return score, label, message


def build_insights(summary, eff: dict, stops, slow_zones) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []

    stop_ratio = to_percent(eff["stop_ratio"])
    if stop_ratio >= 25:
        insights.append(
            {
                "level": "Critique",
                "title": "Temps perdu élevé à l'arrêt",
                "detail": f"{stop_ratio:.1f}% du trajet est passé immobile. Vérifiez les zones de congestion et les fenêtres horaires.",
            }
        )
    elif stop_ratio >= 12:
        insights.append(
            {
                "level": "Moyen",
                "title": "Temps d'arrêt non négligeable",
                "detail": f"{stop_ratio:.1f}% du trajet est immobilisé. Des gains sont possibles via un itinéraire alternatif.",
            }
        )

    if summary.max_speed_kmh >= 120:
        insights.append(
            {
                "level": "Vigilance",
                "title": "Pic de vitesse notable",
                "detail": f"Un pic à {summary.max_speed_kmh:.0f} km/h est détecté. Surveillez la régularité de conduite.",
            }
        )

    if not slow_zones.empty:
        top_zone = slow_zones.iloc[0]
        insights.append(
            {
                "level": "Info",
                "title": "Zone lente prioritaire",
                "detail": f"Zone ({top_zone['lat']:.4f}, {top_zone['lon']:.4f}) avec {int(top_zone['samples'])} points lents et {top_zone['avg_speed_kmh']:.1f} km/h en moyenne.",
            }
        )

    if summary.moving_average_speed_kmh > summary.average_speed_kmh * 1.35:
        insights.append(
            {
                "level": "Opportunité",
                "title": "Fort écart entre vitesse roulante et moyenne totale",
                "detail": "Le trajet est performant en mouvement mais perturbé par des interruptions. Le levier principal est la réduction des arrêts.",
            }
        )

    if not insights:
        insights.append(
            {
                "level": "OK",
                "title": "Trajet équilibré",
                "detail": "Aucune anomalie majeure. La fluidité et la progression sont cohérentes.",
            }
        )

    return insights[:4]


def section_title(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(f"#### {kicker}")
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def style_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#e5e7eb"},
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


st.markdown("""
<style>
.main {
    background: radial-gradient(circle at top right, #172554 0%, #0f172a 35%, #020617 100%);
}
section[data-testid="stSidebar"] {
    background: #0b1220;
}
div[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 10px 14px;
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}
.block-card {
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 14px;
    background: rgba(15, 23, 42, 0.7);
}
</style>
""", unsafe_allow_html=True)

st.title("🚘 DriveSense — Trip Intelligence")
st.caption("Analyse premium d'un trajet voiture à partir d'un fichier GPX : fluidité, points de friction et recommandations actionnables.")

with st.sidebar:
    st.header("Paramètres d'analyse")
    stop_speed_threshold = st.slider("Seuil arrêt (km/h)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    min_stop_duration = st.slider("Durée minimale d'un arrêt (secondes)", min_value=30, max_value=600, value=90, step=30)
    st.caption("Ajustez selon trafic urbain vs autoroute pour éviter les faux positifs.")

uploaded_file = st.file_uploader("Importer un fichier GPX", type=["gpx"])

if not uploaded_file:
    st.info("Chargez un GPX pour générer le rapport de trajet.")
    st.stop()

with st.spinner("Analyse en cours : parsing GPX, segmentation, détection d'événements..."):
    df = parse_gpx_file(uploaded_file)

if df.empty:
    st.error("Aucun point exploitable trouvé dans le GPX (timestamps ou structure invalides).")
    st.stop()

segmented = classify_speed_segments(df)
stops = detect_stops(segmented, speed_threshold_kmh=stop_speed_threshold, min_stop_duration_s=min_stop_duration)
slow_zones = detect_slow_zones(segmented)
eff = trip_efficiency(segmented, stops)
summary = summarize_trip(segmented)

score, score_label, score_message = trip_score(summary, eff, len(stops))
insights = build_insights(summary, eff, stops, slow_zones)

# 1) Hero + score global
section_title("VUE RAPIDE", "Résumé intelligent du trajet", "Compréhension en 5 secondes : score, durée, distance, fluidité.")
hero_left, hero_right = st.columns([1.3, 2.2], gap="large")

with hero_left:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.metric("Score global du trajet", f"{score}/100", score_label)
    st.caption(score_message)
    st.metric("Efficacité mouvement", f"{to_percent(eff['moving_ratio']):.1f}%")
    st.metric("Temps immobilisé", f"{to_percent(eff['stop_ratio']):.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with hero_right:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Distance", f"{summary.total_distance_km:.2f} km")
    k2.metric("Durée", format_duration(summary.total_duration_h))
    k3.metric("Vitesse moyenne", f"{summary.average_speed_kmh:.1f} km/h")
    k4.metric("Vitesse roulante", f"{summary.moving_average_speed_kmh:.1f} km/h")
    k5.metric("Arrêts longs", f"{len(stops)}")

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown("**Résumé automatique**")
    st.write(
        f"Trajet de **{summary.total_distance_km:.1f} km** en **{format_duration(summary.total_duration_h)}**. "
        f"La progression est **{score_label.lower()}**, avec **{to_percent(eff['stop_ratio']):.1f}%** du temps à l'arrêt "
        f"et une directivité de **{to_percent(eff['directness_ratio']):.1f}%**."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# 2) Carte centrale + panneau insights
section_title("LECTURE TRAJET", "Carte et événements clés", "La carte reste centrale, complétée par une lecture analytique des frictions.")
map_col, insight_col = st.columns([2.1, 1], gap="large")

with map_col:
    trip_map = style_figure(map_speed_trace(segmented, stops))
    trip_map.update_layout(height=530)
    st.plotly_chart(trip_map, use_container_width=True)

with insight_col:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown("**Insights automatiques**")
    for item in insights:
        st.markdown(f"- **{item['level']} · {item['title']}**  ")
        st.caption(item["detail"])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# 3) Story chronologique
section_title("NARRATION", "Chronologie de performance", "Les graphiques racontent le trajet : rythme, progression, zones de tension.")

c1, c2 = st.columns(2, gap="large")
with c1:
    speed_fig = style_figure(line_speed_vs_time(segmented))
    speed_fig.update_layout(title="Rythme de conduite (vitesse vs temps)")
    st.plotly_chart(speed_fig, use_container_width=True)

    hist_fig = style_figure(histogram_speed(segmented))
    hist_fig.update_layout(title="Distribution des vitesses")
    st.plotly_chart(hist_fig, use_container_width=True)

with c2:
    dist_fig = style_figure(line_distance_vs_time(segmented))
    dist_fig.update_layout(title="Progression cumulée (distance vs temps)")
    st.plotly_chart(dist_fig, use_container_width=True)

    heat_fig = style_figure(heatmap_figure(segmented))
    heat_fig.update_layout(title="Concentration spatiale et intensité de vitesse")
    st.plotly_chart(heat_fig, use_container_width=True)

st.markdown("---")

# 4) Zones lentes et arrêts, orientés action
section_title("ÉVÉNEMENTS", "Zones lentes et arrêts exploitables", "Tableaux orientés diagnostic opérationnel.")

t1, t2 = st.columns(2, gap="large")
with t1:
    st.markdown("**Zones lentes prioritaires**")
    if slow_zones.empty:
        st.success("Aucune zone lente significative détectée.")
    else:
        show = slow_zones.copy()
        show["avg_speed_kmh"] = show["avg_speed_kmh"].round(1)
        show["distance_km"] = show["distance_km"].round(3)
        st.dataframe(
            show[["lat", "lon", "samples", "avg_speed_kmh", "distance_km"]],
            use_container_width=True,
            hide_index=True,
        )

with t2:
    st.markdown("**Arrêts longs détectés**")
    if stops.empty:
        st.info("Aucun arrêt long détecté avec les paramètres actuels.")
    else:
        display = stops.copy()
        display["duration_min"] = (display["duration_s"] / 60).round(1)
        st.dataframe(
            display[["start", "end", "duration_min", "lat", "lon"]],
            use_container_width=True,
            hide_index=True,
        )

# 5) Détail technique replié
with st.expander("Voir les points enrichis (debug / audit)"):
    st.dataframe(segmented.head(500), use_container_width=True, hide_index=True)

st.caption("Conseil produit : comparez ce trajet avec vos prochains fichiers GPX sur le même itinéraire pour objectiver les gains.")
