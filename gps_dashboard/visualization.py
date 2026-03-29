from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEGMENT_COLORS = {"slow": "#ef4444", "normal": "#3b82f6", "fast": "#22c55e"}


def map_speed_trace(df: pd.DataFrame, stops: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    # Colored route by speed segment.
    for segment, color in SEGMENT_COLORS.items():
        part = df[df["segment_type"] == segment]
        if part.empty:
            continue
        fig.add_trace(
            go.Scattermap(
                lat=part["latitude"],
                lon=part["longitude"],
                mode="markers",
                marker={"size": 6, "color": color, "opacity": 0.7},
                name=f"{segment.title()} segment",
                text=[f"{v:.1f} km/h" for v in part["speed_kmh"]],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Route skeleton to keep continuity.
    fig.add_trace(
        go.Scattermap(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="lines",
            line={"width": 2, "color": "#111827"},
            name="Route",
            hoverinfo="skip",
        )
    )

    if not stops.empty:
        fig.add_trace(
            go.Scattermap(
                lat=stops["lat"],
                lon=stops["lon"],
                mode="markers",
                marker={"size": 11, "color": "#f59e0b", "symbol": "star"},
                name="Stops",
                text=[f"Stop {d/60:.1f} min" for d in stops["duration_s"]],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        map={"style": "open-street-map", "zoom": 11, "center": {"lat": df["latitude"].mean(), "lon": df["longitude"].mean()}},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        height=500,
    )
    return fig


def heatmap_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    fig = px.density_map(
        df,
        lat="latitude",
        lon="longitude",
        z="speed_kmh",
        radius=18,
        center={"lat": df["latitude"].mean(), "lon": df["longitude"].mean()},
        zoom=10,
        map_style="open-street-map",
        title="Heatmap de densité du trajet",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0}, height=450)
    return fig


def line_speed_vs_time(df: pd.DataFrame) -> go.Figure:
    return px.line(df, x="timestamp", y="speed_kmh", title="Vitesse vs temps", labels={"timestamp": "Temps", "speed_kmh": "Vitesse (km/h)"})


def line_distance_vs_time(df: pd.DataFrame) -> go.Figure:
    return px.line(df, x="timestamp", y="cum_distance_km", title="Distance cumulée vs temps", labels={"timestamp": "Temps", "cum_distance_km": "Distance (km)"})


def histogram_speed(df: pd.DataFrame) -> go.Figure:
    return px.histogram(df, x="speed_kmh", nbins=35, title="Distribution des vitesses", labels={"speed_kmh": "Vitesse (km/h)"})
