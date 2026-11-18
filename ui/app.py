"""Streamlit evaluator dashboard; all predictions are requested from the API."""

import os
from typing import Any

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 8


def api_base_url() -> str:
    return os.environ.get("PATHSNAP_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def get_json(path: str) -> Any:
    response = requests.get(
        f"{api_base_url()}{path}", timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(
        f"{api_base_url()}{path}",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def render_route_map(prediction: dict[str, Any]) -> None:
    gps_points = prediction.get("gps_coordinates", [])
    route_groups = [
        ("HMM route", prediction.get("route_hmm") or [], "blue"),
        (
            "Classifier-only route",
            prediction.get("route_classifier") or [],
            "green",
        ),
    ]
    if gps_points:
        center = [
            sum(point["latitude"] for point in gps_points) / len(gps_points),
            sum(point["longitude"] for point in gps_points) / len(gps_points),
        ]
        zoom = 13
    else:
        center = [0, 0]
        zoom = 2

    route_map = folium.Map(location=center, zoom_start=zoom)
    for point in gps_points:
        folium.CircleMarker(
            location=[point["latitude"], point["longitude"]],
            radius=4,
            color="orange",
            fill=True,
            tooltip="GPS observation",
        ).add_to(route_map)

    for label, polylines, color in route_groups:
        for polyline in polylines:
            coordinates = [
                [point["latitude"], point["longitude"]]
                for point in polyline.get("points", [])
            ]
            if coordinates:
                folium.PolyLine(
                    coordinates, color=color, weight=4, tooltip=label
                ).add_to(route_map)

    st_folium(route_map, use_container_width=True, height=500)
    if not gps_points and not any(group[1] for group in route_groups):
        st.info("No GPS points or predicted routes are available in stub mode.")


def main() -> None:
    st.set_page_config(page_title="PathSnap evaluator", layout="wide")
    st.title("PathSnap")
    st.write(
        "Distributed map-matching and route reconstruction — "
        "Member 4 evaluator dashboard."
    )
    st.caption(f"API: {api_base_url()}")

    try:
        health = get_json("/health")
        health_message = (
            f"API status: {health['status']} · "
            f"Model: {health['model_name']} · "
            f"Schema: {health['schema_version']}"
        )
        if health["status"] == "ok":
            st.success(health_message)
        else:
            st.warning(health_message)
    except requests.RequestException as exc:
        st.error(f"Cannot reach the PathSnap API at {api_base_url()}: {exc}")
        st.stop()

    try:
        trip_ids = get_json("/trips")
    except requests.RequestException as exc:
        st.error(f"Could not load trips from the API: {exc}")
        st.stop()

    if not trip_ids:
        st.warning(
            "No trip folders were discovered. Configure "
            "`PATHSNAP_EVALUATION_DIR` and add trip folders before predicting."
        )
        st.stop()

    selected_trip = st.selectbox("Trip", trip_ids)
    if st.button("Run Prediction", type="primary"):
        try:
            prediction = post_json("/predict", {"trip_id": selected_trip})
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Prediction request failed: {detail}")
            st.stop()
        except requests.RequestException as exc:
            st.error(f"Could not contact the prediction API: {exc}")
            st.stop()

        if prediction.get("is_stub"):
            st.info("Walking-skeleton response: no real prediction was generated.")
        st.subheader("GPS points and predicted route")
        render_route_map(prediction)
        if not prediction.get("route_classifier"):
            st.caption(
                "Classifier-only route layer: awaiting Member 2 classification output."
            )

        st.subheader("Metrics and diagnostics")
        if prediction.get("metrics"):
            st.json(prediction["metrics"])
        else:
            st.info("No evaluation metrics are available.")
        st.json(prediction.get("diagnostics", {}))
        st.subheader("Stage timings")
        st.json(prediction.get("timings", {}))


if __name__ == "__main__":
    main()
