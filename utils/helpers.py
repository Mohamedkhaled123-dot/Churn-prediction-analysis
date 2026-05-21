import requests
import streamlit as st


def load_css(path: str):
    try:
        with open(path, "r", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom stylesheet not found. Using default theme.")


def load_lottie_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def format_currency(value, symbol="$"):
    return f"{symbol}{value:,.2f}"


def status_badge(value: str):
    color = "#22c55e" if value == "Good" else "#fb7185"
    return f"<span style='display:inline-block;padding:0.5rem 0.75rem;border-radius:999px;background:{color}22;color:{color};font-size:0.90rem;'>{value}</span>"
