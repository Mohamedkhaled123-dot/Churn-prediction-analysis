import streamlit as st


def render_kpi(title: str, value: str, change: str = None, description: str = None):
    st.markdown("<div class='glass-card'>\n  <div style='display:flex;justify-content:space-between;align-items:center;'>\n    <p class='kpi-title'>{}</p>\n    <p style='margin:0;color:#60a5fa;'>{}</p>\n  </div>\n  <h2 class='kpi-value'>{}</h2>\n  {}\n</div>".format(
        title,
        change or "",
        value,
        f"<p style='color:#9ca3af;margin-top:0.5rem;font-size:0.92rem;'>{description}</p>" if description else ""
    ), unsafe_allow_html=True)
