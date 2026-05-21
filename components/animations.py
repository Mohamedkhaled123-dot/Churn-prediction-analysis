import streamlit as st

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st_lottie = None


def render_lottie_animation(lottie_data, height: int = 380):
    if st_lottie is None:
        st.info("Lottie animation support is unavailable. Install streamlit-lottie to enable animations.")
        return
    if lottie_data is not None:
        st_lottie(lottie_data, height=height, key=f"lottie_{hash(str(lottie_data))}")
    else:
        st.info("Animation unavailable. Please check your network connection.")
