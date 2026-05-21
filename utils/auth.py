import streamlit as st

USERS = {
    "admin": "Enterprise2026",
    "analyst": "AIretention"
}


def authenticate(username: str, password: str) -> bool:
    return USERS.get(username) == password


def render_sign_in():
    st.title("ChurnAI Studio")
    st.write("Secure access to the customer churn intelligence platform.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if authenticate(username, password):
            st.session_state["authenticated"] = True
            st.session_state["current_user"] = username
            st.rerun()
        else:
            st.error("Invalid credentials. Try admin / Enterprise2026.")


def logout():
    for key in ["authenticated", "current_user", "training_results", "conversation", "raw_df", "cleaned_df", "active_df"]:
        if key in st.session_state:
            del st.session_state[key]
