import streamlit as st

# Streamlit app setup
st.set_page_config(page_title="AEV - AEA", layout="wide")


# Check if `mw_object` is available
if 'mw_object' in st.session_state and st.session_state.mw_object:
    mw_object = st.session_state.mw_object
    mw_object = mw_object.copy()

    