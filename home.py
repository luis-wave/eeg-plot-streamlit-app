import tempfile
from pathlib import Path

import streamlit as st
import toml
from mywaveanalytics.libraries import (eeg_computational_library, filters,
                                       mywaveanalytics, references)

# Initialize Streamlit session state for shared data
if "mw_object" not in st.session_state:
    st.session_state.mw_object = None

st.title("EEG Analysis Dashboard")


# Function to read version from pyproject.toml
def get_version_from_pyproject():
    try:
        pyproject_data = toml.load("pyproject.toml")
        return pyproject_data["tool"]["poetry"]["version"]
    except Exception as e:
        st.error(f"Error reading version from pyproject.toml: {e}")
        return "Unknown"


def calculate_eqi(mw_object):
    try:
        mw_copy = mw_object.copy()
        filters.eeg_filter(mw_copy, 1, 25)
        filters.notch(mw_copy)
        filters.resample(mw_copy)
        tcp_eeg = references.temporal_central_parasagittal(mw_copy)

        eqi_features, z_scored_eqi = eeg_computational_library.calculate_eqi(tcp_eeg)
        eqi_predictions, eqi_score = eeg_computational_library.eqi_svm_inference(
            z_scored_eqi
        )

        return round(eqi_score)
    except Exception as e:
        st.error(f"EEG quality assessment failed for the following reason: {e}")


def load_mw_object(path, eegtype):
    try:
        mw_object = mywaveanalytics.MyWaveAnalytics(path, None, None, eegtype)
        return mw_object
    except Exception as e:
        st.error(f"Loading failed for {path}: {e}")
        return None


def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_file.name).suffix
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Failed to save the uploaded file: {e}")
        return None


uploaded_file = st.file_uploader("Upload an EEG file", type=["dat", "edf"])

if uploaded_file is not None:
    st.session_state.fname = uploaded_file.name

    # Save uploaded file to disk to make it accessible by file path
    saved_path = save_uploaded_file(uploaded_file)
    if saved_path:
        # Determine EEG type based on file extension
        eeg_type = (
            0
            if uploaded_file.name.lower().endswith(".dat")
            else 10
            if uploaded_file.name.lower().endswith(".edf")
            else None
        )
        if eeg_type is None:
            st.error("Unsupported file type.")
        else:
            mw_object = load_mw_object(saved_path, eeg_type)
            st.success("EEG Data loaded successfully!")

            with st.spinner("Processing..."):
                st.session_state.mw_object = mw_object.copy()

                eqi = calculate_eqi(mw_object)

                # Save the relevant state
                st.session_state.eqi = eqi

                st.switch_page("pages/epochs.py")

else:
    st.info("Please upload an EEG file.")


# Footer section
version = get_version_from_pyproject()
footer_html = f"""
    <div style='position: fixed; bottom: 0; left: 0; padding: 10px;'>
        <span>Version: {version}</span>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
