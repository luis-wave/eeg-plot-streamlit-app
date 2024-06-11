# epochs.py

import mne
import numpy as np
import streamlit as st
from pipeline import PersistPipeline


st.set_page_config(page_title="EEG Epoch Generator", layout="wide")

if 'mw_object' not in st.session_state:
    st.error("Please load EEG data")
else:

    st.title("EEG Epoch Generator")

    col1, col2 = st.columns(2)

    if st.session_state.filename and ('/tmp/' not in st.session_state.filename) :
        #st.header(st.session_state.filename)
        col1.metric("Filename", st.session_state.filename)
    elif st.session_state.eeg_id:
        col1.metric("EEGId", st.session_state.eeg_id)

    col2.metric("Recording Date", st.session_state.recording_date )

    def run_persist_pipeline(mw_object):
        try:
            pipeline = PersistPipeline(mw_object)
            st.success("EEG Data loaded successfully!")
            return pipeline
        except Exception as e:
            st.error(f"Epoch analysis failed: {e}")

    # Check if `mw_object` is available
    if 'mw_object' in st.session_state and st.session_state.mw_object:
        mw_object = st.session_state.mw_object
        mw_object = mw_object.copy()

        eqi = st.session_state.get('eqi', None)
        ref = st.session_state.get('ref', 'le')
        time_win = st.session_state.get('time_win', 20)

        if eqi < 60:
            time_win = 20
        if eqi < 40:
            time_win = 10
        if eqi < 20:
            time_win = 5

        st.metric(f"EEG Quality Index", eqi)

        selected_ref_index = 0 if eqi > 60 else 1
        ref_options = [
            "linked ears",
            "centroid",
            "bipolar transverse",
            "bipolar longitudinal"
        ]

        ref = st.selectbox(
            "Choose EEG reference",
            options=ref_options,
            index=selected_ref_index
        )

        if ref == "linked ears":
            ref = "le"
        elif ref == "centroid":
            ref = "cz"
        elif ref == "bipolar transverse":
            ref = "btm"
        elif ref == "bipolar longitudinal":
            ref = "blm"
        else:
            ref = "tcp"

        time_win = st.number_input(
            "Enter time window (seconds)",
            min_value=3,
            max_value=30,
            value=time_win,
            step=5
        )


        with st.spinner("Running pipeline..."):
            pipeline = run_persist_pipeline(mw_object)
            pipeline.run(ref=ref, time_win=time_win)

        with st.spinner("Drawing all epochs..."):
            fig = pipeline.plot_3d_psd()
            st.plotly_chart(fig)
            pipeline.data['average_psds'] = pipeline.data['flattened_psds'].apply(lambda x: np.array(x).reshape(19,-1).mean(axis=0)[11:51])

            st.write("Epoch metadata")
            st.dataframe(pipeline.data, use_container_width=True, column_order = ['average_psds','sync_score', 'alpha', 'bads', 'n_bads'],  column_config={
                "average_psds": st.column_config.AreaChartColumn(label="Average PSD")
            })


        epoch_num = int(st.number_input(
            "Enter epoch number"
        ))
        if st.button("Generate epoch graph"):
            with st.spinner("Drawing..."):
                pipeline.combined_plot(epoch_num)
                pipeline.reset(mw_object)

        if st.button("Generate top 20 epoch graphs"):
            with st.spinner("Drawing.."):
                pipeline.generate_graphs()
                pipeline.reset(mw_object)

        pipeline.reset(mw_object)

    else:
        st.error("No EEG data available. Please upload an EEG file on the main page.")
