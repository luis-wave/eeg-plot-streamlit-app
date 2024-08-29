import streamlit as st
import plotly.graph_objects as go
import pandas as pd


from dsp import graph_preprocessing as waev
from graphs.eeg_viewer import draw_eeg_graph


def eeg_visualization_dashboard():
    if "aev_aea_montage" not in st.session_state:
        st.session_state.aev_aea_montage = None

    if "aev_aea_fname" not in st.session_state:
        st.session_state.aev_aea_fname = ""

    if "aev_montage_data" not in st.session_state:
        st.session_state.aev_montage_data = None

    # Streamlit app setup
    # st.set_page_config(page_title="AEV - AEA", layout="wide")



    # Check if `mw_object` is available
if 'mw_object' in st.session_state and st.session_state.mw_object:
    mw_object = st.session_state.get("mw_object", None)
    mw_object = mw_object.copy()

    # Create a dropdown widget
    curr_montage = st.selectbox(
        "Montage",
        ["a1a2", "cz", "bpt", "tcp", "avg", "ref"],
    )

    # Components
    with st.container() as row1:
        st.info(st.session_state.get("fname", "No EEG uploaded..."))

        # Create a dropdown widget
        curr_montage = st.selectbox(
            "Montage",
            ["a1a2", "cz", "bpt", "tcp", "avg", "ref"],
            label_visibility="collapsed",
        )

        # gets a dictionary of data for each montage, using session states if the montage is the same as previous
        def get_data(mw_object, curr_montage, streamlit_set_data=True, normalize_df=True):
            eeg_dict = waev.get_data_from_mw_object(mw_object, picks="eeg", get_dict=True, resample=64.0, normalize_df=normalize_df)
            
            if streamlit_set_data:
                st.session_state.aev_montage_data = eeg_dict
            else:
                df = eeg_dict[curr_montage]["df"]
                channels = eeg_dict[curr_montage]["channels"]
                return df, channels
                
        # If the current montage is different than the previous montage
        if (st.session_state.get("aev_aea_montage", None) != curr_montage) or (st.session_state.get("fname", None) != st.session_state.get("aev_aea_fname", None)):            
            # 
            if (st.session_state.get("fname", None) != st.session_state.get("aev_aea_fname", None)):
                get_data(mw_object, curr_montage, normalize_df=True) # df = waev.normalize_dataframe(df)

                # Reset the session state
                st.session_state.saved_onsets_collection = dict(selection=dict(points=[]))
                st.session_state.aev_aea_fname = st.session_state.get("fname", None)
            
            # Set the current montages data for visualization
            st.session_state.aev_aea_montage = curr_montage


        # Create plot
        # fig = create_plotly_figure(df, sfreq, channels, curr_montage)
        fig = draw_eeg_graph(
            st.session_state.aev_montage_data[curr_montage].get("df", None), 
            st.session_state.aev_montage_data[curr_montage].get("raw", None).info['sfreq'],
            st.session_state.aev_montage_data[curr_montage].get("channels", None), 
            curr_montage,
        )

        # Create plotly event
        event = st.plotly_chart(
            fig, 
            config=dict(
                scrollZoom=False,
                displayModeBar=True,
                displaylogo=False,
                modeBarButtonsToRemove=['zoom2d', 'pan2d', 'select2d', 'lasso2d', 
                                        'zoomIn2d', 'zoomOut2d', 'autoScale2d']
            ),
            use_container_width=True, 
            on_select="rerun",
            selection_mode="points",
            key="plotly_selection_onsets"
        )

    with st.container() as row2:
        col1, col2, col3 = st.columns(3)
        
        with col1:

            def save_onsets_button_click():
                # st.session_state.saved_onsets_collection = st.session_state.get('plotly_selection_onsets',  dict(selection=dict(points=[])))
                # st.session_state.saved_onsets_collection = st.session_state.get('plotly_selection_onsets',  dict(selection=dict(points=[])))

                print(st.session_state.get('plotly_selection_onsets',  dict(selection=dict(points=[]))))
            st.button("Save Onsets", type="primary", on_click=save_onsets_button_click)

            def auto_save_onsets():
                # st.session_state.saved_onsets_collection = 
                print('Auto saved selected onsets')
            
            # Streamlit Dataframe that displays selected onsets from plotly
            st.dataframe(    # st.data_editor(
                event["selection"]["points"][:],
                key="plot_collections_df",
                # num_rows="dynamic",
                column_config=dict(
                    curve_number=None,
                    point_number=None,
                    point_index=None,
                    y=None,
                    x="     Onsets     "
                ),
                # on_change=auto_save_onsets()
                on_select=auto_save_onsets,

            )
            # event

    with col2: 
        # Label for saved data
        st.markdown("### **Saved Onsets**")
        
        def add_to_text_area():
            # print("RUNNING CALLBACK --- add_to_text_area()")
            points = st.session_state['saved_onsets_collection']['selection'].get('points', [])
            # print(f"POINTS: {points}")
            
            # extract 'x' values from each point
            x_values = [point['x'] for point in points if 'x' in point]

            # reformat onsets
            formatted_onset_list = waev.reformat_timestamps(x_values)

            # list of onsets
            formatted_onset_string = ', '.join(map(str, formatted_onset_list))

            # create a comma-separated string of 'x' values
            st.session_state.onset_text_box = formatted_onset_string


        # def concat_saving_onsets():
        #     pass
        #     st.session_state.get(
        #         'saved_onsets_collection', dict(selection=dict(points=[]))
        #     )["selection"]["points"][:]

        # Create Streamlit Dataframe (data editor) 
        st.data_editor(
            st.session_state.get('saved_onsets_collection', dict(selection=dict(points=[])))["selection"]["points"][:],
            key="saved_plot_collections_df",
            num_rows="dynamic",
            column_config=dict(
                curve_number=None,
                point_number=None,
                point_index=None,
                y=None,
                x=" Saved Onsets "
            ),
            on_change=add_to_text_area()
        )

    with col3:
        # Label for saved data
        st.markdown("### **Listed Saved Onsets**")

        onset_text = st.text_area("loaded from the 'Saved Onsets' table", height=400, key="onset_text_box")

# To run the function as a Streamlit app
if __name__ == "__main__":
    eeg_visualization_dashboard()





