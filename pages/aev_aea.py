import streamlit as st
import plotly.graph_objects as go
import pandas as pd


from srcwaev.WaveToolBelt import slwaevforms as waev


if "aev_aea_data" not in st.session_state:
    st.session_state.aev_df = None

if "aev_aea_channels" not in st.session_state:
    st.session_state.aev_channels = None

if "aev_aea_montage" not in st.session_state:
    st.session_state.aev_aea_montage = None

if "aev_aea_fname" not in st.session_state:
    st.session_state.aev_aea_fname = None

# Streamlit app setup
st.set_page_config(page_title="AEV - AEA", layout="wide")


"""

"""
def create_plotly_figure(df, channels, curr_montage, offset_value=1.0):
    # Initialize fig object
    fig = go.Figure()

    # Inverse channels
    channels = channels[::-1]

    # Add traces to fig
    for i, channel in enumerate(channels):
        offset = i * offset_value
        fig.add_trace(
            go.Scattergl(
                x=df["Time"],
                y=df[channel] + offset,
                mode='lines+markers',
                name=channel,
                line=dict(
                    color="black",
                    width=0.8,
                ),
                marker=dict(
                    size=2,
                    opacity=0.05,
                    color="#4200db", # dark purple
                ),
                # hovertemplate="%{x}<extra></extra>",  
            )
        )

    # Create custom y-axis tick labels and positions
    yticks = [i * offset_value for i in range(len(channels))]
    ytick_labels = channels

    # Format the fig
    fig.update_layout(
        # title='EEG',
        xaxis=dict(
            domain=[0.0, 1.0],
            rangeslider=dict(
                visible=True,
            ),
            range=[0, 20],  # Set this according to the full range of your data
            # type='linear',
        ),
        yaxis={
            "tickvals": yticks,
            "ticktext": ytick_labels,
            "tickmode": "array",
            "range": [-0.5, max(yticks) + offset_value]
        },
        # dragmode='select',
        height=800,
        # width=1600,
    )

    return fig



# Create a dropdown widget
curr_montage = st.selectbox(
    "Montage",
    ["a1a2", "cz", "bpt", "tcp", "avg", "ref"],
)


# Check if `mw_object` is available
if 'mw_object' in st.session_state and st.session_state.mw_object:
    mw_object = st.session_state.mw_object
    mw_object = mw_object.copy()

    def get_data(mw_object, curr_montage):
        eeg_dict = waev.get_data_from_mw_object(mw_object, picks="eeg", get_dict=True)
        df = eeg_dict[curr_montage]["df"]
        channels = eeg_dict[curr_montage]["channels"]
        return df, channels


    # If the montage didnt change, do nothing
    if st.session_state.get("aev_aea_montage", None) == curr_montage:
        print(f"MONTAGE IS THE SAME: {st.session_state.get('aev_aea_montage', None)}, {curr_montage}")
        
    # If the current montage is different than the previous montage
    elif st.session_state.get("aev_aea_montage", None) != curr_montage:
        print(f"MONTAGE CHANGED: {st.session_state.get('aev_aea_montage', None)}, {curr_montage}")
        df, channels = get_data(mw_object, curr_montage)
        df = waev.normalize_dataframe(df)

        st.session_state.aev_aea_df = df
        st.session_state.aev_aea_channels = channels
        st.session_state.aev_aea_montage = curr_montage
    

    # Create plot
    # fig = create_plotly_figure(df, channels, curr_montage)
    fig = create_plotly_figure(
        st.session_state.get('aev_aea_df', None), 
        st.session_state.get('aev_aea_channels', None), 
        st.session_state.get('aev_aea_montage', None),
    )

    # Create plotly event
    event = st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select="rerun",
        selection_mode="points",
    )

    # Create Streamlit Dataframe
    st.dataframe(
        event["selection"]["points"][:],
        key="aea_collections_df",
        column_config=dict(
            curve_number=None,
            point_number=None,
            point_index=None,
            y=None,
            x="     Onset     "
        ),

    )

