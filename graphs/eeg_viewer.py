import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def draw_eeg_graph(df, sfreq, channels, curr_montage, offset_value=1.0):
    # Initialize fig object
    fig = go.Figure()

    # Inverse channels
    channels = channels[::-1]

    # Add traces to fig
    for i, channel in enumerate(channels):
        offset = i * offset_value
        fig.add_trace(
            go.Scattergl(
                x=df['Timestamps'],
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
    # print('\n', yticks, '\n')

    # Format the fig
    fig.update_layout(
        # title="",
        xaxis=dict(
            # domain=[0.0, 1.0],
            rangeslider=dict(
                visible=True,
                thickness=0.06,  # adjust thickness (0.1 means 10% of the plot height)
                range=[0, df.index[-1]], # range of beginning to end
                # range=[df['Times'].iloc[0], df['Times'].iloc[-1]],
            ),
            # Set this to the first 20 seconds of data -- df.index[df['Timestamps'] == '00:00:20.000'].tolist()[0]]
            range=[0, int(sfreq*20)],  
            nticks=10,
            # type='linear',
        ),
        yaxis={
            "tickvals": yticks,
            "ticktext": ytick_labels,
            "tickmode": "array",
            "range": [(-1.5), (max(yticks) + offset_value+0.5)],
            # 'autorange': False,
        },
        # sliders=sliders,
        # dragmode="select",SSSS
        legend=dict(
            traceorder="reversed",
            # itemsizing='constant'
            ),
        height=700,
        # width=1600,
        margin=dict(t=20,l=0,r=0,b=5) 
    )

    return fig
