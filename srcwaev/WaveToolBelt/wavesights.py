"""
create a class that inherits from MWA and uses MWA functions to e.g. generate power spectrums that then get used by WIM to analyze an EEG.

INSIGHTS
    1. return an EEG that has been run through autoreject
    2. given a patient folder, create a dictionary about the patients EEG (EEG_info)
        a. file paths for the EEG
        b. power spectrum data
    3. a window/app that displays three empty PSDs
        - the user can click point on the graph
            - a list of coordinates will display to the right of the 3 stacked graphs as points are clicked
        - after points have been entered, creating mock PSDs 
            - pressing "Run Comparison" will:
                - take every EEG_info dict and compare the PSDs from that EEG to the user entered PSDs in the app
                - create a csv of 
                    - PID, EEGID(if available), and all comparison statistics
        
"""

