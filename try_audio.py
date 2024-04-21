import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.sound import sound

st.title("some title")
if st.button('Record'):
    with st.spinner(f'Recording for {DURATION} seconds ....'):
        sound.record()
    st.success("Recording completed")