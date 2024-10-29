import streamlit as st

import utils.preprocess as preprocess

st.title("My App")
st.write("Hello from Streamlit!")

# Add a simple interactive element
if st.button("Click me"):
    st.write("Button clicked!")