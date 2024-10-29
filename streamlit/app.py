import streamlit as st

st.title("My App")
st.write("Hello from Streamlit!")

# Add a simple interactive element
if st.button("Click me"):
    st.write("Button clicked!")