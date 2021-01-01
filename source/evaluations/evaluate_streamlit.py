import streamlit as st

st.write("<style>red{color:red; text-decoration: underline;} orange{color:orange}</style>", unsafe_allow_html=True)


in_text = st.text_area("Input Text")
st.markdown(f"Output <red>text</red>:<sub>{in_text}</sub>", unsafe_allow_html=True)
