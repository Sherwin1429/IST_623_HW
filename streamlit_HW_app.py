import streamlit as st

st.set_page_config(
    page_title="HW Manager",
    page_icon="📚"
)

hw1 = st.Page(
    "HW/HW1.py",
    title="Homework 1"
)

hw2 = st.Page(
    "HW/HW2.py",
    title="Homework 2",
    default=True
)

hw3 = st.Page(
    "HW/HW3.py",
    title="Homework 3"
)

pg = st.navigation([hw1, hw2, hw3])

pg.run()