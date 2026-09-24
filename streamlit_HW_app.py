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
    title="Homework 2"
    
)

hw3 = st.Page(
    "HW/HW3.py",
    title="Homework 3"
   
)

hw4 = st.Page(
    "HW/HW4.py",
    title="Homework 4"
)

hw5 = st.Page(
    "HW/HW5.py",
    title="Homework 5",
    default=True
)

pg = st.navigation([hw1, hw2, hw3, hw4, hw5])

pg.run()