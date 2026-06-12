import streamlit as st
import pandas as pd
import requests
st.session_sta
st.write("test ecriture 1")
st.title("Titre 1")
a=st.text_input("entrée")
b=st.slider("slider", 0, 100, 50)
if a=!="":
    st.write(f"Vous avez entré : {a} et sélectionné : {b}")
