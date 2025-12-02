import streamlit as st
import streamlit.components.v1 as components

st.markdown("### 📊 TERMINAL INDICES BOURSIERS")

# Lire le fichier HTML
html_url = "https://raw.githubusercontent.com/brucecassis/pages_html/main/HOME.html"

import requests
response = requests.get(html_url)
html_content = response.text

# Afficher avec components
components.html(html_content, height=900, scrolling=True)
