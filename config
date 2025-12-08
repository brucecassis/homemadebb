import os
import streamlit as st

try:
    SUPABASE_URL = st.secrets["SUPABASE_AUTH_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_AUTH_KEY"]
except:
    SUPABASE_URL = os.getenv("SUPABASE_AUTH_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_AUTH_KEY")
