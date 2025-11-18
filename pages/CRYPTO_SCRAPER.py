import streamlit as st
from supabase import create_client, Client

st.set_page_config(
    page_title="Crypto Scraper",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CRYPTO DATA SCRAPER")

# Test de connexion Supabase
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    st.success("✅ Connected to Supabase!")
    
    # Afficher les datasets existants
    st.markdown("### 📂 Stored Datasets")
    
    response = supabase.table('crypto_datasets').select("*").execute()
    
    if response.data:
        st.dataframe(response.data)
    else:
        st.info("No datasets yet. Create your first one below!")
    
except Exception as e:
    st.error(f"❌ Connection error: {e}")
    st.info("Please add SUPABASE_URL and SUPABASE_KEY to your Streamlit secrets")
