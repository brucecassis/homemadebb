import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Bond Screener",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# STYLE BLOOMBERG
# =============================================
st.markdown("""
<style>
    .stApp { background-color: #000000 !important; }
    .main { background-color: #000000 !important; color: #FFAA00 !important; }
    h1, h2, h3, h4, p, div, span { font-family: 'Courier New', monospace !important; color: #FFAA00 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #333; }
    .stButton > button { border: 1px solid #FFAA00; background: #111; color: #FFAA00; border-radius: 0; }
    .stButton > button:hover { background: #FFAA00; color: #000; }
    /* Spinner custom */
    .stSpinner > div { border-top-color: #FFAA00 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================
# MOTEUR DE DONNÉES : LE SCRAPER
# =============================================

@st.cache_data(ttl=300) # Cache de 5 minutes pour éviter de se faire bannir par le site
def get_real_bonds_data():
    """
    Scrape les données réelles des obligations depuis Markets Insider.
    C'est la seule méthode gratuite viable actuellement.
    """
    url = "https://markets.businessinsider.com/bonds"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return pd.DataFrame() # Retourne vide si erreur
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Le site a souvent plusieurs tables, on cherche celle des "Most Active" ou "Top Yields"
        # Cette structure dépend du HTML actuel de Business Insider
        tables = soup.find_all('table', class_='table')
        
        bonds_list = []
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]: # Skip header
                cols = row.find_all('td')
                if len(cols) >= 4:
                    try:
                        # Extraction des données brutes
                        name = cols[0].text.strip()
                        country = "N/A" # Pas toujours dispo direct
                        
                        # Nettoyage des valeurs (enlever %, \n, etc)
                        price_txt = cols[1].text.strip().replace(',', '')
                        yield_txt = cols[3].text.strip().replace('%', '').replace(',', '')
                        date_txt = cols[5].text.strip() if len(cols) > 5 else "N/A"
                        
                        # Conversion sécurisée
                        price = float(price_txt) if price_txt and price_txt != '-' else 0.0
                        ytm = float(yield_txt) if yield_txt and yield_txt != '-' else 0.0
                        
                        # Déduction du type (Simpliste pour l'exemple)
                        bond_type = "GOV" if "Treasury" in name or "Bund" in name else "CORP"
                        
                        bonds_list.append({
                            "Issuer": name,
                            "Price": price,
                            "YTM (%)": ytm,
                            "Change": cols[2].text.strip(),
                            "Maturity": date_txt,
                            "Type": bond_type,
                            "Rating": "N/A" # Pas dispo en public gratuit sans clic
                        })
                    except:
                        continue
                        
        df = pd.DataFrame(bonds_list)
        
        # Si le scraping échoue (structure HTML changée), on met une backup data
        if df.empty:
            return get_backup_data()
            
        return df
        
    except Exception as e:
        st.error(f"Erreur de connexion aux données marchés: {e}")
        return get_backup_data()

def get_backup_data():
    """Données de secours si le scraping échoue"""
    return pd.DataFrame([
        {'Issuer': 'US TREASURY NOTE 10Y', 'Price': 98.50, 'YTM (%)': 4.25, 'Change': '+0.10', 'Maturity': '2034', 'Type': 'GOV'},
        {'Issuer': 'GERMANY BUND 10Y', 'Price': 99.10, 'YTM (%)': 2.35, 'Change': '-0.05', 'Maturity': '2034', 'Type': 'GOV'},
        {'Issuer': 'APPLE INC 2028', 'Price': 97.40, 'YTM (%)': 4.10, 'Change': '0.00', 'Maturity': '2028', 'Type': 'CORP'},
    ])

# =============================================
# INTERFACE
# =============================================

# HEADER
st.markdown("""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;">
    <div>📡 REAL-TIME BOND FEED (WEB)</div>
    <div style="font-size:10px">SOURCE: MARKETS INSIDER</div>
</div>
<br>
""", unsafe_allow_html=True)

# BOUTON REFRESH MANUEL (Car le scraping est lent)
col_btn, col_status = st.columns([1, 5])
with col_btn:
    if st.button("📥 FETCH DATA"):
        st.cache_data.clear() # Force le rechargement

with col_status:
    st.caption("Données extraites en temps réel. Cliquez pour actualiser.")

# CHARGEMENT DES DONNÉES
with st.spinner('Connexion aux marchés obligataires internationaux...'):
    df_bonds = get_real_bonds_data()

# STATISTIQUES RAPIDES
if not df_bonds.empty:
    avg_yield = df_bonds['YTM (%)'].mean()
    max_yield = df_bonds['YTM (%)'].max()
    count = len(df_bonds)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("OBLIGATIONS SUIVIES", count)
    c2.metric("RENDEMENT MOYEN", f"{avg_yield:.2f}%")
    c3.metric("TOP YIELD", f"{max_yield:.2f}%")

# SCREENER (FILTRES)
st.markdown("---")
col_search, col_sort = st.columns([3, 1])

with col_search:
    search = st.text_input("🔍 FILTRER PAR NOM (Ex: Treasury, Apple, Euro...)", "")

with col_sort:
    min_yield = st.number_input("YIELD MIN (%)", 0.0, 20.0, 3.0)

# FILTRAGE PANDAS
if not df_bonds.empty:
    filtered_df = df_bonds.copy()
    
    if search:
        filtered_df = filtered_df[filtered_df['Issuer'].str.contains(search, case=False)]
    
    filtered_df = filtered_df[filtered_df['YTM (%)'] >= min_yield]
    
    # TABLEAU
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Issuer": st.column_config.TextColumn("Nom de l'Obligation", width="large"),
            "Price": st.column_config.NumberColumn("Prix Actuel", format="$%.2f"),
            "YTM (%)": st.column_config.ProgressColumn(
                "Rendement (Yield)", 
                format="%.2f%%", 
                min_value=0, 
                max_value=10
            ),
            "Change": st.column_config.TextColumn("Var.", width="small"),
            "Maturity": "Maturité"
        },
        height=500
    )
else:
    st.error("Impossible de récupérer les données. Le site source est peut-être inaccessible.")

# FOOTER
st.markdown("---")
st.caption("NOTE TECHNIQUE: Ce module utilise du Web Scraping sur 'markets.businessinsider.com'. Si la structure HTML de leur site change, ce code devra être mis à jour.")
