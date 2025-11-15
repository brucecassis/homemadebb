import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import time

# Configuration de la page
st.set_page_config(
    page_title="EDGAR Terminal",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Bloomberg style
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    .bloomberg-header {
        background: #FFAA00;
        padding: 5px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 14px;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        border-bottom: 2px solid #FFAA00;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }
    
    h1, h2, h3, h4 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 12px !important;
        margin: 8px 0 !important;
        border-bottom: 1px solid #333;
        padding-bottom: 4px !important;
    }
    
    .stButton > button {
        background-color: #333;
        color: #FFAA00;
        font-weight: bold;
        border: 1px solid #FFAA00;
        padding: 6px 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0px;
        font-size: 10px;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000;
    }
    
    .stTextInput input {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        text-transform: uppercase;
    }
    
    .stTextInput input:focus {
        border-color: #FFF;
        box-shadow: 0 0 3px #FFAA00;
    }
    
    .stSelectbox select {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        font-family: 'Courier New', monospace;
    }
    
    .filing-item {
        background-color: #0a0a0a;
        border-left: 3px solid #FFAA00;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-bottom: 1px solid #222;
        font-family: 'Courier New', monospace;
    }
    
    .filing-type {
        color: #00FF00;
        font-size: 11px;
        font-weight: 700;
        margin: 0;
        line-height: 1.3;
    }
    
    .filing-date {
        color: #FFAA00;
        font-size: 10px;
        margin-top: 3px;
    }
    
    .filing-description {
        color: #999;
        font-size: 9px;
        margin-top: 3px;
    }
    
    hr {
        border-color: #333333;
        margin: 5px 0;
    }
    
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .dataframe {
        background-color: #000 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
    }
    
    .dataframe th {
        background-color: #111 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        border: 1px solid #333 !important;
    }
    
    .dataframe td {
        background-color: #000 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
    }
    
    a {
        color: #FFAA00 !important;
        text-decoration: none;
        font-weight: bold;
    }
    
    a:hover {
        color: #FFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - EDGAR FILINGS</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# Fonction pour récupérer les données SEC EDGAR
@st.cache_data(ttl=3600)
def get_company_cik(ticker):
    """Récupère le CIK d'une entreprise à partir de son ticker"""
    try:
        headers = {
            'User-Agent': 'VotreNom votre.email@exemple.com',  # ← METTEZ VOTRE EMAIL ICI
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        # Utiliser l'API officielle SEC pour les tickers
        cik_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(cik_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Chercher le ticker dans les données
            for key, value in data.items():
                if value['ticker'].upper() == ticker.upper():
                    cik = str(value['cik_str']).zfill(10)
                    company_name = value['title']
                    return cik, company_name
        
        return None, None
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération du CIK: {e}")
        return None, None

@st.cache_data(ttl=3600)
def get_company_filings(cik, filing_type='', count=20):
    """Récupère les filings SEC d'une entreprise"""
    try:
        headers = {
            'User-Agent': 'lightinyourcar@gmail.com',  # ← METTEZ VOTRE EMAIL ICI
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # Retirer les zéros au début pour l'URL
        cik_no_zeros = str(int(cik))
        
        # URL de l'API SEC EDGAR
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        time.sleep(0.1)  # Respecter les limites de rate de la SEC
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extraire les filings récents
            filings = data.get('filings', {}).get('recent', {})
            
            if filings:
                # Créer le DataFrame
                forms = filings.get('form', [])
                dates = filings.get('filingDate', [])
                accessions = filings.get('accessionNumber', [])
                primary_docs = filings.get('primaryDocument', [])
                descriptions = filings.get('primaryDocDescription', [])
                
                # S'assurer que toutes les listes ont la même longueur
                min_length = min(len(forms), len(dates), len(accessions), len(primary_docs))
                
                df = pd.DataFrame({
                    'Form Type': forms[:min_length],
                    'Filing Date': dates[:min_length],
                    'Accession Number': accessions[:min_length],
                    'Primary Document': primary_docs[:min_length],
                    'Description': descriptions[:min_length] if descriptions else ['N/A'] * min_length
                })
                
                # Filtrer par type si spécifié
                if filing_type and filing_type != 'ALL':
                    df = df[df['Form Type'] == filing_type]
                
                return df.head(count), cik_no_zeros
        
        return None, None
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des filings: {e}")
        return None, None

def create_filing_url(cik, accession_number, primary_doc):
    """Crée l'URL de téléchargement d'un filing"""
    # Nettoyer le numéro d'accession (enlever les tirets)
    accession_clean = accession_number.replace('-', '')
    
    # Utiliser le CIK sans zéros
    cik_clean = str(int(cik))
    
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{primary_doc}"
    return url

def download_filing_content(url):
    """Télécharge le contenu d'un filing"""
    try:
        headers = {
            'User-Agent': 'VotreNom votre.email@exemple.com',  # ← METTEZ VOTRE EMAIL ICI
        }
        time.sleep(0.1)  # Respecter le rate limit
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

# ===== INTERFACE PRINCIPALE =====
st.markdown("### 📋 SEC EDGAR FILINGS SEARCH")

# Zone de recherche
col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    ticker_input = st.text_input(
        "TICKER SYMBOL",
        placeholder="Ex: AAPL, TSLA, MSFT...",
        help="Entrez le ticker de l'entreprise",
        key="ticker_edgar"
    ).upper()

with col2:
    filing_types = ['ALL', '10-K', '10-Q', '8-K', '20-F', 'DEF 14A', 'S-1', '4', '13F-HR', 'S-3', 'S-4', '3', '424B2', '424B5']
    filing_type_select = st.selectbox(
        "FORM TYPE",
        options=filing_types,
        help="Type de formulaire SEC"
    )

with col3:
    search_button = st.button("🔍 SEARCH", use_container_width=True, key="search_edgar")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)

# Traitement de la recherche
if search_button and ticker_input:
    with st.spinner(f'🔍 Recherche des filings pour {ticker_input}...'):
        # Récupérer le CIK
        cik, company_name = get_company_cik(ticker_input)
        
        if cik and company_name:
            st.success(f"✅ **{company_name}** (CIK: {cik})")
            
            # Récupérer les filings
            filing_type_filter = filing_type_select if filing_type_select != 'ALL' else ''
            result = get_company_filings(cik, filing_type_filter, count=50)
            
            if result and result[0] is not None:
                filings_df, cik_clean = result
                
                if len(filings_df) > 0:
                    st.markdown(f"### 📊 FILINGS FOUND: {len(filings_df)}")
                    
                    # Afficher chaque filing
                    for idx, row in filings_df.iterrows():
                        filing_url = create_filing_url(cik_clean, row['Accession Number'], row['Primary Document'])
                        
                        # Conteneur pour chaque filing
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"""
                                <div class="filing-item">
                                    <div class="filing-type">{row['Form Type']}</div>
                                    <div class="filing-date">📅 {row['Filing Date']}</div>
                                    <div class="filing-description">{row['Description']}</div>
                                    <div style="color: #666; font-size: 9px; margin-top: 5px;">Acc. No: {row['Accession Number']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                # Lien VIEW
                                st.markdown(f"[📄 VIEW FILING]({filing_url})", unsafe_allow_html=True)
                                
                                # Télécharger le contenu
                                filing_content = download_filing_content(filing_url)
                                if filing_content:
                                    st.download_button(
                                        label="💾 DOWNLOAD",
                                        data=filing_content,
                                        file_name=f"{ticker_input}_{row['Form Type']}_{row['Filing Date']}.html",
                                        mime="text/html",
                                        use_container_width=True,
                                        key=f"download_{idx}"
                                    )
                                else:
                                    st.button("💾 ERROR", disabled=True, use_container_width=True, key=f"download_disabled_{idx}")
                            
                            st.markdown('<div style="border-bottom: 1px solid #222; margin: 8px 0;"></div>', unsafe_allow_html=True)
                    
                else:
                    st.warning("⚠️ Aucun filing trouvé pour ce ticker et ce type de formulaire.")
            else:
                st.warning("⚠️ Aucun filing trouvé pour ce ticker.")
        
        else:
            st.error(f"❌ Impossible de trouver le ticker '{ticker_input}'. Vérifiez l'orthographe.")

elif search_button:
    st.warning("⚠️ Veuillez entrer un ticker.")

# Info section
st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
st.markdown("### ℹ️ FILING TYPES INFORMATION")

filing_info = {
    '10-K': 'Annual Report - Rapport annuel complet',
    '10-Q': 'Quarterly Report - Rapport trimestriel',
    '8-K': 'Current Report - Événements importants',
    '20-F': 'Annual Report (Foreign) - Rapport annuel entreprises étrangères',
    'DEF 14A': 'Proxy Statement - Documents pour assemblées générales',
    'S-1': 'Registration Statement - Enregistrement IPO',
    '4': 'Insider Trading - Transactions des dirigeants',
    '13F-HR': 'Institutional Holdings - Positions des fonds',
    'S-3': 'Securities Registration - Enregistrement de titres',
    '424B5': 'Prospectus - Prospectus de placement'
}

cols_info = st.columns(2)
for idx, (form, description) in enumerate(filing_info.items()):
    with cols_info[idx % 2]:
        st.markdown(f"**{form}:** {description}")

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | SEC EDGAR DATABASE | LAST UPDATE: {last_update}
</div>
""", unsafe_allow_html=True)
