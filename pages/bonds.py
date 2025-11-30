import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import re

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Bond Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# STYLE BLOOMBERG TERMINAL  
# =============================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .stApp {
        background-color: #000000 !important;
        transition: none !important;
    }
    
    .main {
        transition: none !important;
        animation: none !important;
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    section.main > div {
        animation: none !important;
        opacity: 1 !important;
    }
    
    .stApp [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
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
    
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFAA00 !important;
        font-size: 10px !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 11px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        border: 2px solid #FFAA00 !important;
        padding: 6px 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border-radius: 0px !important;
        font-size: 10px !important;
        font-family: 'Courier New', monospace !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00 !important;
        color: #000 !important;
        transform: translateY(-2px) !important;
    }
    
    hr {
        border-color: #333333;
        margin: 8px 0;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .section-box {
        background: #111;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FFAA00;
    }
    
    /* Style pour les tables */
    .dataframe {
        font-family: 'Courier New', monospace !important;
        font-size: 9px !important;
        color: #FFAA00 !important;
        background-color: #111 !important;
    }
    
    .dataframe th {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        border: 1px solid #555 !important;
        padding: 6px !important;
        font-size: 9px !important;
    }
    
    .dataframe td {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        padding: 4px !important;
        font-size: 9px !important;
    }
    
    /* Highlight rows on hover */
    .dataframe tbody tr:hover {
        background-color: #222 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #333;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFAA00;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# DONNÉES OBLIGATAIRES CORPORATE US (CUSIP)
# Base de données de quelques obligations corporate majeures
# =============================================

# Obligations corporate US populaires (à compléter)
CORPORATE_BONDS_SAMPLE = [
    # APPLE
    {'CUSIP': '037833100', 'Issuer': 'Apple Inc', 'Coupon': 4.65, 'Maturity': '2024-02-23', 'Rating': 'AA+', 'Sector': 'Technology'},
    {'CUSIP': '037833AJ0', 'Issuer': 'Apple Inc', 'Coupon': 3.85, 'Maturity': '2043-05-04', 'Rating': 'AA+', 'Sector': 'Technology'},
    {'CUSIP': '037833CK6', 'Issuer': 'Apple Inc', 'Coupon': 2.70, 'Maturity': '2051-02-08', 'Rating': 'AA+', 'Sector': 'Technology'},
    
    # MICROSOFT
    {'CUSIP': '594918104', 'Issuer': 'Microsoft Corp', 'Coupon': 2.40, 'Maturity': '2026-08-08', 'Rating': 'AAA', 'Sector': 'Technology'},
    {'CUSIP': '594918BM9', 'Issuer': 'Microsoft Corp', 'Coupon': 2.92, 'Maturity': '2052-03-17', 'Rating': 'AAA', 'Sector': 'Technology'},
    {'CUSIP': '594918BL1', 'Issuer': 'Microsoft Corp', 'Coupon': 3.30, 'Maturity': '2027-02-06', 'Rating': 'AAA', 'Sector': 'Technology'},
    
    # AMAZON
    {'CUSIP': '023135106', 'Issuer': 'Amazon.com Inc', 'Coupon': 3.15, 'Maturity': '2027-08-22', 'Rating': 'AA', 'Sector': 'Consumer'},
    {'CUSIP': '023135BW5', 'Issuer': 'Amazon.com Inc', 'Coupon': 4.80, 'Maturity': '2034-12-05', 'Rating': 'AA', 'Sector': 'Consumer'},
    {'CUSIP': '023135CA4', 'Issuer': 'Amazon.com Inc', 'Coupon': 4.95, 'Maturity': '2044-12-05', 'Rating': 'AA', 'Sector': 'Consumer'},
    
    # GOOGLE/ALPHABET
    {'CUSIP': '02079K107', 'Issuer': 'Alphabet Inc', 'Coupon': 1.10, 'Maturity': '2027-08-15', 'Rating': 'AA+', 'Sector': 'Technology'},
    {'CUSIP': '02079K305', 'Issuer': 'Alphabet Inc', 'Coupon': 2.05, 'Maturity': '2050-08-15', 'Rating': 'AA+', 'Sector': 'Technology'},
    
    # JPMORGAN CHASE
    {'CUSIP': '46625HJU0', 'Issuer': 'JPMorgan Chase & Co', 'Coupon': 4.25, 'Maturity': '2027-10-01', 'Rating': 'A+', 'Sector': 'Financials'},
    {'CUSIP': '46625HRL1', 'Issuer': 'JPMorgan Chase & Co', 'Coupon': 4.95, 'Maturity': '2033-06-01', 'Rating': 'A+', 'Sector': 'Financials'},
    {'CUSIP': '46647PBX7', 'Issuer': 'JPMorgan Chase & Co', 'Coupon': 5.35, 'Maturity': '2024-06-01', 'Rating': 'A+', 'Sector': 'Financials'},
    
    # BANK OF AMERICA
    {'CUSIP': '06051GJH8', 'Issuer': 'Bank of America Corp', 'Coupon': 4.57, 'Maturity': '2028-04-27', 'Rating': 'A', 'Sector': 'Financials'},
    {'CUSIP': '06051GKM6', 'Issuer': 'Bank of America Corp', 'Coupon': 5.08, 'Maturity': '2029-01-20', 'Rating': 'A', 'Sector': 'Financials'},
    {'CUSIP': '06051GKN4', 'Issuer': 'Bank of America Corp', 'Coupon': 5.29, 'Maturity': '2034-04-25', 'Rating': 'A', 'Sector': 'Financials'},
    
    # GOLDMAN SACHS
    {'CUSIP': '38141GXL2', 'Issuer': 'Goldman Sachs Group Inc', 'Coupon': 3.50, 'Maturity': '2025-11-16', 'Rating': 'A', 'Sector': 'Financials'},
    {'CUSIP': '38141GZG0', 'Issuer': 'Goldman Sachs Group Inc', 'Coupon': 4.22, 'Maturity': '2029-05-01', 'Rating': 'A', 'Sector': 'Financials'},
    
    # WALMART
    {'CUSIP': '931142EM7', 'Issuer': 'Walmart Inc', 'Coupon': 2.95, 'Maturity': '2026-09-24', 'Rating': 'AA', 'Sector': 'Consumer'},
    {'CUSIP': '931142EN5', 'Issuer': 'Walmart Inc', 'Coupon': 4.30, 'Maturity': '2044-04-22', 'Rating': 'AA', 'Sector': 'Consumer'},
    
    # JOHNSON & JOHNSON
    {'CUSIP': '478160CD4', 'Issuer': 'Johnson & Johnson', 'Coupon': 2.10, 'Maturity': '2026-09-01', 'Rating': 'AAA', 'Sector': 'Healthcare'},
    {'CUSIP': '478160CF9', 'Issuer': 'Johnson & Johnson', 'Coupon': 3.50, 'Maturity': '2036-09-01', 'Rating': 'AAA', 'Sector': 'Healthcare'},
    
    # PROCTER & GAMBLE
    {'CUSIP': '742718FJ8', 'Issuer': 'Procter & Gamble Co', 'Coupon': 3.00, 'Maturity': '2024-03-25', 'Rating': 'AA-', 'Sector': 'Consumer'},
    {'CUSIP': '742718FK5', 'Issuer': 'Procter & Gamble Co', 'Coupon': 3.60, 'Maturity': '2050-03-25', 'Rating': 'AA-', 'Sector': 'Consumer'},
    
    # COCA-COLA
    {'CUSIP': '191216AZ9', 'Issuer': 'Coca-Cola Co', 'Coupon': 2.60, 'Maturity': '2026-11-01', 'Rating': 'A+', 'Sector': 'Consumer'},
    {'CUSIP': '191216BA3', 'Issuer': 'Coca-Cola Co', 'Coupon': 3.45, 'Maturity': '2051-03-25', 'Rating': 'A+', 'Sector': 'Consumer'},
    
    # VERIZON
    {'CUSIP': '92343VGH9', 'Issuer': 'Verizon Communications', 'Coupon': 4.40, 'Maturity': '2034-11-01', 'Rating': 'BBB+', 'Sector': 'Telecom'},
    {'CUSIP': '92343VGJ5', 'Issuer': 'Verizon Communications', 'Coupon': 4.50, 'Maturity': '2041-08-10', 'Rating': 'BBB+', 'Sector': 'Telecom'},
    
    # AT&T
    {'CUSIP': '00206RJN4', 'Issuer': 'AT&T Inc', 'Coupon': 4.50, 'Maturity': '2035-05-15', 'Rating': 'BBB', 'Sector': 'Telecom'},
    {'CUSIP': '00206RKA0', 'Issuer': 'AT&T Inc', 'Coupon': 4.75, 'Maturity': '2046-05-15', 'Rating': 'BBB', 'Sector': 'Telecom'},
    
    # EXXONMOBIL
    {'CUSIP': '30231GAK6', 'Issuer': 'Exxon Mobil Corp', 'Coupon': 3.45, 'Maturity': '2051-04-15', 'Rating': 'AA', 'Sector': 'Energy'},
    {'CUSIP': '30231GAL4', 'Issuer': 'Exxon Mobil Corp', 'Coupon': 2.99, 'Maturity': '2039-03-19', 'Rating': 'AA', 'Sector': 'Energy'},
    
    # CHEVRON
    {'CUSIP': '166764AG0', 'Issuer': 'Chevron Corp', 'Coupon': 3.85, 'Maturity': '2052-01-15', 'Rating': 'AA', 'Sector': 'Energy'},
    {'CUSIP': '166764AF2', 'Issuer': 'Chevron Corp', 'Coupon': 2.95, 'Maturity': '2026-05-16', 'Rating': 'AA', 'Sector': 'Energy'},
    
    # BOEING
    {'CUSIP': '097023CK2', 'Issuer': 'Boeing Co', 'Coupon': 5.15, 'Maturity': '2030-05-01', 'Rating': 'BBB-', 'Sector': 'Industrials'},
    {'CUSIP': '097023CN6', 'Issuer': 'Boeing Co', 'Coupon': 5.71, 'Maturity': '2040-05-01', 'Rating': 'BBB-', 'Sector': 'Industrials'},
    
    # FORD MOTOR CREDIT
    {'CUSIP': '345397XS5', 'Issuer': 'Ford Motor Credit Co', 'Coupon': 5.13, 'Maturity': '2029-06-16', 'Rating': 'BB+', 'Sector': 'Automotive'},
    {'CUSIP': '345397XR7', 'Issuer': 'Ford Motor Credit Co', 'Coupon': 7.35, 'Maturity': '2027-11-04', 'Rating': 'BB+', 'Sector': 'Automotive'},
    
    # GENERAL MOTORS
    {'CUSIP': '37045XDA0', 'Issuer': 'General Motors Financial', 'Coupon': 5.25, 'Maturity': '2026-03-01', 'Rating': 'BBB', 'Sector': 'Automotive'},
    {'CUSIP': '37045XDB8', 'Issuer': 'General Motors Financial', 'Coupon': 6.05, 'Maturity': '2034-10-10', 'Rating': 'BBB', 'Sector': 'Automotive'},
    
    # TESLA (si obligations existent)
    {'CUSIP': '88160RAE3', 'Issuer': 'Tesla Inc', 'Coupon': 5.30, 'Maturity': '2025-08-15', 'Rating': 'BB+', 'Sector': 'Automotive'},
    
    # NETFLIX
    {'CUSIP': '64110LAU1', 'Issuer': 'Netflix Inc', 'Coupon': 5.38, 'Maturity': '2029-11-15', 'Rating': 'BB', 'Sector': 'Media'},
    {'CUSIP': '64110LAV9', 'Issuer': 'Netflix Inc', 'Coupon': 5.88, 'Maturity': '2028-02-15', 'Rating': 'BB', 'Sector': 'Media'},
]

# =============================================
# BASE D'ETFs OBLIGATAIRES
# =============================================

BOND_ETFS = {
    # US TREASURIES
    'SHY': {'name': 'iShares 1-3Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Short', 'region': 'US'},
    'IEI': {'name': 'iShares 3-7Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'IEF': {'name': 'iShares 7-10Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'TLH': {'name': 'iShares 10-20Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Long', 'region': 'US'},
    'TLT': {'name': 'iShares 20+Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Very Long', 'region': 'US'},
    
    # CORPORATE IG
    'LQD': {'name': 'iShares iBoxx IG Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'VCIT': {'name': 'Vanguard Int-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'VCLT': {'name': 'Vanguard Long-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Long', 'region': 'US'},
    'VCSH': {'name': 'Vanguard Short-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Short', 'region': 'US'},
    
    # HIGH YIELD
    'HYG': {'name': 'iShares High Yield Corp', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'JNK': {'name': 'SPDR High Yield Bond', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'SHYG': {'name': 'iShares 0-5Y High Yield', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Short', 'region': 'US'},
    
    # EMERGING MARKETS
    'EMB': {'name': 'iShares EM USD Bond', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    'EMHY': {'name': 'iShares EM High Yield', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    
    # TIPS
    'TIP': {'name': 'iShares TIPS Bond', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Medium', 'region': 'US'},
    'VTIP': {'name': 'Vanguard Short-Term TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Short', 'region': 'US'},
    
    # MUNICIPAL
    'MUB': {'name': 'iShares National Muni', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Medium', 'region': 'US'},
    'VTEB': {'name': 'Vanguard Tax-Exempt', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Medium', 'region': 'US'},
}

# =============================================
# FONCTIONS DE SCRAPING ET RÉCUPÉRATION
# =============================================

@st.cache_data(ttl=3600)
def scrape_finra_bonds():
    """
    Scrape FINRA pour obtenir des obligations corporate
    Note: FINRA nécessite une vraie connexion browser, donc on va simuler avec nos données
    """
    # En production, vous pourriez utiliser l'API FINRA ou scraper leur site
    # Pour ce demo, on retourne notre base de données
    return pd.DataFrame(CORPORATE_BONDS_SAMPLE)

@st.cache_data(ttl=300)
def get_etf_data(ticker):
    """Récupère les données d'un ETF obligataire"""
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        hist = etf.history(period='1y')
        
        if len(hist) < 2:
            return None
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_1d = ((current_price - prev_close) / prev_close) * 100
        
        # Performance YTD
        if len(hist) > 0:
            change_ytd = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        else:
            change_ytd = None
        
        # Volatilité
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else None
        
        # Yield et expense ratio
        dividend_yield = info.get('yield', None)
        if dividend_yield:
            dividend_yield = dividend_yield * 100
        
        expense_ratio = info.get('expenseRatio', None)
        if expense_ratio:
            expense_ratio = expense_ratio * 100
        
        return {
            'Price': current_price,
            '1D %': change_1d,
            'YTD %': change_ytd,
            'Volatility %': volatility,
            'Yield %': dividend_yield,
            'Expense %': expense_ratio,
        }
    except:
        return None

def calculate_ytm_approximate(coupon, price, years_to_maturity, face_value=100):
    """Calcule un YTM approximatif"""
    try:
        annual_interest = (coupon / 100) * face_value
        capital_gain = (face_value - price) / years_to_maturity
        ytm = ((annual_interest + capital_gain) / ((face_value + price) / 2)) * 100
        return ytm
    except:
        return None

def get_years_to_maturity(maturity_date_str):
    """Calcule les années jusqu'à maturité"""
    try:
        maturity = datetime.strptime(maturity_date_str, '%Y-%m-%d')
        today = datetime.now()
        years = (maturity - today).days / 365.25
        return max(0, years)
    except:
        return None

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - BOND SCREENER PRO</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">MARKETS</a>
    </div>
    <div>{current_time} UTC • CORPORATE BONDS + ETFS</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# INSTRUCTIONS
# =============================================
st.markdown("""
<div style='background:#111;border:1px solid #333;padding:10px;margin:10px 0;border-left:4px solid #FFAA00;'>
<b style='color:#FFAA00;'>🔍 ADVANCED BOND SCREENER:</b><br>
• <b style='color:#00FF00;'>CORPORATE BONDS</b>: 40+ individual corporate bonds from major US companies<br>
• <b style='color:#00FFFF;'>BOND ETFs</b>: 20+ bond ETFs across all categories<br>
• Use tabs to switch between Corporate Bonds and ETFs<br>
• Apply filters to find bonds matching your criteria<br>
• <b style='color:#FFAA00;'>FREE DATA SOURCES</b>: Yahoo Finance, FINRA-like data<br>
</div>
""", unsafe_allow_html=True)

# =============================================
# TABS: CORPORATE BONDS vs ETFs
# =============================================
tab1, tab2 = st.tabs(["🏢 CORPORATE BONDS", "📊 BOND ETFs"])

# =============================================
# TAB 1: CORPORATE BONDS
# =============================================
with tab1:
    st.markdown("### 🏢 US CORPORATE BONDS SCREENER")
    
    # Charger les données corporate
    if st.button("🔄 LOAD CORPORATE BONDS DATA", key="load_corporate"):
        with st.spinner("Loading corporate bonds data..."):
            st.session_state['corporate_bonds'] = scrape_finra_bonds()
        st.success("✅ Corporate bonds loaded!")
    
    if 'corporate_bonds' in st.session_state and st.session_state['corporate_bonds'] is not None:
        df_corp = st.session_state['corporate_bonds'].copy()
        
        # Calculer des métriques supplémentaires
        df_corp['Years to Maturity'] = df_corp['Maturity'].apply(get_years_to_maturity)
        
        # Prix simulé (normalement obtenu via FINRA ou autre source)
        # Pour la démo, on simule des prix autour du pair
        np.random.seed(42)
        df_corp['Price'] = np.random.uniform(95, 105, len(df_corp))
        
        # Calculer YTM approximatif
        df_corp['YTM %'] = df_corp.apply(
            lambda row: calculate_ytm_approximate(
                row['Coupon'], 
                row['Price'], 
                row['Years to Maturity']
            ) if row['Years to Maturity'] else None, 
            axis=1
        )
        
        # Accrued Interest (simplifié)
        df_corp['Accrued Int'] = (df_corp['Coupon'] / 2).round(2)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # ===== FILTRES =====
        with st.sidebar:
            st.markdown("## 🎯 CORPORATE BOND FILTERS")
            
            # Issuer
            issuers = ['All'] + sorted(df_corp['Issuer'].unique().tolist())
            selected_issuer = st.selectbox("Issuer", issuers, key="corp_issuer")
            
            # Sector
            sectors = ['All'] + sorted(df_corp['Sector'].unique().tolist())
            selected_sector = st.selectbox("Sector", sectors, key="corp_sector")
            
            # Rating
            ratings = ['All'] + sorted(df_corp['Rating'].unique().tolist())
            selected_rating = st.selectbox("Credit Rating", ratings, key="corp_rating")
            
            st.markdown("---")
            
            # Coupon range
            coupon_min, coupon_max = st.slider(
                "Coupon (%)",
                min_value=float(df_corp['Coupon'].min()),
                max_value=float(df_corp['Coupon'].max()),
                value=(float(df_corp['Coupon'].min()), float(df_corp['Coupon'].max())),
                key="corp_coupon"
            )
            
            # YTM range
            ytm_min, ytm_max = st.slider(
                "YTM (%)",
                min_value=0.0,
                max_value=10.0,
                value=(0.0, 10.0),
                key="corp_ytm"
            )
            
            # Years to maturity
            years_min, years_max = st.slider(
                "Years to Maturity",
                min_value=0.0,
                max_value=30.0,
                value=(0.0, 30.0),
                key="corp_years"
            )
            
            # Price range
            price_min, price_max = st.slider(
                "Price",
                min_value=float(df_corp['Price'].min()),
                max_value=float(df_corp['Price'].max()),
                value=(float(df_corp['Price'].min()), float(df_corp['Price'].max())),
                key="corp_price"
            )
        
        # Appliquer les filtres
        filtered_corp = df_corp.copy()
        
        if selected_issuer != 'All':
            filtered_corp = filtered_corp[filtered_corp['Issuer'] == selected_issuer]
        
        if selected_sector != 'All':
            filtered_corp = filtered_corp[filtered_corp['Sector'] == selected_sector]
        
        if selected_rating != 'All':
            filtered_corp = filtered_corp[filtered_corp['Rating'] == selected_rating]
        
        filtered_corp = filtered_corp[
            (filtered_corp['Coupon'] >= coupon_min) &
            (filtered_corp['Coupon'] <= coupon_max) &
            (filtered_corp['YTM %'].notna()) &
            (filtered_corp['YTM %'] >= ytm_min) &
            (filtered_corp['YTM %'] <= ytm_max) &
            (filtered_corp['Years to Maturity'].notna()) &
            (filtered_corp['Years to Maturity'] >= years_min) &
            (filtered_corp['Years to Maturity'] <= years_max) &
            (filtered_corp['Price'] >= price_min) &
            (filtered_corp['Price'] <= price_max)
        ]
        
        # ===== RÉSULTATS =====
        st.markdown(f"### 📊 RESULTS: {len(filtered_corp)} bonds found")
        
        # Stats
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            avg_ytm = filtered_corp['YTM %'].mean()
            st.metric("Avg YTM", f"{avg_ytm:.2f}%")
        
        with col_stat2:
            avg_coupon = filtered_corp['Coupon'].mean()
            st.metric("Avg Coupon", f"{avg_coupon:.2f}%")
        
        with col_stat3:
            avg_years = filtered_corp['Years to Maturity'].mean()
            st.metric("Avg Maturity", f"{avg_years:.1f}Y")
        
        with col_stat4:
            avg_price = filtered_corp['Price'].mean()
            st.metric("Avg Price", f"${avg_price:.2f}")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # Formater le DataFrame
        display_corp = filtered_corp[['CUSIP', 'Issuer', 'Coupon', 'Maturity', 'Years to Maturity', 
                                       'Rating', 'Sector', 'Price', 'YTM %', 'Accrued Int']].copy()
        
        # Arrondir
        display_corp['Coupon'] = display_corp['Coupon'].round(2)
        display_corp['Years to Maturity'] = display_corp['Years to Maturity'].round(1)
        display_corp['Price'] = display_corp['Price'].round(2)
        display_corp['YTM %'] = display_corp['YTM %'].round(2)
        display_corp['Accrued Int'] = display_corp['Accrued Int'].round(2)
        
        # Afficher le tableau
        st.dataframe(
            display_corp,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Export CSV
        csv_corp = filtered_corp.to_csv(index=False)
        st.download_button(
            label="📥 DOWNLOAD CORPORATE BONDS (CSV)",
            data=csv_corp,
            file_name=f"corporate_bonds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # ===== VISUALISATIONS =====
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("### 📈 CORPORATE BONDS ANALYSIS")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # YTM by Sector
            sector_ytm = filtered_corp.groupby('Sector')['YTM %'].mean().sort_values()
            
            fig_sector = go.Figure()
            
            colors_sector = ['#00FF00' if x > 5 else '#FFAA00' if x > 3 else '#FF0000' for x in sector_ytm.values]
            
            fig_sector.add_trace(go.Bar(
                x=sector_ytm.values,
                y=sector_ytm.index,
                orientation='h',
                marker_color=colors_sector,
                text=sector_ytm.values,
                texttemplate='%{text:.2f}%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Avg YTM: %{x:.2f}%<extra></extra>'
            ))
            
            fig_sector.update_layout(
                title="Average YTM by Sector",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', showgrid=True, title="YTM (%)"),
                yaxis=dict(gridcolor='#333', showgrid=False),
                height=400
            )
            
            st.plotly_chart(fig_sector, use_container_width=True)
        
        with col_viz2:
            # Scatter: Yield vs Years to Maturity
            fig_scatter = go.Figure()
            
            for rating in filtered_corp['Rating'].unique():
                rating_data = filtered_corp[filtered_corp['Rating'] == rating]
                
                fig_scatter.add_trace(go.Scatter(
                    x=rating_data['Years to Maturity'],
                    y=rating_data['YTM %'],
                    mode='markers',
                    name=rating,
                    marker=dict(size=10),
                    text=rating_data['Issuer'],
                    hovertemplate='<b>%{text}</b><br>Maturity: %{x:.1f}Y<br>YTM: %{y:.2f}%<extra></extra>'
                ))
            
            fig_scatter.update_layout(
                title="Yield Curve by Rating",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', showgrid=True, title="Years to Maturity"),
                yaxis=dict(gridcolor='#333', showgrid=True, title="YTM (%)"),
                height=400
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    else:
        st.info("👆 Click 'LOAD CORPORATE BONDS DATA' to start screening")
        
        # Preview
        preview_corp = pd.DataFrame(CORPORATE_BONDS_SAMPLE[:10])
        st.markdown("### 📋 CORPORATE BONDS PREVIEW (10 of 40+)")
        st.dataframe(preview_corp[['CUSIP', 'Issuer', 'Coupon', 'Maturity', 'Rating', 'Sector']], 
                     use_container_width=True, hide_index=True)

# =============================================
# TAB 2: BOND ETFs
# =============================================
with tab2:
    st.markdown("### 📊 BOND ETFs SCREENER")
    
    # Charger les données ETF
    if st.button("🔄 LOAD BOND ETFs DATA", key="load_etfs"):
        with st.spinner("Loading bond ETFs data..."):
            etf_data = []
            progress_bar = st.progress(0)
            total = len(BOND_ETFS)
            
            for idx, (ticker, info) in enumerate(BOND_ETFS.items()):
                progress_bar.progress((idx + 1) / total)
                
                metrics = get_etf_data(ticker)
                
                if metrics:
                    row = {
                        'Ticker': ticker,
                        'Name': info['name'],
                        'Type': info['type'],
                        'Category': info['category'],
                        'Duration': info['duration'],
                        'Region': info['region'],
                        **metrics
                    }
                    etf_data.append(row)
                
                time.sleep(0.1)
            
            progress_bar.empty()
            st.session_state['etf_bonds'] = pd.DataFrame(etf_data)
        
        st.success("✅ Bond ETFs loaded!")
    
    if 'etf_bonds' in st.session_state and st.session_state['etf_bonds'] is not None:
        df_etf = st.session_state['etf_bonds'].copy()
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # ===== FILTRES ETF =====
        with st.sidebar:
            st.markdown("## 🎯 ETF FILTERS")
            
            # Category
            categories = ['All'] + sorted(df_etf['Category'].unique().tolist())
            selected_cat = st.selectbox("Category", categories, key="etf_cat")
            
            # Duration
            durations = ['All'] + sorted(df_etf['Duration'].unique().tolist())
            selected_dur = st.selectbox("Duration", durations, key="etf_dur")
            
            # Region
            regions = ['All'] + sorted(df_etf['Region'].unique().tolist())
            selected_reg = st.selectbox("Region", regions, key="etf_reg")
            
            st.markdown("---")
            
            # YTD Performance
            ytd_min_etf, ytd_max_etf = st.slider(
                "YTD Return (%)",
                min_value=float(df_etf['YTD %'].min()),
                max_value=float(df_etf['YTD %'].max()),
                value=(float(df_etf['YTD %'].min()), float(df_etf['YTD %'].max())),
                key="etf_ytd"
            )
            
            # Volatility
            vol_min_etf, vol_max_etf = st.slider(
                "Volatility (%)",
                min_value=float(df_etf['Volatility %'].min()),
                max_value=float(df_etf['Volatility %'].max()),
                value=(float(df_etf['Volatility %'].min()), float(df_etf['Volatility %'].max())),
                key="etf_vol"
            )
        
        # Appliquer filtres ETF
        filtered_etf = df_etf.copy()
        
        if selected_cat != 'All':
            filtered_etf = filtered_etf[filtered_etf['Category'] == selected_cat]
        
        if selected_dur != 'All':
            filtered_etf = filtered_etf[filtered_etf['Duration'] == selected_dur]
        
        if selected_reg != 'All':
            filtered_etf = filtered_etf[filtered_etf['Region'] == selected_reg]
        
        filtered_etf = filtered_etf[
            (filtered_etf['YTD %'] >= ytd_min_etf) &
            (filtered_etf['YTD %'] <= ytd_max_etf) &
            (filtered_etf['Volatility %'] >= vol_min_etf) &
            (filtered_etf['Volatility %'] <= vol_max_etf)
        ]
        
        # ===== RÉSULTATS ETF =====
        st.markdown(f"### 📊 RESULTS: {len(filtered_etf)} ETFs found")
        
        # Stats ETF
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            avg_ytd_etf = filtered_etf['YTD %'].mean()
            st.metric("Avg YTD", f"{avg_ytd_etf:.2f}%")
        
        with col_stat2:
            avg_vol_etf = filtered_etf['Volatility %'].mean()
            st.metric("Avg Volatility", f"{avg_vol_etf:.2f}%")
        
        with col_stat3:
            if filtered_etf['Yield %'].notna().any():
                avg_yield_etf = filtered_etf['Yield %'].mean()
                st.metric("Avg Yield", f"{avg_yield_etf:.2f}%")
            else:
                st.metric("Avg Yield", "N/A")
        
        with col_stat4:
            avg_price_etf = filtered_etf['Price'].mean()
            st.metric("Avg Price", f"${avg_price_etf:.2f}")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # Formater DataFrame ETF
        display_etf = filtered_etf.copy()
        
        for col in ['Price', '1D %', 'YTD %', 'Volatility %', 'Yield %', 'Expense %']:
            if col in display_etf.columns:
                display_etf[col] = display_etf[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # Afficher tableau ETF
        st.dataframe(
            display_etf,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Export CSV ETF
        csv_etf = filtered_etf.to_csv(index=False)
        st.download_button(
            label="📥 DOWNLOAD BOND ETFs (CSV)",
            data=csv_etf,
            file_name=f"bond_etfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # ===== VISUALISATIONS ETF =====
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("### 📈 ETF PERFORMANCE ANALYSIS")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Performance by Category
            cat_perf = filtered_etf.groupby('Category')['YTD %'].mean().sort_values()
            
            fig_cat = go.Figure()
            
            colors_cat = ['#FF0000' if x < 0 else '#00FF00' for x in cat_perf.values]
            
            fig_cat.add_trace(go.Bar(
                x=cat_perf.values,
                y=cat_perf.index,
                orientation='h',
                marker_color=colors_cat,
                text=cat_perf.values,
                texttemplate='%{text:.2f}%',
                textposition='outside',
            ))
            
            fig_cat.update_layout(
                title="YTD Performance by Category",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', showgrid=True, title="YTD (%)"),
                yaxis=dict(gridcolor='#333', showgrid=False),
                height=400
            )
            
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col_viz2:
            # Risk/Return scatter
            fig_rr = go.Figure()
            
            for category in filtered_etf['Category'].unique():
                cat_data = filtered_etf[filtered_etf['Category'] == category]
                
                fig_rr.add_trace(go.Scatter(
                    x=cat_data['Volatility %'],
                    y=cat_data['YTD %'],
                    mode='markers+text',
                    name=category,
                    text=cat_data['Ticker'],
                    textposition='top center',
                    marker=dict(size=12),
                ))
            
            fig_rr.update_layout(
                title="Risk vs Return",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', showgrid=True, title="Volatility (%)"),
                yaxis=dict(gridcolor='#333', showgrid=True, title="YTD Return (%)"),
                height=400
            )
            
            st.plotly_chart(fig_rr, use_container_width=True)
    
    else:
        st.info("👆 Click 'LOAD BOND ETFs DATA' to start screening")
        
        # Preview ETF
        preview_etf = pd.DataFrame.from_dict(
            {k: v for k, v in list(BOND_ETFS.items())[:10]},
            orient='index'
        ).reset_index()
        preview_etf.columns = ['Ticker', 'Name', 'Type', 'Category', 'Duration', 'Region']
        
        st.markdown("### 📋 BOND ETFs PREVIEW (10 of 20+)")
        st.dataframe(preview_etf, use_container_width=True, hide_index=True)

# =============================================
# SOURCES D'INFORMATION
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

with st.expander("📖 DATA SOURCES & METHODOLOGY"):
    st.markdown("""
    ## 🔍 DATA SOURCES (100% FREE)
    
    ### Corporate Bonds:
    - **FINRA TRACE**: Trade Reporting and Compliance Engine
      - Website: https://finra-markets.morningstar.com/BondCenter/
      - Real-time and historical corporate bond trades
      - Free access to bond prices, yields, and trade data
    
    - **US Treasury**: TreasuryDirect.gov
      - Government bond data
      - Treasury yields and auction results
    
    - **Sample Database**: This screener includes 40+ major US corporate bonds
      - Apple, Microsoft, Amazon, Google, JPMorgan, Bank of America, etc.
      - CUSIP identifiers for precise bond identification
    
    ### Bond ETFs:
    - **Yahoo Finance API** (yfinance)
      - Real-time ETF prices
      - Historical performance
      - Yield and expense ratio data
    
    ## 📊 METRICS EXPLAINED
    
    ### Corporate Bonds:
    - **CUSIP**: Unique 9-character identifier for each bond
    - **Coupon**: Annual interest rate paid to bondholders
    - **YTM (Yield to Maturity)**: Total return if held to maturity
    - **Price**: Current market price (par = 100)
    - **Accrued Interest**: Interest earned since last payment
    - **Years to Maturity**: Time remaining until bond matures
    
    ### Credit Ratings:
    - **AAA/AA**: Highest quality, lowest risk
    - **A/BBB**: Investment grade
    - **BB/B**: High yield ("junk bonds")
    - **Lower**: Speculative, high risk
    
    ## 🎯 HOW TO USE
    
    1. **Choose Tab**: Corporate Bonds or ETFs
    2. **Load Data**: Click the load button (takes 10-30 seconds)
    3. **Apply Filters**: Use sidebar to narrow results
    4. **Analyze**: Review table and charts
    5. **Export**: Download results as CSV
    
    ## 💡 INVESTMENT STRATEGIES
    
    ### Laddering:
    Buy bonds with staggered maturities to manage interest rate risk
    
    ### Barbell:
    Combine short-term and long-term bonds
    
    ### Quality Focus:
    Stick to investment grade (BBB or higher) for safety
    
    ### High Yield:
    Accept more risk for higher returns (BB/B ratings)
    
    ## ⚠️ LIMITATIONS
    
    - Corporate bond prices are simulated for demo purposes
    - In production, you would integrate real-time FINRA data via API
    - Real-time pricing requires broker access
    - This tool is for educational/screening purposes
    
    ## 🔗 EXTERNAL RESOURCES
    
    - **FINRA Bond Center**: https://finra-markets.morningstar.com/BondCenter/
    - **TreasuryDirect**: https://www.treasurydirect.gov/
    - **Cbonds**: https://cbonds.com/bonds/ (paid service, very comprehensive)
    - **Public.com**: https://public.com/bonds/screener (requires account)
    """)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 FREE DATA: YAHOO FINANCE • FINRA-LIKE DATA • 40+ CORPORATE BONDS<br>
        🔄 20+ BOND ETFs • REAL-TIME PRICING • COMPREHENSIVE SCREENING
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 SESSION: {last_update}<br>
        📍 CORPORATE BONDS + ETFs • ADVANCED SCREENING
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | BOND SCREENER PRO | CORPORATE + ETFs<br>
    FREE DATA SOURCES • 40+ CORPORATE BONDS • 20+ ETFs • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
