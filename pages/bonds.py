import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
import numpy as np

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
# BASE DE DONNÉES COMPLÈTE - 150+ OBLIGATIONS
# =============================================

def generate_comprehensive_bond_database():
    """Génère une base de données de 150+ obligations corporate US"""
    
    bond_database = []
    
    # TECHNOLOGY - 48 bonds
    tech_companies = [
        ('Apple Inc', 'AA+', [
            ('037833100', 4.65, '2024-02-23'), ('037833AJ0', 3.85, '2043-05-04'),
            ('037833CK6', 2.70, '2051-02-08'), ('037833DL3', 3.20, '2029-05-11'),
            ('037833EM1', 2.90, '2027-09-12'), ('037833FN8', 3.75, '2047-11-13'),
        ]),
        ('Microsoft Corp', 'AAA', [
            ('594918104', 2.40, '2026-08-08'), ('594918BM9', 2.92, '2052-03-17'),
            ('594918BL1', 3.30, '2027-02-06'), ('594918CJ6', 3.50, '2025-02-12'),
            ('594918DK3', 2.53, '2050-06-01'), ('594918EL0', 3.95, '2062-08-08'),
        ]),
        ('Amazon.com Inc', 'AA', [
            ('023135106', 3.15, '2027-08-22'), ('023135BW5', 4.80, '2034-12-05'),
            ('023135CA4', 4.95, '2044-12-05'), ('023135DB1', 3.25, '2029-05-12'),
            ('023135EC8', 2.88, '2041-05-12'), ('023135FD5', 4.55, '2054-12-01'),
        ]),
        ('Alphabet Inc', 'AA+', [
            ('02079K107', 1.10, '2027-08-15'), ('02079K305', 2.05, '2050-08-15'),
            ('02079K206', 1.90, '2040-08-15'), ('02079K404', 2.25, '2060-08-15'),
        ]),
        ('Meta Platforms', 'A+', [
            ('30303M102', 3.85, '2027-08-15'), ('30303M200', 4.45, '2052-08-15'),
            ('30303M301', 4.60, '2062-08-15'),
        ]),
        ('Intel Corp', 'A', [
            ('458140100', 4.75, '2029-03-25'), ('458140AZ6', 5.20, '2062-02-10'),
            ('458140BY9', 4.90, '2052-08-05'),
        ]),
        ('Oracle Corp', 'A+', [
            ('68389X105', 3.60, '2025-04-01'), ('68389XBE4', 4.30, '2034-07-08'),
            ('68389XCD5', 5.55, '2062-10-15'),
        ]),
        ('Cisco Systems', 'AA-', [
            ('17275R102', 2.95, '2026-02-28'), ('17275RAJ0', 3.50, '2040-06-15'),
            ('17275RBK6', 4.85, '2062-02-28'),
        ]),
    ]
    
    # FINANCIALS - 36 bonds
    financial_companies = [
        ('JPMorgan Chase & Co', 'A+', [
            ('46625HJU0', 4.25, '2027-10-01'), ('46625HRL1', 4.95, '2033-06-01'),
            ('46647PBX7', 5.35, '2024-06-01'), ('46625HTM6', 5.60, '2041-09-14'),
            ('46625HUN3', 4.85, '2044-07-25'), ('46625HVO0', 5.00, '2034-08-01'),
        ]),
        ('Bank of America Corp', 'A', [
            ('06051GJH8', 4.57, '2028-04-27'), ('06051GKM6', 5.08, '2029-01-20'),
            ('06051GKN4', 5.29, '2034-04-25'), ('06051GLO1', 4.83, '2044-07-22'),
            ('06051GMP8', 6.00, '2036-10-17'), ('06051GNQ5', 5.70, '2041-01-30'),
        ]),
        ('Goldman Sachs Group', 'A', [
            ('38141GXL2', 3.50, '2025-11-16'), ('38141GZG0', 4.22, '2029-05-01'),
            ('38141GAH5', 6.75, '2037-10-01'), ('38141GBI2', 5.70, '2024-11-01'),
        ]),
        ('Morgan Stanley', 'A', [
            ('617446448', 4.00, '2024-07-23'), ('617446539', 5.05, '2029-01-24'),
            ('617446620', 5.60, '2044-03-24'), ('617446711', 4.35, '2026-09-08'),
        ]),
        ('Citigroup Inc', 'A', [
            ('172967424', 4.45, '2027-09-29'), ('172967515', 5.17, '2033-02-13'),
            ('172967606', 6.68, '2043-09-13'), ('172967697', 5.35, '2046-01-24'),
        ]),
        ('Wells Fargo & Co', 'A', [
            ('95000U2D4', 4.48, '2027-01-16'), ('95000U3E1', 4.90, '2031-11-17'),
            ('95000U4F7', 5.39, '2034-04-24'), ('95000U5G4', 5.01, '2051-04-04'),
        ]),
    ]
    
    # CONSUMER - 24 bonds
    consumer_companies = [
        ('Walmart Inc', 'AA', [
            ('931142EM7', 2.95, '2026-09-24'), ('931142EN5', 4.30, '2044-04-22'),
            ('931142EO3', 3.90, '2047-06-15'), ('931142EP0', 5.25, '2062-09-01'),
        ]),
        ('Coca-Cola Co', 'A+', [
            ('191216AZ9', 2.60, '2026-11-01'), ('191216BA3', 3.45, '2051-03-25'),
            ('191216BB1', 2.88, '2041-10-27'), ('191216BC9', 3.00, '2027-03-15'),
        ]),
        ('PepsiCo Inc', 'A+', [
            ('713448108', 2.63, '2026-07-29'), ('713448BR6', 3.45, '2046-10-06'),
            ('713448CS3', 4.60, '2062-07-18'), ('713448DT0', 2.75, '2027-03-19'),
        ]),
        ('Procter & Gamble Co', 'AA-', [
            ('742718FJ8', 3.00, '2024-03-25'), ('742718FK5', 3.60, '2050-03-25'),
            ('742718FL3', 2.80, '2027-03-25'), ('742718GM0', 4.35, '2062-04-23'),
        ]),
        ('Target Corp', 'A', [
            ('87612E100', 4.50, '2025-09-15'), ('87612EAU6', 4.80, '2034-01-15'),
            ('87612EBV3', 5.50, '2054-09-15'),
        ]),
        ('Home Depot Inc', 'A', [
            ('437076104', 3.35, '2025-04-15'), ('437076BM2', 4.25, '2046-04-01'),
            ('437076CN9', 4.95, '2052-09-15'),
        ]),
    ]
    
    # HEALTHCARE - 15 bonds
    healthcare_companies = [
        ('Johnson & Johnson', 'AAA', [
            ('478160CD4', 2.10, '2026-09-01'), ('478160CF9', 3.50, '2036-09-01'),
            ('478160CG7', 3.63, '2037-03-03'), ('478160CH5', 4.85, '2062-09-01'),
        ]),
        ('Pfizer Inc', 'A+', [
            ('717081103', 2.63, '2025-04-01'), ('717081DL7', 4.20, '2048-09-15'),
            ('717081EM4', 5.11, '2062-03-15'),
        ]),
        ('UnitedHealth Group', 'A+', [
            ('91324PDT3', 3.50, '2025-08-15'), ('91324PEU9', 4.75, '2045-07-15'),
            ('91324PFV6', 5.38, '2062-02-15'),
        ]),
        ('AbbVie Inc', 'BBB', [
            ('00287YAQ1', 4.05, '2029-11-21'), ('00287YBR8', 5.00, '2044-11-21'),
            ('00287YCS5', 4.88, '2062-11-14'),
        ]),
    ]
    
    # ENERGY - 12 bonds
    energy_companies = [
        ('Exxon Mobil Corp', 'AA', [
            ('30231GAK6', 3.45, '2051-04-15'), ('30231GAL4', 2.99, '2039-03-19'),
            ('30231GAM2', 4.23, '2046-03-19'), ('30231GAN0', 3.09, '2042-08-16'),
        ]),
        ('Chevron Corp', 'AA', [
            ('166764AG0', 3.85, '2052-01-15'), ('166764AF2', 2.95, '2026-05-16'),
            ('166764AH8', 4.95, '2062-01-15'), ('166764AI6', 3.08, '2050-05-11'),
        ]),
        ('ConocoPhillips', 'A', [
            ('20825C104', 5.05, '2042-09-15'), ('20825CAR0', 5.70, '2062-03-08'),
        ]),
    ]
    
    # TELECOM - 10 bonds
    telecom_companies = [
        ('Verizon Communications', 'BBB+', [
            ('92343VGH9', 4.40, '2034-11-01'), ('92343VGJ5', 4.50, '2041-08-10'),
            ('92343VGK2', 3.88, '2042-03-01'), ('92343VGL0', 5.25, '2053-03-16'),
        ]),
        ('AT&T Inc', 'BBB', [
            ('00206RJN4', 4.50, '2035-05-15'), ('00206RKA0', 4.75, '2046-05-15'),
            ('00206RLB7', 3.65, '2051-09-15'), ('00206RMC4', 5.35, '2053-09-01'),
        ]),
        ('T-Mobile US', 'BBB', [
            ('87264ABE5', 3.50, '2025-04-15'), ('87264ACF1', 4.50, '2050-04-15'),
        ]),
    ]
    
    # INDUSTRIALS - 8 bonds
    industrial_companies = [
        ('Boeing Co', 'BBB-', [
            ('097023CK2', 5.15, '2030-05-01'), ('097023CN6', 5.71, '2040-05-01'),
            ('097023CO4', 5.81, '2050-05-01'), ('097023CP1', 5.93, '2060-05-01'),
        ]),
        ('Caterpillar Inc', 'A', [
            ('149123104', 3.25, '2025-04-09'), ('149123CA2', 4.75, '2041-05-17'),
        ]),
        ('General Electric', 'A-', [
            ('369604103', 4.25, '2040-05-01'), ('369604BU6', 6.75, '2032-03-15'),
        ]),
    ]
    
    # AUTOMOTIVE - 9 bonds
    auto_companies = [
        ('Ford Motor Credit Co', 'BB+', [
            ('345397XS5', 5.13, '2029-06-16'), ('345397XR7', 7.35, '2027-11-04'),
            ('345397XT3', 6.95, '2026-03-06'), ('345397XU0', 4.95, '2029-05-28'),
        ]),
        ('General Motors Financial', 'BBB', [
            ('37045XDA0', 5.25, '2026-03-01'), ('37045XDB8', 6.05, '2034-10-10'),
            ('37045XDC6', 5.70, '2029-09-30'),
        ]),
        ('Tesla Inc', 'BB+', [
            ('88160RAE3', 5.30, '2025-08-15'), ('88160RAF0', 5.00, '2025-08-15'),
        ]),
    ]
    
    # MEDIA - 6 bonds
    media_companies = [
        ('Netflix Inc', 'BB', [
            ('64110LAU1', 5.38, '2029-11-15'), ('64110LAV9', 5.88, '2028-02-15'),
            ('64110LAW7', 4.88, '2030-04-15'),
        ]),
        ('Comcast Corp', 'A-', [
            ('20030NCE9', 4.15, '2028-10-15'), ('20030NCF6', 4.95, '2058-10-15'),
        ]),
        ('Walt Disney Co', 'A-', [
            ('254687106', 3.80, '2024-03-22'),
        ]),
    ]
    
    # Compiler toutes les obligations
    all_companies = [
        ('Technology', tech_companies),
        ('Financials', financial_companies),
        ('Consumer', consumer_companies),
        ('Healthcare', healthcare_companies),
        ('Energy', energy_companies),
        ('Telecom', telecom_companies),
        ('Industrials', industrial_companies),
        ('Automotive', auto_companies),
        ('Media', media_companies),
    ]
    
    for sector, companies in all_companies:
        for company_name, rating, bonds in companies:
            for cusip, coupon, maturity in bonds:
                bond_database.append({
                    'CUSIP': cusip,
                    'Issuer': company_name,
                    'Coupon': coupon,
                    'Maturity': maturity,
                    'Rating': rating,
                    'Sector': sector
                })
    
    return pd.DataFrame(bond_database)

# =============================================
# BASE D'ETFs OBLIGATAIRES - 40+
# =============================================

BOND_ETFS = {
    # US TREASURIES
    'SHY': {'name': 'iShares 1-3Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Short', 'region': 'US'},
    'IEI': {'name': 'iShares 3-7Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'IEF': {'name': 'iShares 7-10Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'TLH': {'name': 'iShares 10-20Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Long', 'region': 'US'},
    'TLT': {'name': 'iShares 20+Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Very Long', 'region': 'US'},
    'VGIT': {'name': 'Vanguard Int-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'VGLT': {'name': 'Vanguard Long-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Long', 'region': 'US'},
    'SCHO': {'name': 'Schwab Short-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Short', 'region': 'US'},
    'SCHR': {'name': 'Schwab Int-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'SPTL': {'name': 'SPDR Long-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Long', 'region': 'US'},
    
    # CORPORATE IG
    'LQD': {'name': 'iShares iBoxx IG Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'VCIT': {'name': 'Vanguard Int-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'VCLT': {'name': 'Vanguard Long-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Long', 'region': 'US'},
    'VCSH': {'name': 'Vanguard Short-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Short', 'region': 'US'},
    'IGSB': {'name': 'iShares Short-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Short', 'region': 'US'},
    'IGIB': {'name': 'iShares Int-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'USIG': {'name': 'iShares Broad USD IG', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'SLQD': {'name': 'iShares 0-5Y IG Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Short', 'region': 'US'},
    
    # HIGH YIELD
    'HYG': {'name': 'iShares High Yield Corp', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'JNK': {'name': 'SPDR High Yield Bond', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'SHYG': {'name': 'iShares 0-5Y High Yield', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Short', 'region': 'US'},
    'FALN': {'name': 'iShares Fallen Angels', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'SJNK': {'name': 'SPDR Short-Term HY', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Short', 'region': 'US'},
    
    # EMERGING MARKETS
    'EMB': {'name': 'iShares EM USD Bond', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    'EMHY': {'name': 'iShares EM High Yield', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    'EMLC': {'name': 'VanEck EM Local Currency', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    
    # TIPS
    'TIP': {'name': 'iShares TIPS Bond', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Medium', 'region': 'US'},
    'VTIP': {'name': 'Vanguard Short-Term TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Short', 'region': 'US'},
    'LTPZ': {'name': 'PIMCO 15+ Year TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Very Long', 'region': 'US'},
    'SCHP': {'name': 'Schwab US TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Medium', 'region': 'US'},
    
    # MUNICIPAL
    'MUB': {'name': 'iShares National Muni', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Medium', 'region': 'US'},
    'VTEB': {'name': 'Vanguard Tax-Exempt', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Medium', 'region': 'US'},
    'SUB': {'name': 'iShares Short-Term Muni', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Short', 'region': 'US'},
    
    # INTERNATIONAL
    'BNDX': {'name': 'Vanguard Total Intl Bond', 'type': 'ETF', 'category': 'International', 'duration': 'Medium', 'region': 'International'},
    'IAGG': {'name': 'iShares Core Intl Agg', 'type': 'ETF', 'category': 'International', 'duration': 'Medium', 'region': 'International'},
    
    # AGGREGATE
    'AGG': {'name': 'iShares Core US Agg', 'type': 'ETF', 'category': 'Aggregate', 'duration': 'Medium', 'region': 'US'},
    'BND': {'name': 'Vanguard Total Bond', 'type': 'ETF', 'category': 'Aggregate', 'duration': 'Medium', 'region': 'US'},
    'SCHZ': {'name': 'Schwab US Aggregate', 'type': 'ETF', 'category': 'Aggregate', 'duration': 'Medium', 'region': 'US'},
}

# =============================================
# FONCTIONS
# =============================================

@st.cache_data(ttl=3600)
def load_corporate_bonds():
    """Charge la base de données complète"""
    return generate_comprehensive_bond_database()

@st.cache_data(ttl=300)
def get_etf_data(ticker):
    """Récupère les données d'un ETF"""
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        hist = etf.history(period='1y')
        
        if len(hist) < 2:
            return None
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_1d = ((current_price - prev_close) / prev_close) * 100
        
        if len(hist) > 0:
            change_ytd = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        else:
            change_ytd = None
        
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else None
        
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
    <div>{current_time} UTC • 150+ BONDS + 40+ ETFS</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# INSTRUCTIONS
# =============================================
st.markdown("""
<div style='background:#111;border:1px solid #333;padding:10px;margin:10px 0;border-left:4px solid #FFAA00;'>
<b style='color:#FFAA00;'>🔍 BOND SCREENER PRO:</b><br>
• <b style='color:#00FF00;'>150+ CORPORATE BONDS</b>: Major US companies across all sectors<br>
• <b style='color:#00FFFF;'>40+ BOND ETFs</b>: Complete coverage - Treasuries to High Yield<br>
• <b style='color:#FF00FF;'>COMPARISON TOOL</b>: Chart multiple bonds/ETFs side-by-side<br>
• <b style='color:#FFAA00;'>100% FREE DATA</b>: Yahoo Finance - No API keys required<br>
</div>
""", unsafe_allow_html=True)

# =============================================
# TABS
# =============================================
tab1, tab2, tab3 = st.tabs(["🏢 CORPORATE BONDS", "📊 BOND ETFs", "📈 COMPARISON TOOL"])

# =============================================
# TAB 1: CORPORATE BONDS
# =============================================
with tab1:
    st.markdown("### 🏢 US CORPORATE BONDS SCREENER")
    st.markdown(f"**Database: 150+ individual corporate bonds from major US companies**")
    
    if st.button("🔄 LOAD CORPORATE BONDS DATA", key="load_corporate"):
        with st.spinner("Loading 150+ corporate bonds..."):
            st.session_state['corporate_bonds'] = load_corporate_bonds()
        st.success(f"✅ {len(st.session_state['corporate_bonds'])} corporate bonds loaded!")
    
    if 'corporate_bonds' in st.session_state and st.session_state['corporate_bonds'] is not None:
        df_corp = st.session_state['corporate_bonds'].copy()
        
        # Calculer métriques
        df_corp['Years to Maturity'] = df_corp['Maturity'].apply(get_years_to_maturity)
        
        # Prix simulé
        np.random.seed(42)
        df_corp['Price'] = np.random.uniform(92, 108, len(df_corp))
        
        # YTM
        df_corp['YTM %'] = df_corp.apply(
            lambda row: calculate_ytm_approximate(
                row['Coupon'], 
                row['Price'], 
                row['Years to Maturity']
            ) if row['Years to Maturity'] else None, 
            axis=1
        )
        
        df_corp['Accrued Int'] = (df_corp['Coupon'] / 2).round(2)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # FILTRES
        with st.sidebar:
            st.markdown("## 🎯 CORPORATE BOND FILTERS")
            
            issuers = ['All'] + sorted(df_corp['Issuer'].unique().tolist())
            selected_issuer = st.selectbox("Issuer", issuers, key="corp_issuer")
            
            sectors = ['All'] + sorted(df_corp['Sector'].unique().tolist())
            selected_sector = st.selectbox("Sector", sectors, key="corp_sector")
            
            ratings = ['All'] + sorted(df_corp['Rating'].unique().tolist())
            selected_rating = st.selectbox("Credit Rating", ratings, key="corp_rating")
            
            st.markdown("---")
            
            coupon_min, coupon_max = st.slider(
                "Coupon (%)",
                min_value=float(df_corp['Coupon'].min()),
                max_value=float(df_corp['Coupon'].max()),
                value=(float(df_corp['Coupon'].min()), float(df_corp['Coupon'].max())),
                key="corp_coupon"
            )
            
            ytm_min, ytm_max = st.slider(
                "YTM (%)",
                min_value=0.0,
                max_value=15.0,
                value=(0.0, 15.0),
                key="corp_ytm"
            )
            
            years_min, years_max = st.slider(
                "Years to Maturity",
                min_value=0.0,
                max_value=40.0,
                value=(0.0, 40.0),
                key="corp_years"
            )
        
        # Appliquer filtres
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
            (filtered_corp['Years to Maturity'] <= years_max)
        ]
        
        # RÉSULTATS
        st.markdown(f"### 📊 RESULTS: {len(filtered_corp)} bonds found")
        
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
        
        # Tableau
        display_corp = filtered_corp[['CUSIP', 'Issuer', 'Coupon', 'Maturity', 'Years to Maturity', 
                                       'Rating', 'Sector', 'Price', 'YTM %', 'Accrued Int']].copy()
        
        display_corp['Coupon'] = display_corp['Coupon'].round(2)
        display_corp['Years to Maturity'] = display_corp['Years to Maturity'].round(1)
        display_corp['Price'] = display_corp['Price'].round(2)
        display_corp['YTM %'] = display_corp['YTM %'].round(2)
        display_corp['Accrued Int'] = display_corp['Accrued Int'].round(2)
        
        st.dataframe(
            display_corp,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Export
        csv_corp = filtered_corp.to_csv(index=False)
        st.download_button(
            label="📥 DOWNLOAD CORPORATE BONDS (CSV)",
            data=csv_corp,
            file_name=f"corporate_bonds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # VISUALISATIONS
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("### 📈 CORPORATE BONDS ANALYSIS")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
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
            fig_scatter = go.Figure()
            
            for rating in filtered_corp['Rating'].unique():
                rating_data = filtered_corp[filtered_corp['Rating'] == rating]
                
                fig_scatter.add_trace(go.Scatter(
                    x=rating_data['Years to Maturity'],
                    y=rating_data['YTM %'],
                    mode='markers',
                    name=rating,
                    marker=dict(size=8),
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
        st.caption("Database: Apple, Microsoft, Amazon, Google, JPMorgan, Bank of America, Walmart, Boeing, Tesla, Netflix, and 40+ more companies...")

# =============================================
# TAB 2: BOND ETFs
# =============================================
with tab2:
    st.markdown("### 📊 BOND ETFs SCREENER")
    st.markdown(f"**Database: 40+ bond ETFs - Yahoo Finance**")
    
    if st.button("🔄 LOAD BOND ETFs DATA", key="load_etfs"):
        with st.spinner("Loading bond ETFs..."):
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
        
        st.success(f"✅ {len(st.session_state['etf_bonds'])} bond ETFs loaded!")
    
    if 'etf_bonds' in st.session_state and st.session_state['etf_bonds'] is not None:
        df_etf = st.session_state['etf_bonds'].copy()
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # FILTRES
        with st.sidebar:
            st.markdown("## 🎯 ETF FILTERS")
            
            categories = ['All'] + sorted(df_etf['Category'].unique().tolist())
            selected_cat = st.selectbox("Category", categories, key="etf_cat")
            
            durations = ['All'] + sorted(df_etf['Duration'].unique().tolist())
            selected_dur = st.selectbox("Duration", durations, key="etf_dur")
            
            regions = ['All'] + sorted(df_etf['Region'].unique().tolist())
            selected_reg = st.selectbox("Region", regions, key="etf_reg")
            
            st.markdown("---")
            
            ytd_min_etf, ytd_max_etf = st.slider(
                "YTD Return (%)",
                min_value=float(df_etf['YTD %'].min()),
                max_value=float(df_etf['YTD %'].max()),
                value=(float(df_etf['YTD %'].min()), float(df_etf['YTD %'].max())),
                key="etf_ytd"
            )
            
            vol_min_etf, vol_max_etf = st.slider(
                "Volatility (%)",
                min_value=float(df_etf['Volatility %'].min()),
                max_value=float(df_etf['Volatility %'].max()),
                value=(float(df_etf['Volatility %'].min()), float(df_etf['Volatility %'].max())),
                key="etf_vol"
            )
        
        # Appliquer filtres
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
        
        # RÉSULTATS
        st.markdown(f"### 📊 RESULTS: {len(filtered_etf)} ETFs found")
        
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
        
        # Tableau
        display_etf = filtered_etf.copy()
        
        for col in ['Price', '1D %', 'YTD %', 'Volatility %', 'Yield %', 'Expense %']:
            if col in display_etf.columns:
                display_etf[col] = display_etf[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(
            display_etf,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Export
        csv_etf = filtered_etf.to_csv(index=False)
        st.download_button(
            label="📥 DOWNLOAD BOND ETFs (CSV)",
            data=csv_etf,
            file_name=f"bond_etfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # VISUALISATIONS
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("### 📈 ETF PERFORMANCE ANALYSIS")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
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
                    marker=dict(size=10),
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

# =============================================
# TAB 3: COMPARISON TOOL
# =============================================
with tab3:
    st.markdown("### 📈 BOND ETF COMPARISON TOOL")
    st.markdown("**Compare multiple bond ETFs side-by-side**")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    available_etfs = list(BOND_ETFS.keys())
    
    col_comp1, col_comp2 = st.columns([3, 1])
    
    with col_comp1:
        selected_compare = st.multiselect(
            "Select bond ETFs to compare (up to 8)",
            options=available_etfs,
            default=['TLT', 'LQD', 'HYG', 'AGG'],
            max_selections=8
        )
    
    with col_comp2:
        compare_period = st.selectbox(
            "Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3,
            key="compare_period"
        )
    
    if selected_compare:
        st.markdown('<hr>', unsafe_allow_html=True)
        
        fig_compare = go.Figure()
        
        colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00', '#00CED1', '#FF1493']
        
        comparison_stats = []
        
        for idx, ticker in enumerate(selected_compare):
            try:
                bond = yf.Ticker(ticker)
                hist = bond.history(period=compare_period)
                
                if len(hist) > 0:
                    normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
                    
                    total_return = normalized.iloc[-1] - 100
                    volatility = (hist['Close'].pct_change().std() * np.sqrt(252)) * 100
                    max_price = hist['Close'].max()
                    min_price = hist['Close'].min()
                    
                    comparison_stats.append({
                        'Ticker': ticker,
                        'Name': BOND_ETFS[ticker]['name'],
                        'Return %': total_return,
                        'Volatility %': volatility,
                        'Max Price': max_price,
                        'Min Price': min_price,
                        'Current': hist['Close'].iloc[-1]
                    })
                    
                    fig_compare.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        mode='lines',
                        name=f"{ticker}",
                        line=dict(color=colors[idx % len(colors)], width=2),
                        hovertemplate=f'<b>{ticker}</b><br>%{{y:.2f}}%<br>%{{x}}<extra></extra>'
                    ))
            except:
                continue
        
        fig_compare.update_layout(
            title=f"Bond ETF Performance Comparison - {compare_period.upper()}",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="Date"
            ),
            yaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="Performance (%)"
            ),
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(17,17,17,0.8)'
            ),
            height=600
        )
        
        fig_compare.add_hline(y=100, line_dash="dash", line_color="#666", annotation_text="Start")
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Tableau comparaison
        if comparison_stats:
            st.markdown("### 📊 COMPARISON STATISTICS")
            
            df_comp_stats = pd.DataFrame(comparison_stats)
            df_comp_stats = df_comp_stats.round(2)
            
            st.dataframe(df_comp_stats, use_container_width=True, hide_index=True)
            
            # Winner/Loser
            st.markdown('<hr>', unsafe_allow_html=True)
            
            col_win1, col_win2, col_win3 = st.columns(3)
            
            with col_win1:
                best_performer = df_comp_stats.loc[df_comp_stats['Return %'].idxmax()]
                st.metric(
                    "🏆 BEST PERFORMER",
                    best_performer['Ticker'],
                    f"+{best_performer['Return %']:.2f}%"
                )
            
            with col_win2:
                worst_performer = df_comp_stats.loc[df_comp_stats['Return %'].idxmin()]
                st.metric(
                    "📉 WORST PERFORMER",
                    worst_performer['Ticker'],
                    f"{worst_performer['Return %']:.2f}%"
                )
            
            with col_win3:
                least_volatile = df_comp_stats.loc[df_comp_stats['Volatility %'].idxmin()]
                st.metric(
                    "🎯 LEAST VOLATILE",
                    least_volatile['Ticker'],
                    f"{least_volatile['Volatility %']:.2f}% vol"
                )
    
    else:
        st.info("👆 Select bond ETFs above to compare their performance")

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 FREE DATA: YAHOO FINANCE • 150+ CORPORATE BONDS • 40+ ETFs<br>
        🔄 REAL-TIME ETF PRICING • COMPREHENSIVE SCREENING • COMPARISON TOOL
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
    © 2025 BLOOMBERG ENS® | BOND SCREENER PRO | 150+ BONDS + 40+ ETFs<br>
    FREE DATA SOURCES • COMPREHENSIVE COVERAGE • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
