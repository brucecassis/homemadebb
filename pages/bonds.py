import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup
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
        font-size: 10px !important;
        color: #FFAA00 !important;
        background-color: #111 !important;
    }
    
    .dataframe th {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        border: 1px solid #555 !important;
        padding: 8px !important;
    }
    
    .dataframe td {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        padding: 6px !important;
    }
    
    /* Highlight rows on hover */
    .dataframe tbody tr:hover {
        background-color: #222 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# DONNÉES OBLIGATAIRES - BASE COMPLÈTE
# =============================================

# Base de données des ETFs obligataires avec leurs caractéristiques
BOND_UNIVERSE = {
    # US TREASURIES
    'SHY': {
        'name': 'iShares 1-3 Year Treasury Bond',
        'type': 'Government',
        'region': 'US',
        'duration': 'Short',
        'maturity': '1-3Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'Treasury'
    },
    'IEI': {
        'name': 'iShares 3-7 Year Treasury Bond',
        'type': 'Government',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '3-7Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'Treasury'
    },
    'IEF': {
        'name': 'iShares 7-10 Year Treasury Bond',
        'type': 'Government',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '7-10Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'Treasury'
    },
    'TLH': {
        'name': 'iShares 10-20 Year Treasury Bond',
        'type': 'Government',
        'region': 'US',
        'duration': 'Long',
        'maturity': '10-20Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'Treasury'
    },
    'TLT': {
        'name': 'iShares 20+ Year Treasury Bond',
        'type': 'Government',
        'region': 'US',
        'duration': 'Very Long',
        'maturity': '20+Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'Treasury'
    },
    'VGIT': {
        'name': 'Vanguard Intermediate-Term Treasury',
        'type': 'Government',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '3-10Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'Treasury'
    },
    'VGLT': {
        'name': 'Vanguard Long-Term Treasury',
        'type': 'Government',
        'region': 'US',
        'duration': 'Long',
        'maturity': '10+Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'Treasury'
    },
    
    # US INVESTMENT GRADE CORPORATE
    'LQD': {
        'name': 'iShares iBoxx $ Investment Grade Corp',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BBB/A',
        'currency': 'USD',
        'category': 'Investment Grade'
    },
    'VCIT': {
        'name': 'Vanguard Intermediate-Term Corporate',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BBB/A',
        'currency': 'USD',
        'category': 'Investment Grade'
    },
    'VCLT': {
        'name': 'Vanguard Long-Term Corporate',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Long',
        'maturity': '10+Y',
        'credit': 'BBB/A',
        'currency': 'USD',
        'category': 'Investment Grade'
    },
    'VCSH': {
        'name': 'Vanguard Short-Term Corporate',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Short',
        'maturity': '1-5Y',
        'credit': 'BBB/A',
        'currency': 'USD',
        'category': 'Investment Grade'
    },
    'IGSB': {
        'name': 'iShares Short-Term Corporate',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Short',
        'maturity': '1-5Y',
        'credit': 'BBB/A',
        'currency': 'USD',
        'category': 'Investment Grade'
    },
    'IGIB': {
        'name': 'iShares Intermediate-Term Corporate',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BBB/A',
        'currency': 'USD',
        'category': 'Investment Grade'
    },
    
    # US HIGH YIELD
    'HYG': {
        'name': 'iShares iBoxx $ High Yield Corporate',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BB/B',
        'currency': 'USD',
        'category': 'High Yield'
    },
    'JNK': {
        'name': 'SPDR Bloomberg High Yield Bond',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BB/B',
        'currency': 'USD',
        'category': 'High Yield'
    },
    'SHYG': {
        'name': 'iShares 0-5 Year High Yield Corporate',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Short',
        'maturity': '0-5Y',
        'credit': 'BB/B',
        'currency': 'USD',
        'category': 'High Yield'
    },
    'FALN': {
        'name': 'iShares Fallen Angels USD Bond',
        'type': 'Corporate',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BB',
        'currency': 'USD',
        'category': 'High Yield'
    },
    
    # EMERGING MARKETS
    'EMB': {
        'name': 'iShares J.P. Morgan USD EM Bond',
        'type': 'Government',
        'region': 'Emerging',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BBB/BB',
        'currency': 'USD',
        'category': 'Emerging Markets'
    },
    'EMHY': {
        'name': 'iShares Emerging Markets High Yield',
        'type': 'Corporate',
        'region': 'Emerging',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BB/B',
        'currency': 'USD',
        'category': 'Emerging Markets'
    },
    'EMLC': {
        'name': 'VanEck J.P. Morgan EM Local Currency',
        'type': 'Government',
        'region': 'Emerging',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'BBB/BB',
        'currency': 'Local',
        'category': 'Emerging Markets'
    },
    
    # TIPS (INFLATION-PROTECTED)
    'TIP': {
        'name': 'iShares TIPS Bond',
        'type': 'Government',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'TIPS'
    },
    'VTIP': {
        'name': 'Vanguard Short-Term TIPS',
        'type': 'Government',
        'region': 'US',
        'duration': 'Short',
        'maturity': '0-5Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'TIPS'
    },
    'LTPZ': {
        'name': 'PIMCO 15+ Year US TIPS',
        'type': 'Government',
        'region': 'US',
        'duration': 'Very Long',
        'maturity': '15+Y',
        'credit': 'AAA',
        'currency': 'USD',
        'category': 'TIPS'
    },
    
    # MUNICIPAL BONDS
    'MUB': {
        'name': 'iShares National Muni Bond',
        'type': 'Municipal',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'AA/A',
        'currency': 'USD',
        'category': 'Municipal'
    },
    'VTEB': {
        'name': 'Vanguard Tax-Exempt Bond',
        'type': 'Municipal',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'AA/A',
        'currency': 'USD',
        'category': 'Municipal'
    },
    'SUB': {
        'name': 'iShares Short-Term National Muni Bond',
        'type': 'Municipal',
        'region': 'US',
        'duration': 'Short',
        'maturity': '1-5Y',
        'credit': 'AA/A',
        'currency': 'USD',
        'category': 'Municipal'
    },
    
    # INTERNATIONAL DEVELOPED
    'BNDX': {
        'name': 'Vanguard Total International Bond',
        'type': 'Government',
        'region': 'International',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'AA/A',
        'currency': 'Hedged USD',
        'category': 'International'
    },
    'IAGG': {
        'name': 'iShares Core International Aggregate',
        'type': 'Mixed',
        'region': 'International',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'AA/A',
        'currency': 'Hedged USD',
        'category': 'International'
    },
    
    # AGGREGATE
    'AGG': {
        'name': 'iShares Core US Aggregate Bond',
        'type': 'Mixed',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'AA/A',
        'currency': 'USD',
        'category': 'Aggregate'
    },
    'BND': {
        'name': 'Vanguard Total Bond Market',
        'type': 'Mixed',
        'region': 'US',
        'duration': 'Medium',
        'maturity': '5-10Y',
        'credit': 'AA/A',
        'currency': 'USD',
        'category': 'Aggregate'
    },
}

# =============================================
# FONCTIONS DE RÉCUPÉRATION DE DONNÉES
# =============================================

@st.cache_data(ttl=300)
def get_bond_metrics(ticker):
    """Récupère les métriques détaillées d'un ETF obligataire"""
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        hist = etf.history(period='1y')
        
        if len(hist) < 2:
            return None
        
        # Prix et variations
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_1d = ((current_price - prev_close) / prev_close) * 100
        
        # Performance sur différentes périodes
        if len(hist) >= 5:
            change_5d = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100
        else:
            change_5d = None
            
        if len(hist) >= 20:
            change_1m = ((current_price - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]) * 100
        else:
            change_1m = None
            
        if len(hist) >= 60:
            change_3m = ((current_price - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60]) * 100
        else:
            change_3m = None
        
        change_ytd = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        
        # Volume
        avg_volume = hist['Volume'].tail(20).mean()
        
        # Volatilité (écart-type annualisé)
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Yield (si disponible)
        dividend_yield = info.get('yield', None)
        if dividend_yield:
            dividend_yield = dividend_yield * 100  # Convertir en %
        
        # Expense ratio
        expense_ratio = info.get('expenseRatio', None)
        if expense_ratio:
            expense_ratio = expense_ratio * 100  # Convertir en %
        
        # Assets Under Management
        aum = info.get('totalAssets', None)
        
        # Duration (si disponible dans le nom ou description)
        # Note: Yahoo Finance ne fournit pas toujours la duration directement
        
        return {
            'ticker': ticker,
            'price': current_price,
            'change_1d': change_1d,
            'change_5d': change_5d,
            'change_1m': change_1m,
            'change_3m': change_3m,
            'change_ytd': change_ytd,
            'volume': avg_volume,
            'volatility': volatility,
            'yield': dividend_yield,
            'expense_ratio': expense_ratio,
            'aum': aum,
        }
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def build_bond_screener_data():
    """Construit la base de données complète pour le screener"""
    data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(BOND_UNIVERSE)
    
    for idx, (ticker, info) in enumerate(BOND_UNIVERSE.items()):
        status_text.text(f"Loading {ticker}... ({idx+1}/{total})")
        progress_bar.progress((idx + 1) / total)
        
        metrics = get_bond_metrics(ticker)
        
        if metrics:
            row = {
                'Ticker': ticker,
                'Name': info['name'],
                'Type': info['type'],
                'Region': info['region'],
                'Duration': info['duration'],
                'Maturity': info['maturity'],
                'Credit': info['credit'],
                'Category': info['category'],
                'Price': metrics['price'],
                '1D %': metrics['change_1d'],
                '5D %': metrics['change_5d'],
                '1M %': metrics['change_1m'],
                '3M %': metrics['change_3m'],
                'YTD %': metrics['change_ytd'],
                'Volatility %': metrics['volatility'],
                'Yield %': metrics['yield'],
                'Expense %': metrics['expense_ratio'],
                'AUM $M': metrics['aum'] / 1e6 if metrics['aum'] else None,
            }
            data.append(row)
        
        time.sleep(0.1)  # Éviter de surcharger l'API
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(data)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - BOND SCREENER</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">MARKETS</a>
    </div>
    <div>{current_time} UTC • FIXED INCOME SCREENING</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# INSTRUCTIONS
# =============================================
st.markdown("""
<div style='background:#111;border:1px solid #333;padding:10px;margin:10px 0;border-left:4px solid #FFAA00;'>
<b style='color:#FFAA00;'>🔍 BOND SCREENER - INSTRUCTIONS:</b><br>
• Cliquez sur "🔄 LOAD BOND DATA" pour charger les données (peut prendre 30-60 secondes)<br>
• Utilisez les filtres dans la barre latérale pour affiner votre recherche<br>
• Cliquez sur les en-têtes de colonnes pour trier les résultats<br>
• Les données incluent 40+ ETFs obligataires couvrant toutes les catégories<br>
</div>
""", unsafe_allow_html=True)

# =============================================
# BOUTON DE CHARGEMENT
# =============================================
col_load1, col_load2, col_load3 = st.columns([1, 1, 4])

with col_load1:
    if st.button("🔄 LOAD BOND DATA", key="load_data"):
        st.session_state['bond_data'] = None
        st.session_state['load_requested'] = True

with col_load2:
    if st.button("🗑️ CLEAR DATA", key="clear_data"):
        st.session_state['bond_data'] = None
        st.session_state['load_requested'] = False
        st.rerun()

# Initialiser les données si demandé
if 'load_requested' not in st.session_state:
    st.session_state['load_requested'] = False

if 'bond_data' not in st.session_state:
    st.session_state['bond_data'] = None

if st.session_state['load_requested'] and st.session_state['bond_data'] is None:
    with st.spinner("📊 Loading bond data from Yahoo Finance..."):
        st.session_state['bond_data'] = build_bond_screener_data()
    st.success("✅ Data loaded successfully!")
    st.session_state['load_requested'] = False

# =============================================
# AFFICHAGE DES DONNÉES
# =============================================
if st.session_state['bond_data'] is not None:
    df = st.session_state['bond_data'].copy()
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### 🔍 SCREENING FILTERS")
    
    # ===== SIDEBAR FILTERS =====
    with st.sidebar:
        st.markdown("## 🎯 FILTERS")
        
        # Type de bond
        bond_types = ['All'] + sorted(df['Type'].unique().tolist())
        selected_type = st.selectbox("Bond Type", bond_types)
        
        # Région
        regions = ['All'] + sorted(df['Region'].unique().tolist())
        selected_region = st.selectbox("Region", regions)
        
        # Catégorie
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_category = st.selectbox("Category", categories)
        
        # Duration
        durations = ['All'] + sorted(df['Duration'].unique().tolist())
        selected_duration = st.selectbox("Duration", durations)
        
        # Credit Rating
        credits = ['All'] + sorted(df['Credit'].unique().tolist())
        selected_credit = st.selectbox("Credit Rating", credits)
        
        st.markdown("---")
        st.markdown("### 📊 PERFORMANCE FILTERS")
        
        # YTD Performance
        ytd_min, ytd_max = st.slider(
            "YTD Return (%)",
            min_value=float(df['YTD %'].min()),
            max_value=float(df['YTD %'].max()),
            value=(float(df['YTD %'].min()), float(df['YTD %'].max()))
        )
        
        # Volatility
        vol_min, vol_max = st.slider(
            "Volatility (%)",
            min_value=float(df['Volatility %'].min()),
            max_value=float(df['Volatility %'].max()),
            value=(float(df['Volatility %'].min()), float(df['Volatility %'].max()))
        )
        
        # Yield (si disponible)
        if df['Yield %'].notna().any():
            yield_min, yield_max = st.slider(
                "Yield (%)",
                min_value=float(df['Yield %'].min()),
                max_value=float(df['Yield %'].max()),
                value=(float(df['Yield %'].min()), float(df['Yield %'].max()))
            )
        else:
            yield_min, yield_max = None, None
        
        # AUM
        if df['AUM $M'].notna().any():
            aum_min = st.number_input(
                "Min AUM ($M)",
                min_value=0.0,
                value=0.0,
                step=100.0
            )
        else:
            aum_min = 0
    
    # ===== APPLIQUER LES FILTRES =====
    filtered_df = df.copy()
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == selected_type]
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    if selected_duration != 'All':
        filtered_df = filtered_df[filtered_df['Duration'] == selected_duration]
    
    if selected_credit != 'All':
        filtered_df = filtered_df[filtered_df['Credit'] == selected_credit]
    
    filtered_df = filtered_df[
        (filtered_df['YTD %'] >= ytd_min) & 
        (filtered_df['YTD %'] <= ytd_max)
    ]
    
    filtered_df = filtered_df[
        (filtered_df['Volatility %'] >= vol_min) & 
        (filtered_df['Volatility %'] <= vol_max)
    ]
    
    if yield_min is not None and yield_max is not None:
        filtered_df = filtered_df[
            (filtered_df['Yield %'] >= yield_min) & 
            (filtered_df['Yield %'] <= yield_max)
        ]
    
    if aum_min > 0:
        filtered_df = filtered_df[filtered_df['AUM $M'] >= aum_min]
    
    # ===== AFFICHAGE DES RÉSULTATS =====
    st.markdown(f"### 📊 RESULTS: {len(filtered_df)} bonds found")
    
    # Statistiques rapides
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        avg_ytd = filtered_df['YTD %'].mean()
        st.metric("Avg YTD Return", f"{avg_ytd:.2f}%")
    
    with col_stat2:
        avg_vol = filtered_df['Volatility %'].mean()
        st.metric("Avg Volatility", f"{avg_vol:.2f}%")
    
    with col_stat3:
        if filtered_df['Yield %'].notna().any():
            avg_yield = filtered_df['Yield %'].mean()
            st.metric("Avg Yield", f"{avg_yield:.2f}%")
        else:
            st.metric("Avg Yield", "N/A")
    
    with col_stat4:
        total_aum = filtered_df['AUM $M'].sum()
        st.metric("Total AUM", f"${total_aum:,.0f}M")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Formater le DataFrame pour l'affichage
    display_df = filtered_df.copy()
    
    # Arrondir les nombres
    numeric_cols = ['Price', '1D %', '5D %', '1M %', '3M %', 'YTD %', 'Volatility %', 'Yield %', 'Expense %', 'AUM $M']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Afficher le tableau avec style
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    # ===== EXPORT CSV =====
    st.markdown('<hr>', unsafe_allow_html=True)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 DOWNLOAD RESULTS (CSV)",
        data=csv,
        file_name=f"bond_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    
    # ===== VISUALISATIONS =====
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### 📈 VISUAL ANALYSIS")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Scatter plot: Risk vs Return
        fig_scatter = go.Figure()
        
        for category in filtered_df['Category'].unique():
            cat_data = filtered_df[filtered_df['Category'] == category]
            
            fig_scatter.add_trace(go.Scatter(
                x=cat_data['Volatility %'],
                y=cat_data['YTD %'],
                mode='markers+text',
                name=category,
                text=cat_data['Ticker'],
                textposition='top center',
                marker=dict(size=10),
                hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>YTD: %{y:.2f}%<extra></extra>'
            ))
        
        fig_scatter.update_layout(
            title="Risk vs Return Analysis",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="Volatility (%)"
            ),
            yaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="YTD Return (%)"
            ),
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_viz2:
        # Bar chart: Performance by Category
        cat_perf = filtered_df.groupby('Category')['YTD %'].mean().sort_values()
        
        fig_bar = go.Figure()
        
        colors = ['#FF0000' if x < 0 else '#00FF00' for x in cat_perf.values]
        
        fig_bar.add_trace(go.Bar(
            x=cat_perf.values,
            y=cat_perf.index,
            orientation='h',
            marker_color=colors,
            text=cat_perf.values,
            texttemplate='%{text:.2f}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Avg YTD: %{x:.2f}%<extra></extra>'
        ))
        
        fig_bar.update_layout(
            title="Average YTD Return by Category",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="YTD Return (%)"
            ),
            yaxis=dict(
                gridcolor='#333',
                showgrid=False
            ),
            height=500
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("👆 Click 'LOAD BOND DATA' to start screening bonds")
    
    # Afficher un aperçu de la base de données
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### 📋 BOND UNIVERSE PREVIEW")
    
    preview_data = []
    for ticker, info in list(BOND_UNIVERSE.items())[:10]:
        preview_data.append({
            'Ticker': ticker,
            'Name': info['name'],
            'Type': info['type'],
            'Category': info['category'],
            'Duration': info['duration'],
            'Credit': info['credit']
        })
    
    preview_df = pd.DataFrame(preview_data)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    
    st.caption(f"Showing 10 of {len(BOND_UNIVERSE)} bonds in database. Click 'LOAD BOND DATA' to see full data.")

# =============================================
# BOND COMPARISON TOOL
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("### 📊 BOND COMPARISON TOOL")

available_tickers = list(BOND_UNIVERSE.keys())

col_comp1, col_comp2 = st.columns([3, 1])

with col_comp1:
    selected_compare = st.multiselect(
        "Select bonds to compare (up to 5)",
        options=available_tickers,
        default=available_tickers[:3],
        max_selections=5
    )

with col_comp2:
    compare_period = st.selectbox(
        "Period",
        options=['1mo', '3mo', '6mo', '1y', '2y'],
        index=2
    )

if selected_compare:
    fig_compare = go.Figure()
    
    colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000']
    
    for idx, ticker in enumerate(selected_compare):
        try:
            bond = yf.Ticker(ticker)
            hist = bond.history(period=compare_period)
            
            if len(hist) > 0:
                normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
                
                fig_compare.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=f"{ticker} - {BOND_UNIVERSE[ticker]['name'][:30]}",
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
        height=500
    )
    
    fig_compare.add_hline(y=100, line_dash="dash", line_color="#666")
    
    st.plotly_chart(fig_compare, use_container_width=True)

# =============================================
# INFO & HELP
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

with st.expander("📖 BOND SCREENER GUIDE"):
    st.markdown("""
    ## 🔍 HOW TO USE THE BOND SCREENER
    
    ### 1. Loading Data
    - Click "LOAD BOND DATA" to fetch real-time data from Yahoo Finance
    - Data includes 40+ bond ETFs across all major categories
    - Loading takes 30-60 seconds (API rate limits)
    
    ### 2. Filtering
    Use the sidebar filters to narrow down your search:
    - **Type**: Government, Corporate, Municipal, Mixed
    - **Region**: US, International, Emerging Markets
    - **Category**: Treasury, Investment Grade, High Yield, TIPS, etc.
    - **Duration**: Short, Medium, Long, Very Long
    - **Credit Rating**: AAA to B ratings
    - **Performance**: Filter by YTD returns
    - **Risk**: Filter by volatility
    - **Yield**: Filter by current yield
    - **AUM**: Minimum assets under management
    
    ### 3. Analyzing Results
    - **Table View**: Sort by any column to find top performers
    - **Risk/Return Chart**: Visual analysis of volatility vs returns
    - **Category Performance**: Compare average returns by category
    - **Comparison Tool**: Chart up to 5 bonds side-by-side
    
    ### 4. Exporting Data
    - Download filtered results as CSV for further analysis
    - Import into Excel or your favorite spreadsheet tool
    
    ## 📊 KEY METRICS EXPLAINED
    
    **YTD Return**: Year-to-date performance in percentage
    
    **Volatility**: Annualized standard deviation of daily returns. Higher = more risk
    
    **Yield**: Current yield, typically reflects interest payments
    
    **Duration**: 
    - Short: < 5 years (less interest rate risk)
    - Medium: 5-10 years (moderate risk)
    - Long: 10+ years (higher interest rate risk)
    
    **Credit Rating**: 
    - AAA: Highest quality, lowest default risk
    - AA/A: High quality
    - BBB: Investment grade
    - BB/B: High yield ("junk"), higher default risk
    
    ## 💡 TIPS FOR BOND INVESTING
    
    1. **Diversify** across durations and credit qualities
    2. **Consider duration** relative to interest rate expectations
    3. **Match duration** to your investment timeline
    4. **Higher yield = higher risk** (usually)
    5. **TIPS** protect against inflation
    6. **Municipal bonds** may offer tax advantages
    7. **Check expense ratios** - lower is better for ETFs
    
    ## 🔗 DATA SOURCES
    
    - Yahoo Finance (yfinance API)
    - Real-time ETF prices and metrics
    - Updated throughout trading day
    - No registration or API key required
    """)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 FREE DATA SOURCE: YAHOO FINANCE API<br>
        🔄 40+ BOND ETFs • ALL MAJOR CATEGORIES • REAL-TIME PRICING
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 SESSION: {last_update}<br>
        📍 BOND SCREENER • POWERED BY YFINANCE
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | BOND SCREENER | FIXED INCOME ANALYSIS<br>
    FREE DATA • 40+ ETFS • COMPREHENSIVE SCREENING • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
