import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Economics Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Votre clé API FRED
FRED_API_KEY = "ce5dbb3d3fcd8669f2fe2cdd9c79a7da"

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
    
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
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
        font-size: 12px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
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
    
    .stTextInput input, .stSelectbox select {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
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
    
    .warning-box {
        background-color: #1a0a00;
        border-left: 3px solid #FF6600;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    .indicator-box {
        background-color: #0a0a0a;
        border: 1px solid #333;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - ECONOMIC DATA</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# Fonctions FRED API
@st.cache_data(ttl=3600)
def get_fred_series(series_id, observation_start=None):
    """Récupère une série de données FRED"""
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json'
        }
        
        if observation_start:
            params['observation_start'] = observation_start
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            
            if observations:
                df = pd.DataFrame(observations)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
                df = df.sort_values('date')
                
                return df
        
        return None
        
    except Exception as e:
        st.error(f"Erreur FRED API pour {series_id}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_fred_series_info(series_id):
    """Récupère les informations sur une série FRED"""
    try:
        url = "https://api.stlouisfed.org/fred/series"
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            series = data.get('seriess', [])
            if series:
                return series[0]
        
        return None
        
    except Exception as e:
        return None

def calculate_yoy_change(df):
    """Calcule la variation Year-over-Year"""
    if df is None or len(df) < 12:
        return None
    
    df = df.copy()
    df['yoy_change'] = df['value'].pct_change(12) * 100
    return df

# Dictionnaire des séries économiques importantes
ECONOMIC_SERIES = {
    'Interest Rates': {
        'FEDFUNDS': 'Fed Funds Rate',
        'DGS10': '10-Year Treasury',
        'DGS2': '2-Year Treasury',
        'DGS30': '30-Year Treasury',
        'MORTGAGE30US': '30Y Mortgage Rate'
    },
    'Inflation': {
        'CPIAUCSL': 'CPI (All Items)',
        'CPILFESL': 'Core CPI',
        'PCEPI': 'PCE Price Index',
        'PCEPILFE': 'Core PCE'
    },
    'Employment': {
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'JTSJOL': 'JOLTS Job Openings',
        'ICSA': 'Initial Claims'
    },
    'Growth': {
        'GDP': 'GDP',
        'GDPC1': 'Real GDP',
        'INDPRO': 'Industrial Production',
        'RSXFS': 'Retail Sales'
    },
    'Markets': {
        'SP500': 'S&P 500',
        'VIXCLS': 'VIX',
        'DEXUSEU': 'EUR/USD',
        'T10Y2Y': '10Y-2Y Spread'
    },
    'Monetary': {
        'M2SL': 'M2 Money Supply',
        'WALCL': 'Fed Balance Sheet'
    }
}

# ONGLETS PRINCIPAUX
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 DASHBOARD", "📈 CUSTOM ANALYSIS", "📐 CONSTRUCTED INDICATORS", "🔬 ECONOMETRIC TESTS", "🔍 SERIES SEARCH", "📥 DOWNLOAD DATA", "📝 NOTES"])# TAB 1: DASHBOARD
with tab1:
    st.markdown("### 📊 ECONOMIC INDICATORS DASHBOARD")
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_dash"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # TAUX D'INTÉRÊT
    st.markdown("#### 💰 INTEREST RATES")
    cols_rates = st.columns(4)
    
    rates_series = [
        ('FEDFUNDS', 'FED FUNDS'),
        ('DGS10', '10Y TREASURY'),
        ('DGS2', '2Y TREASURY'),
        ('MORTGAGE30US', '30Y MORTGAGE')
    ]
    
    for idx, (series_id, label) in enumerate(rates_series):
        with cols_rates[idx]:
            df = get_fred_series(series_id)
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                change = current_value - previous_value
                
                st.metric(
                    label=label,
                    value=f"{current_value:.2f}%",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=label, value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # INFLATION
    st.markdown("#### 📊 INFLATION")
    cols_inflation = st.columns(4)
    
    inflation_series = [
        ('CPIAUCSL', 'CPI'),
        ('CPILFESL', 'CORE CPI'),
        ('PCEPI', 'PCE'),
        ('PCEPILFE', 'CORE PCE')
    ]
    
    for idx, (series_id, label) in enumerate(inflation_series):
        with cols_inflation[idx]:
            df = get_fred_series(series_id)
            if df is not None and len(df) > 12:
                df_yoy = calculate_yoy_change(df)
                current_yoy = df_yoy['yoy_change'].iloc[-1]
                previous_yoy = df_yoy['yoy_change'].iloc[-2]
                change = current_yoy - previous_yoy
                
                st.metric(
                    label=f"{label} YoY",
                    value=f"{current_yoy:.2f}%",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=label, value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # YIELD CURVE
    st.markdown("#### 📉 YIELD CURVE")
    
    treasury_rates = {
        '1M': 'DGS1MO',
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    }
    
    curve_data = []
    maturities = []
    
    for maturity, series_id in treasury_rates.items():
        df = get_fred_series(series_id)
        if df is not None and len(df) > 0:
            curve_data.append(df['value'].iloc[-1])
            maturities.append(maturity)
    
    if curve_data:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=maturities,
            y=curve_data,
            mode='lines+markers',
            line=dict(color='#FFAA00', width=2),
            marker=dict(size=8, color='#00FF00')
        ))
        
        fig_curve.update_layout(
            title="US Treasury Yield Curve",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(title="Maturity", gridcolor='#333'),
            yaxis=dict(title="Yield (%)", gridcolor='#333'),
            height=400
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)

# TAB 2: ANALYSE PERSONNALISÉE
with tab2:
    st.markdown("### 📈 CUSTOM ECONOMIC ANALYSIS")
    
    col_analysis1, col_analysis2 = st.columns([2, 1])
    
    with col_analysis1:
        categories = list(ECONOMIC_SERIES.keys())
        selected_category = st.selectbox(
            "SELECT CATEGORY",
            options=categories,
            key="category_select"
        )
        
        series_in_category = ECONOMIC_SERIES[selected_category]
        
        selected_series = st.multiselect(
            "SELECT SERIES TO COMPARE",
            options=list(series_in_category.keys()),
            default=list(series_in_category.keys())[:2],
            format_func=lambda x: f"{x} - {series_in_category[x]}",
            key="series_multiselect"
        )
    
    with col_analysis2:
        lookback_period = st.selectbox(
            "LOOKBACK PERIOD",
            options=['1Y', '2Y', '5Y', '10Y', 'MAX'],
            index=2,
            key="lookback_select"
        )
    
    if st.button("📊 GENERATE ANALYSIS", use_container_width=True, key="generate_analysis"):
        if selected_series:
            if lookback_period == 'MAX':
                start_date = None
            else:
                years = int(lookback_period[:-1])
                start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
            
            series_data = {}
            for series_id in selected_series:
                df = get_fred_series(series_id, observation_start=start_date)
                if df is not None:
                    series_data[series_id] = df
            
            if series_data:
                fig = go.Figure()
                
                for series_id, df in series_data.items():
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['value'],
                        mode='lines',
                        name=f"{series_id}",
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"{selected_category} Comparison",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title="Value"),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# TAB 3: INDICATEURS CONSTRUITS
with tab3:
    st.markdown("### 📐 CONSTRUCTED MACRO INDICATORS")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #999;">
        Ces indicateurs sont calculés à partir de séries FRED brutes en utilisant des formules économétriques standard.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_constructed"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 1. INFLATION YoY =====
    st.markdown("#### 📊 1. INFLATION YoY")
    st.caption("Formula: `100 * (CPI_t / CPI_{t-12} - 1)` | Series: `CPIAUCSL`")
    
    df_cpi = get_fred_series('CPIAUCSL')
    
    if df_cpi is not None and len(df_cpi) > 12:
        df_cpi_yoy = calculate_yoy_change(df_cpi)
        
        col_inf1, col_inf2, col_inf3 = st.columns([2, 1, 1])
        
        with col_inf1:
            fig_inf = go.Figure()
            fig_inf.add_trace(go.Scatter(
                x=df_cpi_yoy['date'],
                y=df_cpi_yoy['yoy_change'],
                mode='lines',
                line=dict(color='#FFAA00', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 170, 0, 0.1)',
                name='CPI YoY'
            ))
            
            # Ligne cible Fed 2%
            fig_inf.add_hline(y=2, line_dash="dash", line_color="#00FF00", 
                             annotation_text="Fed Target 2%", annotation_position="right")
            
            fig_inf.update_layout(
                title="CPI Year-over-Year Change (%)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="YoY Change (%)"),
                height=350
            )
            
            st.plotly_chart(fig_inf, use_container_width=True)
        
        with col_inf2:
            current_inf = df_cpi_yoy['yoy_change'].iloc[-1]
            previous_inf = df_cpi_yoy['yoy_change'].iloc[-2]
            
            st.metric(
                label="CURRENT CPI YoY",
                value=f"{current_inf:.2f}%",
                delta=f"{current_inf - previous_inf:+.2f}%"
            )
            
            st.metric(
                label="VS FED TARGET",
                value=f"{current_inf - 2:.2f}%",
                delta="Above" if current_inf > 2 else "Below"
            )
        
        with col_inf3:
            # Stats
            st.metric("MAX (5Y)", f"{df_cpi_yoy['yoy_change'].tail(60).max():.2f}%")
            st.metric("MIN (5Y)", f"{df_cpi_yoy['yoy_change'].tail(60).min():.2f}%")
            st.metric("MEAN (5Y)", f"{df_cpi_yoy['yoy_change'].tail(60).mean():.2f}%")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 2. REAL YIELD =====
    st.markdown("#### 📊 2. REAL YIELD (10Y)")
    st.caption("Formula: `DGS10 - CPI_YoY` | Series: `DGS10`, `CPIAUCSL`")
    
    df_10y = get_fred_series('DGS10')
    
    if df_10y is not None and df_cpi_yoy is not None:
        # Merger les deux séries
        df_real_yield = pd.merge(
            df_10y[['date', 'value']].rename(columns={'value': 'nominal_10y'}),
            df_cpi_yoy[['date', 'yoy_change']].rename(columns={'yoy_change': 'inflation'}),
            on='date',
            how='inner'
        )
        
        df_real_yield['real_yield'] = df_real_yield['nominal_10y'] - df_real_yield['inflation']
        
        col_ry1, col_ry2, col_ry3 = st.columns([2, 1, 1])
        
        with col_ry1:
            fig_ry = go.Figure()
            
            fig_ry.add_trace(go.Scatter(
                x=df_real_yield['date'],
                y=df_real_yield['real_yield'],
                mode='lines',
                line=dict(color='#00FF00', width=2),
                name='Real Yield'
            ))
            
            fig_ry.add_trace(go.Scatter(
                x=df_real_yield['date'],
                y=df_real_yield['nominal_10y'],
                mode='lines',
                line=dict(color='#FFAA00', width=1, dash='dash'),
                name='Nominal 10Y'
            ))
            
            fig_ry.add_hline(y=0, line_dash="dot", line_color="#FF0000")
            
            fig_ry.update_layout(
                title="Real Yield vs Nominal 10Y Treasury",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="Yield (%)"),
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_ry, use_container_width=True)
        
        with col_ry2:
            current_real = df_real_yield['real_yield'].iloc[-1]
            current_nominal = df_real_yield['nominal_10y'].iloc[-1]
            
            st.metric(
                label="REAL YIELD",
                value=f"{current_real:.2f}%"
            )
            
            st.metric(
                label="NOMINAL 10Y",
                value=f"{current_nominal:.2f}%"
            )
        
        with col_ry3:
            st.metric("MEAN (5Y)", f"{df_real_yield['real_yield'].tail(60).mean():.2f}%")
            
            if current_real < 0:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                    ⚠️ NEGATIVE REAL YIELD
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                    ✅ POSITIVE REAL YIELD
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 3. YIELD CURVE SLOPES =====
    st.markdown("#### 📊 3. YIELD CURVE SLOPES (RECESSION INDICATORS)")
    
    # 10Y-2Y Spread
    st.caption("**A. 10Y-2Y Spread** | Formula: `DGS10 - DGS2` | Classic recession indicator")
    
    df_10y2y = get_fred_series('T10Y2Y')
    
    if df_10y2y is not None:
        col_yc1, col_yc2, col_yc3 = st.columns([2, 1, 1])
        
        with col_yc1:
            fig_10y2y = go.Figure()
            
            # Zones de récession (à ajouter manuellement ou via NBER data)
            fig_10y2y.add_trace(go.Scatter(
                x=df_10y2y['date'],
                y=df_10y2y['value'],
                mode='lines',
                line=dict(color='#FFAA00', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 170, 0, 0.1)',
                name='10Y-2Y Spread'
            ))
            
            fig_10y2y.add_hline(y=0, line_dash="dash", line_color="#FF0000", 
                               annotation_text="Inversion Line", annotation_position="left")
            
            fig_10y2y.update_layout(
                title="10Y-2Y Treasury Spread (Recession Indicator)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="Spread (%)"),
                height=300
            )
            
            st.plotly_chart(fig_10y2y, use_container_width=True)
        
        with col_yc2:
            current_spread = df_10y2y['value'].iloc[-1]
            
            st.metric(
                label="10Y-2Y SPREAD",
                value=f"{current_spread:.2f}%"
            )
            
            # Signal de récession
            if current_spread < 0:
                inversion_duration = (df_10y2y[df_10y2y['value'] < 0].tail(1)['date'].iloc[0] - 
                                     df_10y2y['date'].iloc[-1]).days
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF0000; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF0000; font-weight: bold;">
                    🔴 INVERTED CURVE
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #FF6600;">
                    Recession signal active
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                    ✅ NORMAL CURVE
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_yc3:
            # Historique des inversions
            inversions = df_10y2y[df_10y2y['value'] < 0]
            st.metric("INVERSIONS (HISTORY)", f"{len(inversions)}")
            st.caption("Number of months with inversion")
    
    st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
    
    # 10Y-3M Spread
    st.caption("**B. 10Y-3M Spread** | Formula: `DGS10 - DGS3MO` | Better leading indicator")
    
    df_10y_data = get_fred_series('DGS10')
    df_3m_data = get_fred_series('DGS3MO')
    
    if df_10y_data is not None and df_3m_data is not None:
        df_10y3m = pd.merge(
            df_10y_data[['date', 'value']].rename(columns={'value': 'dgs10'}),
            df_3m_data[['date', 'value']].rename(columns={'value': 'dgs3mo'}),
            on='date',
            how='inner'
        )
        df_10y3m['spread'] = df_10y3m['dgs10'] - df_10y3m['dgs3mo']
        
        col_yc3m1, col_yc3m2 = st.columns([2, 1])
        
        with col_yc3m1:
            fig_10y3m = go.Figure()
            
            fig_10y3m.add_trace(go.Scatter(
                x=df_10y3m['date'],
                y=df_10y3m['spread'],
                mode='lines',
                line=dict(color='#00FFFF', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 255, 0.1)',
                name='10Y-3M Spread'
            ))
            
            fig_10y3m.add_hline(y=0, line_dash="dash", line_color="#FF0000")
            
            fig_10y3m.update_layout(
                title="10Y-3M Treasury Spread (Leading Recession Indicator)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="Spread (%)"),
                height=300
            )
            
            st.plotly_chart(fig_10y3m, use_container_width=True)
        
        with col_yc3m2:
            current_spread_3m = df_10y3m['spread'].iloc[-1]
            
            st.metric(
                label="10Y-3M SPREAD",
                value=f"{current_spread_3m:.2f}%"
            )
            
            if current_spread_3m < 0:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF0000; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF0000; font-weight: bold;">
                    🔴 INVERTED
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #FF6600;">
                    Strong recession signal
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 4. MONEY SUPPLY GROWTH =====
    st.markdown("#### 📊 4. MONEY SUPPLY GROWTH (M2 YoY)")
    st.caption("Formula: `100 * (M2_t / M2_{t-12} - 1)` | Series: `M2SL`")
    
    df_m2 = get_fred_series('M2SL')
    
    if df_m2 is not None and len(df_m2) > 12:
        df_m2_yoy = calculate_yoy_change(df_m2)
        
        col_m2_1, col_m2_2 = st.columns([2, 1])
        
        with col_m2_1:
            fig_m2 = go.Figure()
            
            fig_m2.add_trace(go.Scatter(
                x=df_m2_yoy['date'],
                y=df_m2_yoy['yoy_change'],
                mode='lines',
                line=dict(color='#FF00FF', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 255, 0.1)',
                name='M2 YoY Growth'
            ))
            
            fig_m2.add_hline(y=0, line_dash="dot", line_color="#FF0000")
            
            fig_m2.update_layout(
                title="M2 Money Supply Year-over-Year Growth (%)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="YoY Growth (%)"),
                height=350
            )
            
            st.plotly_chart(fig_m2, use_container_width=True)
        
        with col_m2_2:
            current_m2_growth = df_m2_yoy['yoy_change'].iloc[-1]
            
            st.metric(
                label="M2 GROWTH YoY",
                value=f"{current_m2_growth:.2f}%"
            )
            
            st.metric(
                label="M2 CURRENT",
                value=f"${df_m2['value'].iloc[-1]/1000:.1f}T"
            )
            
            if current_m2_growth < 0:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                    ⚠️ CONTRACTING M2
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                    Money supply shrinking
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 5. REAL GDP GROWTH =====
    st.markdown("#### 📊 5. REAL GDP QUARTER-ON-QUARTER (ANNUALIZED)")
    st.caption("Formula: `400 * (GDPC1_t / GDPC1_{t-1} - 1)` | Series: `GDPC1`")
    
    df_gdp = get_fred_series('GDPC1')
    
    if df_gdp is not None and len(df_gdp) > 1:
        df_gdp['qoq_annualized'] = 400 * df_gdp['value'].pct_change(1)
        
        col_gdp1, col_gdp2 = st.columns([2, 1])
        
        with col_gdp1:
            fig_gdp = go.Figure()
            
            # Barres pour QoQ growth
            colors = ['#00FF00' if x > 0 else '#FF0000' for x in df_gdp['qoq_annualized']]
            
            fig_gdp.add_trace(go.Bar(
                x=df_gdp['date'],
                y=df_gdp['qoq_annualized'],
                marker_color=colors,
                name='GDP QoQ Growth'
            ))
            
            fig_gdp.add_hline(y=0, line_dash="dot", line_color="#FFAA00")
            
            fig_gdp.update_layout(
                title="Real GDP Quarter-on-Quarter Growth (Annualized %)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="QoQ Growth (%)"),
                height=350
            )
            
            st.plotly_chart(fig_gdp, use_container_width=True)
        
        with col_gdp2:
            current_gdp_growth = df_gdp['qoq_annualized'].iloc[-1]
            previous_gdp_growth = df_gdp['qoq_annualized'].iloc[-2]
            
            st.metric(
                label="LATEST GDP GROWTH",
                value=f"{current_gdp_growth:.2f}%",
                delta=f"{current_gdp_growth - previous_gdp_growth:+.2f}%"
            )
            
            # Compter les quarters négatifs récents
            recent_negative = (df_gdp['qoq_annualized'].tail(2) < 0).sum()
            
            if recent_negative >= 2:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF0000; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF0000; font-weight: bold;">
                    🔴 TECHNICAL RECESSION
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #FF6600;">
                    2 consecutive negative quarters
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 6. TAYLOR RULE =====
    st.markdown("#### 📊 6. TAYLOR RULE IMPLIED RATE")
    st.caption("Formula: `r* + inflation + 0.5*(inflation - target) + 0.5*output_gap`")
    st.caption("Simplified: `r* = 2%, target = 2%, output_gap ≈ 0` → `Taylor Rate = 2 + inflation + 0.5*(inflation - 2)`")
    
    if df_cpi_yoy is not None and df_10y_data is not None:
        # Calculer le taux de Taylor
        r_star = 2.0  # Taux neutre estimé
        inflation_target = 2.0
        
        df_taylor = df_cpi_yoy[['date', 'yoy_change']].copy()
        df_taylor['taylor_rate'] = r_star + df_taylor['yoy_change'] + 0.5 * (df_taylor['yoy_change'] - inflation_target)
        
        # Merger avec Fed Funds
        df_fedfunds = get_fred_series('FEDFUNDS')
        if df_fedfunds is not None:
            df_taylor = pd.merge(
                df_taylor,
                df_fedfunds[['date', 'value']].rename(columns={'value': 'fed_funds'}),
                on='date',
                how='inner'
            )
            
            df_taylor['policy_gap'] = df_taylor['fed_funds'] - df_taylor['taylor_rate']
            
            col_taylor1, col_taylor2 = st.columns([2, 1])
            
            with col_taylor1:
                fig_taylor = go.Figure()
                
                fig_taylor.add_trace(go.Scatter(
                    x=df_taylor['date'],
                    y=df_taylor['taylor_rate'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2),
                    name='Taylor Rule Rate'
                ))
                
                fig_taylor.add_trace(go.Scatter(
                    x=df_taylor['date'],
                    y=df_taylor['fed_funds'],
                    mode='lines',
                    line=dict(color='#00FF00', width=2),
                    name='Actual Fed Funds'
                ))
                
                fig_taylor.update_layout(
                    title="Taylor Rule vs Actual Fed Funds Rate",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title="Rate (%)"),
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_taylor, use_container_width=True)
            
            with col_taylor2:
                current_taylor = df_taylor['taylor_rate'].iloc[-1]
                current_ff = df_taylor['fed_funds'].iloc[-1]
                policy_gap = df_taylor['policy_gap'].iloc[-1]
                
                st.metric(
                    label="TAYLOR RULE RATE",
                    value=f"{current_taylor:.2f}%"
                )
                
                st.metric(
                    label="ACTUAL FED FUNDS",
                    value=f"{current_ff:.2f}%"
                )
                
                st.metric(
                    label="POLICY GAP",
                    value=f"{policy_gap:+.2f}%",
                    delta="Tight" if policy_gap > 0 else "Loose"
                )
                
                if policy_gap > 1:
                    st.markdown("""
                    <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                        <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                        📈 RESTRICTIVE POLICY
                        </p>
                        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                        Fed funds above Taylor rule
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif policy_gap < -1:
                    st.markdown("""
                    <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                        <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                        📉 ACCOMMODATIVE POLICY
                        </p>
                        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                        Fed funds below Taylor rule
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# TAB 5: RECHERCHE
with tab5:
    st.markdown("### 🔍 SEARCH FRED SERIES")
    
    st.markdown("#### 📚 AVAILABLE SERIES BY CATEGORY")
    
    for category, series in ECONOMIC_SERIES.items():
        with st.expander(f"📊 {category}", expanded=False):
            for series_id, description in series.items():
                st.markdown(f"**`{series_id}`** - {description}")
    
    st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔍 CUSTOM SERIES SEARCH")
    
    custom_series_id = st.text_input(
        "Enter FRED Series ID",
        placeholder="Ex: GDP, UNRATE, CPIAUCSL...",
        key="custom_series_input"
    ).upper()
    
    if st.button("🔍 SEARCH", use_container_width=True, key="search_custom"):
        if custom_series_id:
            df = get_fred_series(custom_series_id)
            
            if df is not None:
                st.success(f"✅ Series found: {custom_series_id}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['value'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2)
                ))
                
                fig.update_layout(
                    title=f"{custom_series_id} - Historical Data",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333'),
                    yaxis=dict(gridcolor='#333'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Latest 20 Observations:**")
                st.dataframe(df.tail(20).sort_values('date', ascending=False), use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ Series '{custom_series_id}' not found")
        else:
            st.warning("⚠️ Please enter a series ID")

# TAB 4: TÉLÉCHARGEMENT
with tab4:
    st.markdown("### 📥 DOWNLOAD ECONOMIC DATA")
    
    download_category = st.selectbox(
        "SELECT CATEGORY",
        options=list(ECONOMIC_SERIES.keys()),
        key="download_category"
    )
    
    series_in_download = ECONOMIC_SERIES[download_category]
    
    download_series = st.multiselect(
        "SELECT SERIES",
        options=list(series_in_download.keys()),
        default=list(series_in_download.keys()),
        format_func=lambda x: f"{x} - {series_in_download[x]}",
        key="download_series"
    )
    
    if st.button("📥 DOWNLOAD CSV", use_container_width=True, key="download_button"):
        if download_series:
            combined_df = pd.DataFrame()
            
            for series_id in download_series:
                df = get_fred_series(series_id)
                if df is not None:
                    df_temp = df[['date', 'value']].copy()
                    df_temp = df_temp.rename(columns={'value': series_id})
                    
                    if combined_df.empty:
                        combined_df = df_temp
                    else:
                        combined_df = pd.merge(combined_df, df_temp, on='date', how='outer')
            
            if not combined_df.empty:
                csv_data = combined_df.to_csv(index=False)
                st.download_button(
                    label="💾 DOWNLOAD CSV",
                    data=csv_data,
                    file_name=f"economic_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success(f"✅ Data prepared! {len(combined_df)} observations")
                st.dataframe(combined_df.head(20), use_container_width=True)

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | FEDERAL RESERVE ECONOMIC DATA (FRED) | LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)


# TAB 6: TESTS ÉCONOMÉTRIQUES
with tab6:
    st.markdown("### 🔬 ECONOMETRIC TESTS & MODELS")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #00FF00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
        ⚡ ADVANCED ECONOMETRICS TOOLKIT
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Tests de stationnarité, cointégration, VAR/VECM, Granger causality, ARIMA forecasting, et plus.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sous-onglets pour organiser les tests
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "📊 STATIONARITY", 
        "🔗 COINTEGRATION", 
        "📈 VAR/VECM", 
        "🎯 FORECASTING",
        "🤖 ML MODELS"
    ])
    
    # ===== SUBTAB 1: TESTS DE STATIONNARITÉ =====
    with subtab1:
        st.markdown("#### 📊 STATIONARITY TESTS")
        st.caption("ADF (Augmented Dickey-Fuller) & KPSS Tests")
        
        col_stat1, col_stat2 = st.columns([2, 1])
        
        with col_stat1:
            # Sélection de série
            all_series_flat = []
            for category, series in ECONOMIC_SERIES.items():
                for series_id in series.keys():
                    all_series_flat.append(series_id)
            
            selected_series_stat = st.selectbox(
                "SELECT SERIES FOR STATIONARITY TEST",
                options=all_series_flat,
                key="stat_series_select"
            )
        
        with col_stat2:
            test_type = st.radio(
                "TEST TYPE",
                options=["Both", "ADF Only", "KPSS Only"],
                key="stat_test_type"
            )
        
        if st.button("🔬 RUN STATIONARITY TESTS", use_container_width=True, key="run_stat_test"):
            df_series = get_fred_series(selected_series_stat)
            
            if df_series is not None and len(df_series) > 30:
                try:
                    from statsmodels.tsa.stattools import adfuller, kpss
                    
                    series_values = df_series['value'].dropna()
                    
                    st.markdown(f"### Results for: {selected_series_stat}")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    # ADF Test
                    if test_type in ["Both", "ADF Only"]:
                        with col_res1:
                            st.markdown("**🔹 ADF Test (Null: Unit Root exists)**")
                            
                            adf_result = adfuller(series_values, autolag='AIC')
                            
                            adf_data = {
                                'Metric': ['ADF Statistic', 'p-value', '# Lags Used', '# Observations'],
                                'Value': [f"{adf_result[0]:.4f}", f"{adf_result[1]:.4f}", 
                                         f"{adf_result[2]}", f"{adf_result[3]}"]
                            }
                            
                            st.dataframe(pd.DataFrame(adf_data), hide_index=True, use_container_width=True)
                            
                            st.markdown("**Critical Values:**")
                            crit_vals = pd.DataFrame({
                                'Level': ['1%', '5%', '10%'],
                                'Critical Value': [f"{adf_result[4]['1%']:.4f}", 
                                                  f"{adf_result[4]['5%']:.4f}", 
                                                  f"{adf_result[4]['10%']:.4f}"]
                            })
                            st.dataframe(crit_vals, hide_index=True, use_container_width=True)
                            
                            # Interprétation
                            if adf_result[1] < 0.05:
                                st.markdown("""
                                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                                    <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                                    ✅ STATIONARY (p < 0.05)
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    Reject null hypothesis - No unit root
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                                    <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                                    ⚠️ NON-STATIONARY (p ≥ 0.05)
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    Fail to reject - Unit root present
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # KPSS Test
                    if test_type in ["Both", "KPSS Only"]:
                        with col_res2:
                            st.markdown("**🔹 KPSS Test (Null: Series is stationary)**")
                            
                            kpss_result = kpss(series_values, regression='c', nlags='auto')
                            
                            kpss_data = {
                                'Metric': ['KPSS Statistic', 'p-value', '# Lags Used'],
                                'Value': [f"{kpss_result[0]:.4f}", f"{kpss_result[1]:.4f}", f"{kpss_result[2]}"]
                            }
                            
                            st.dataframe(pd.DataFrame(kpss_data), hide_index=True, use_container_width=True)
                            
                            st.markdown("**Critical Values:**")
                            crit_vals_kpss = pd.DataFrame({
                                'Level': ['10%', '5%', '2.5%', '1%'],
                                'Critical Value': [f"{kpss_result[3]['10%']:.4f}", 
                                                  f"{kpss_result[3]['5%']:.4f}",
                                                  f"{kpss_result[3]['2.5%']:.4f}",
                                                  f"{kpss_result[3]['1%']:.4f}"]
                            })
                            st.dataframe(crit_vals_kpss, hide_index=True, use_container_width=True)
                            
                            # Interprétation (inverse de ADF)
                            if kpss_result[1] > 0.05:
                                st.markdown("""
                                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                                    <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                                    ✅ STATIONARY (p > 0.05)
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    Fail to reject null - Series is stationary
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                                    <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                                    ⚠️ NON-STATIONARY (p ≤ 0.05)
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    Reject null - Series is not stationary
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Test sur les différences
                    st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                    st.markdown("#### 📊 TEST ON FIRST DIFFERENCES")
                    
                    series_diff = series_values.diff().dropna()
                    
                    if len(series_diff) > 30:
                        adf_diff = adfuller(series_diff, autolag='AIC')
                        
                        col_diff1, col_diff2 = st.columns(2)
                        
                        with col_diff1:
                            st.metric("ADF Statistic (Diff)", f"{adf_diff[0]:.4f}")
                            st.metric("p-value (Diff)", f"{adf_diff[1]:.4f}")
                        
                        with col_diff2:
                            if adf_diff[1] < 0.05:
                                st.markdown("""
                                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px;">
                                    <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                                    ✅ FIRST DIFFERENCE IS STATIONARY
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    Series is I(1) - Integrated of order 1
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Graphique de la série
                    st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                    st.markdown("#### 📈 SERIES VISUALIZATION")
                    
                    fig_stat = make_subplots(rows=2, cols=1, 
                                            subplot_titles=('Original Series', 'First Difference'))
                    
                    fig_stat.add_trace(
                        go.Scatter(x=df_series['date'], y=df_series['value'], 
                                  mode='lines', line=dict(color='#FFAA00', width=1)),
                        row=1, col=1
                    )
                    
                    df_diff = df_series.copy()
                    df_diff['diff'] = df_diff['value'].diff()
                    
                    fig_stat.add_trace(
                        go.Scatter(x=df_diff['date'], y=df_diff['diff'], 
                                  mode='lines', line=dict(color='#00FFFF', width=1)),
                        row=2, col=1
                    )
                    
                    fig_stat.update_layout(
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        height=500,
                        showlegend=False
                    )
                    
                    fig_stat.update_xaxes(gridcolor='#333')
                    fig_stat.update_yaxes(gridcolor='#333')
                    
                    st.plotly_chart(fig_stat, use_container_width=True)
                    
                except ImportError:
                    st.error("❌ statsmodels not installed. Run: pip install statsmodels")
                except Exception as e:
                    st.error(f"❌ Error running tests: {e}")
            else:
                st.warning("⚠️ Not enough data points (need > 30)")
    
    # ===== SUBTAB 2: COINTÉGRATION =====
    with subtab2:
        st.markdown("#### 🔗 COINTEGRATION TESTS")
        st.caption("Engle-Granger & Johansen Tests for long-run equilibrium")
        
        col_coint1, col_coint2 = st.columns([2, 1])
        
        with col_coint1:
            series_coint_1 = st.selectbox(
                "SELECT FIRST SERIES",
                options=all_series_flat,
                key="coint_series1"
            )
            
            series_coint_2 = st.selectbox(
                "SELECT SECOND SERIES",
                options=all_series_flat,
                index=1,
                key="coint_series2"
            )
        
        with col_coint2:
            coint_test_type = st.radio(
                "TEST METHOD",
                options=["Engle-Granger", "Johansen"],
                key="coint_test_method"
            )
        
        if st.button("🔬 RUN COINTEGRATION TEST", use_container_width=True, key="run_coint_test"):
            df1 = get_fred_series(series_coint_1)
            df2 = get_fred_series(series_coint_2)
            
            if df1 is not None and df2 is not None:
                try:
                    from statsmodels.tsa.stattools import coint
                    from statsmodels.tsa.vector_ar.vecm import coint_johansen
                    
                    # Merger les deux séries
                    df_merged = pd.merge(
                        df1[['date', 'value']].rename(columns={'value': 'series1'}),
                        df2[['date', 'value']].rename(columns={'value': 'series2'}),
                        on='date',
                        how='inner'
                    )
                    
                    if len(df_merged) > 30:
                        if coint_test_type == "Engle-Granger":
                            st.markdown("### 📊 ENGLE-GRANGER COINTEGRATION TEST")
                            
                            # Test de cointégration
                            coint_stat, pvalue, crit_vals = coint(df_merged['series1'], df_merged['series2'])
                            
                            col_eg1, col_eg2 = st.columns(2)
                            
                            with col_eg1:
                                st.metric("Test Statistic", f"{coint_stat:.4f}")
                                st.metric("p-value", f"{pvalue:.4f}")
                            
                            with col_eg2:
                                st.markdown("**Critical Values:**")
                                crit_df = pd.DataFrame({
                                    'Level': ['1%', '5%', '10%'],
                                    'Value': [f"{crit_vals[0]:.4f}", f"{crit_vals[1]:.4f}", f"{crit_vals[2]:.4f}"]
                                })
                                st.dataframe(crit_df, hide_index=True, use_container_width=True)
                            
                            # Interprétation
                            if pvalue < 0.05:
                                st.markdown("""
                                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 10px; margin: 10px 0;">
                                    <p style="margin: 0; font-size: 11px; color: #00FF00; font-weight: bold;">
                                    ✅ COINTEGRATED (p < 0.05)
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    Series share a long-run equilibrium relationship. A VECM model can be estimated.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 10px; margin: 10px 0;">
                                    <p style="margin: 0; font-size: 11px; color: #FF6600; font-weight: bold;">
                                    ⚠️ NOT COINTEGRATED (p ≥ 0.05)
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    No evidence of long-run equilibrium. Consider differencing or VAR in levels.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        else:  # Johansen
                            st.markdown("### 📊 JOHANSEN COINTEGRATION TEST")
                            
                            data_matrix = df_merged[['series1', 'series2']].values
                            
                            johansen_result = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
                            
                            st.markdown("**Trace Statistic:**")
                            trace_df = pd.DataFrame({
                                'Rank': ['r = 0', 'r ≤ 1'],
                                'Trace Stat': [f"{johansen_result.trace_stat[0]:.4f}", 
                                              f"{johansen_result.trace_stat[1]:.4f}"],
                                'Crit 5%': [f"{johansen_result.trace_stat_crit_vals[0, 1]:.4f}",
                                           f"{johansen_result.trace_stat_crit_vals[1, 1]:.4f}"]
                            })
                            st.dataframe(trace_df, hide_index=True, use_container_width=True)
                            
                            # Interprétation
                            if johansen_result.trace_stat[0] > johansen_result.trace_stat_crit_vals[0, 1]:
                                st.markdown("""
                                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 10px; margin: 10px 0;">
                                    <p style="margin: 0; font-size: 11px; color: #00FF00; font-weight: bold;">
                                    ✅ AT LEAST 1 COINTEGRATING RELATIONSHIP
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                                    Series are cointegrated. VECM is appropriate.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Graphique des deux séries
                        st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                        st.markdown("#### 📈 SERIES COMPARISON")
                        
                        fig_coint = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig_coint.add_trace(
                            go.Scatter(x=df_merged['date'], y=df_merged['series1'], 
                                      name=series_coint_1, line=dict(color='#FFAA00', width=2)),
                            secondary_y=False
                        )
                        
                        fig_coint.add_trace(
                            go.Scatter(x=df_merged['date'], y=df_merged['series2'], 
                                      name=series_coint_2, line=dict(color='#00FFFF', width=2)),
                            secondary_y=True
                        )
                        
                        fig_coint.update_layout(
                            title=f"{series_coint_1} vs {series_coint_2}",
                            paper_bgcolor='#000',
                            plot_bgcolor='#111',
                            font=dict(color='#FFAA00', size=10),
                            height=400
                        )
                        
                        fig_coint.update_xaxes(gridcolor='#333')
                        fig_coint.update_yaxes(gridcolor='#333')
                        
                        st.plotly_chart(fig_coint, use_container_width=True)
                    
                    else:
                        st.warning("⚠️ Not enough overlapping data points")
                
                except ImportError:
                    st.error("❌ statsmodels not installed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ===== SUBTAB 3: VAR/VECM =====
    with subtab3:
        st.markdown("#### 📈 VAR/VECM MODELS")
        st.caption("Vector Autoregression & Vector Error Correction Models")
        
        st.markdown("""
        <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
            <p style="margin: 0; font-size: 10px; color: #FFAA00;">
            🔧 VAR/VECM implementation requires substantial computation. 
            Select 2-3 series and a reasonable lag order.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sélection multi-séries
        var_series = st.multiselect(
            "SELECT 2-3 SERIES FOR VAR",
            options=all_series_flat,
            default=[all_series_flat[0], all_series_flat[1]],
            max_selections=3,
            key="var_series_select"
        )
        
        col_var1, col_var2 = st.columns(2)
        
        with col_var1:
            var_lags = st.slider("NUMBER OF LAGS", min_value=1, max_value=12, value=4, key="var_lags")
        
        with col_var2:
            var_model_type = st.selectbox("MODEL TYPE", options=["VAR", "VECM"], key="var_model_type")
        
        if st.button("🔬 ESTIMATE VAR MODEL", use_container_width=True, key="run_var"):
            if len(var_series) >= 2:
                try:
                    from statsmodels.tsa.api import VAR
                    from statsmodels.tsa.vector_ar.vecm import VECM
                    
                    # Récupérer toutes les séries
                    dfs = []
                    for sid in var_series:
                        df_temp = get_fred_series(sid)
                        if df_temp is not None:
                            dfs.append(df_temp[['date', 'value']].rename(columns={'value': sid}))
                    
                    # Merger
                    df_var = dfs[0]
                    for df in dfs[1:]:
                        df_var = pd.merge(df_var, df, on='date', how='inner')
                    
                    df_var = df_var.dropna()
                    
                    if len(df_var) > var_lags * 3:
                        data_for_var = df_var[var_series].values
                        
                        if var_model_type == "VAR":
                            st.markdown("### 📊 VAR MODEL RESULTS")
                            
                            model = VAR(data_for_var)
                            results = model.fit(var_lags)
                            
                            st.markdown(f"**Model Summary:**")
                            st.text(str(results.summary()))
                            
                            # Granger causality
                            st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                            st.markdown("#### 🎯 GRANGER CAUSALITY TESTS")
                            
                            from statsmodels.tsa.stattools import grangercausalitytests
                            
                            for i, caused in enumerate(var_series):
                                for causing in var_series:
                                    if caused != causing:
                                        st.markdown(f"**Does {causing} Granger-cause {caused}?**")
                                        
                                        data_gc = df_var[[caused, causing]].values
                                        gc_res = grangercausalitytests(data_gc, maxlag=var_lags, verbose=False)
                                        
                                        # Extraire p-value du test F
                                        pval = gc_res[var_lags][0]['ssr_ftest'][1]
                                        
                                        if pval < 0.05:
                                            st.success(f"✅ YES (p={pval:.4f})")
                                        else:
                                            st.info(f"❌ NO (p={pval:.4f})")
                            
                            # Impulse Response
                            st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                            st.markdown("#### 📈 IMPULSE RESPONSE FUNCTIONS")
                            
                            irf = results.irf(10)
                            
                            fig_irf = irf.plot(impulse=var_series[0], response=var_series[1])
                            st.pyplot(fig_irf)
                        
                        else:  # VECM
                            st.markdown("### 📊 VECM MODEL (under construction)")
                            st.info("VECM estimation requires cointegration testing first")
                    
                    else:
                        st.warning("⚠️ Not enough data for VAR estimation")
                
                except ImportError:
                    st.error("❌ statsmodels not installed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Select at least 2 series")
    
    # ===== SUBTAB 4: FORECASTING =====
    with subtab4:
        st.markdown("#### 🎯 TIME SERIES FORECASTING")
        st.caption("ARIMA, SARIMAX, and Exponential Smoothing")
        
        col_fcst1, col_fcst2 = st.columns([2, 1])
        
        with col_fcst1:
            fcst_series = st.selectbox(
                "SELECT SERIES TO FORECAST",
                options=all_series_flat,
                key="fcst_series"
            )
        
        with col_fcst2:
            fcst_horizon = st.slider("FORECAST HORIZON (periods)", min_value=3, max_value=24, value=12, key="fcst_horizon")
            
            fcst_method = st.selectbox(
                "METHOD",
                options=["Auto ARIMA", "SARIMAX", "Exponential Smoothing"],
                key="fcst_method"
            )
        
        if st.button("🎯 GENERATE FORECAST", use_container_width=True, key="run_forecast"):
            df_fcst = get_fred_series(fcst_series)
            
            if df_fcst is not None and len(df_fcst) > 50:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    
                    st.markdown(f"### 📊 FORECAST: {fcst_series}")
                    
                    series_data = df_fcst['value'].values
                    dates = df_fcst['date'].values
                    
                    if fcst_method == "Auto ARIMA":
                        # Simple ARIMA(1,1,1) pour démo
                        model = ARIMA(series_data, order=(1,1,1))
                        fitted = model.fit()
                        
                        forecast = fitted.forecast(steps=fcst_horizon)
                        
                        # Graphique
                        fig_fcst = go.Figure()
                        
                        fig_fcst.add_trace(go.Scatter(
                            x=dates,
                            y=series_data,
                            mode='lines',
                            name='Historical',
                            line=dict(color='#FFAA00', width=2)
                        ))
                        
                        # Dates futures
                        last_date = pd.to_datetime(dates[-1])
                        freq = pd.infer_freq(dates)
                        if freq is None:
                            freq = 'M'  # Default monthly
                        
                        future_dates = pd.date_range(start=last_date, periods=fcst_horizon+1, freq=freq)[1:]
                        
                        fig_fcst.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast,
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#00FF00', width=2, dash='dash')
                        ))
                        
                        fig_fcst.update_layout(
                            title=f"{fcst_series} - ARIMA Forecast",
                            paper_bgcolor='#000',
                            plot_bgcolor='#111',
                            font=dict(color='#FFAA00', size=10),
                            xaxis=dict(gridcolor='#333'),
                            yaxis=dict(gridcolor='#333'),
                            height=400
                        )
                        
                        st.plotly_chart(fig_fcst, use_container_width=True)
                        
                        # Statistiques du modèle
                        st.markdown("**Model Statistics:**")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("AIC", f"{fitted.aic:.2f}")
                        
                        with col_stat2:
                            st.metric("BIC", f"{fitted.bic:.2f}")
                        
                        with col_stat3:
                            st.metric("Log-Likelihood", f"{fitted.llf:.2f}")
                        
                        # Forecast values table
                        st.markdown("**Forecast Values:**")
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecast': forecast,
                            'Lower 95%': forecast - 1.96 * fitted.forecast(steps=fcst_horizon).std(),
                            'Upper 95%': forecast + 1.96 * fitted.forecast(steps=fcst_horizon).std()
                        })
                        st.dataframe(forecast_df, hide_index=True, use_container_width=True)
                    
                    elif fcst_method == "SARIMAX":
                        st.info("🔧 SARIMAX with seasonal components (12,1,1)(1,1,1,12)")
                        
                        # SARIMAX avec saisonnalité
                        try:
                            model = SARIMAX(series_data, order=(1,1,1), seasonal_order=(1,1,1,12))
                            fitted = model.fit(disp=False)
                            
                            forecast = fitted.forecast(steps=fcst_horizon)
                            
                            # Graphique
                            fig_fcst = go.Figure()
                            
                            fig_fcst.add_trace(go.Scatter(
                                x=dates,
                                y=series_data,
                                mode='lines',
                                name='Historical',
                                line=dict(color='#FFAA00', width=2)
                            ))
                            
                            last_date = pd.to_datetime(dates[-1])
                            freq = pd.infer_freq(dates) or 'M'
                            future_dates = pd.date_range(start=last_date, periods=fcst_horizon+1, freq=freq)[1:]
                            
                            fig_fcst.add_trace(go.Scatter(
                                x=future_dates,
                                y=forecast,
                                mode='lines',
                                name='Forecast',
                                line=dict(color='#00FF00', width=2, dash='dash')
                            ))
                            
                            # Confidence intervals
                            forecast_ci = fitted.get_forecast(steps=fcst_horizon).conf_int()
                            
                            fig_fcst.add_trace(go.Scatter(
                                x=future_dates,
                                y=forecast_ci.iloc[:, 0],
                                mode='lines',
                                name='Lower 95%',
                                line=dict(color='#666', width=1, dash='dot'),
                                showlegend=True
                            ))
                            
                            fig_fcst.add_trace(go.Scatter(
                                x=future_dates,
                                y=forecast_ci.iloc[:, 1],
                                mode='lines',
                                name='Upper 95%',
                                line=dict(color='#666', width=1, dash='dot'),
                                fill='tonexty',
                                fillcolor='rgba(102, 102, 102, 0.2)',
                                showlegend=True
                            ))
                            
                            fig_fcst.update_layout(
                                title=f"{fcst_series} - SARIMAX Forecast",
                                paper_bgcolor='#000',
                                plot_bgcolor='#111',
                                font=dict(color='#FFAA00', size=10),
                                xaxis=dict(gridcolor='#333'),
                                yaxis=dict(gridcolor='#333'),
                                height=400
                            )
                            
                            st.plotly_chart(fig_fcst, use_container_width=True)
                            
                            # Model stats
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("AIC", f"{fitted.aic:.2f}")
                            with col_s2:
                                st.metric("BIC", f"{fitted.bic:.2f}")
                            with col_s3:
                                st.metric("Log-Likelihood", f"{fitted.llf:.2f}")
                        
                        except Exception as e:
                            st.error(f"SARIMAX error: {e}. Trying simple ARIMA instead.")
                            # Fallback to simple ARIMA
                            model = ARIMA(series_data, order=(1,1,1))
                            fitted = model.fit()
                            forecast = fitted.forecast(steps=fcst_horizon)
                            st.warning("Fell back to ARIMA(1,1,1)")
                    
                    elif fcst_method == "Exponential Smoothing":
                        st.info("🔧 Exponential Smoothing (Holt-Winters)")
                        
                        try:
                            model = ExponentialSmoothing(series_data, seasonal_periods=12, trend='add', seasonal='add')
                            fitted = model.fit()
                            
                            forecast = fitted.forecast(steps=fcst_horizon)
                            
                            # Graphique
                            fig_fcst = go.Figure()
                            
                            fig_fcst.add_trace(go.Scatter(
                                x=dates,
                                y=series_data,
                                mode='lines',
                                name='Historical',
                                line=dict(color='#FFAA00', width=2)
                            ))
                            
                            last_date = pd.to_datetime(dates[-1])
                            freq = pd.infer_freq(dates) or 'M'
                            future_dates = pd.date_range(start=last_date, periods=fcst_horizon+1, freq=freq)[1:]
                            
                            fig_fcst.add_trace(go.Scatter(
                                x=future_dates,
                                y=forecast,
                                mode='lines',
                                name='Forecast',
                                line=dict(color='#00FF00', width=2, dash='dash')
                            ))
                            
                            fig_fcst.update_layout(
                                title=f"{fcst_series} - Exponential Smoothing Forecast",
                                paper_bgcolor='#000',
                                plot_bgcolor='#111',
                                font=dict(color='#FFAA00', size=10),
                                xaxis=dict(gridcolor='#333'),
                                yaxis=dict(gridcolor='#333'),
                                height=400
                            )
                            
                            st.plotly_chart(fig_fcst, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Exponential Smoothing error: {e}")
                
                except ImportError:
                    st.error("❌ statsmodels not installed. Run: pip install statsmodels")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Need at least 50 data points for forecasting")
    
    # ===== SUBTAB 5: MACHINE LEARNING =====
    with subtab5:
        st.markdown("#### 🤖 MACHINE LEARNING MODELS")
        st.caption("Random Forest, XGBoost, and Neural Networks for economic forecasting")
        
        st.markdown("""
        <div style="background-color: #0a0a0a; border-left: 3px solid #FF00FF; padding: 10px; margin: 10px 0;">
            <p style="margin: 0; font-size: 10px; color: #FF00FF; font-weight: bold;">
            🚀 ADVANCED ML FORECASTING
            </p>
            <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
            Uses multiple economic indicators as features to predict target variable.
            Includes automatic feature engineering (lags, rolling means, momentum indicators).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_ml1, col_ml2 = st.columns([2, 1])
        
        with col_ml1:
            ml_target = st.selectbox(
                "TARGET VARIABLE TO PREDICT",
                options=['CPIAUCSL', 'UNRATE', 'GDP', 'DGS10', 'FEDFUNDS'],
                key="ml_target"
            )
            
            ml_features = st.multiselect(
                "FEATURE VARIABLES (predictors)",
                options=[s for s in all_series_flat if s != ml_target],
                default=['FEDFUNDS', 'M2SL', 'UNRATE'][:3] if ml_target != 'FEDFUNDS' else ['CPIAUCSL', 'M2SL', 'UNRATE'],
                key="ml_features"
            )
        
        with col_ml2:
            ml_model_type = st.selectbox(
                "MODEL TYPE",
                options=["Random Forest", "XGBoost", "Linear Regression"],
                key="ml_model"
            )
            
            ml_lags = st.slider("NUMBER OF LAGS", min_value=1, max_value=12, value=6, key="ml_lags")
            
            ml_horizon = st.slider("FORECAST HORIZON", min_value=1, max_value=12, value=3, key="ml_horizon")
        
        if st.button("🤖 TRAIN ML MODEL", use_container_width=True, key="train_ml"):
            if len(ml_features) > 0:
                try:
                    import numpy as np
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                    from sklearn.model_selection import train_test_split
                    
                    st.markdown(f"### 🤖 ML MODEL: {ml_model_type}")
                    st.info(f"Target: {ml_target} | Features: {', '.join(ml_features)}")
                    
                    # Récupérer toutes les séries
                    all_series_needed = [ml_target] + ml_features
                    dfs_ml = []
                    
                    for sid in all_series_needed:
                        df_temp = get_fred_series(sid)
                        if df_temp is not None:
                            dfs_ml.append(df_temp[['date', 'value']].rename(columns={'value': sid}))
                    
                    # Merger
                    if len(dfs_ml) > 0:
                        df_ml = dfs_ml[0]
                        for df in dfs_ml[1:]:
                            df_ml = pd.merge(df_ml, df, on='date', how='inner')
                        
                        df_ml = df_ml.dropna()
                        df_ml = df_ml.sort_values('date').reset_index(drop=True)
                        
                        if len(df_ml) > 50:
                            # Feature engineering
                            st.markdown("**🔧 Feature Engineering...**")
                            
                            features_engineered = pd.DataFrame()
                            
                            # Ajouter les lags
                            for feature in ml_features:
                                for lag in range(1, ml_lags + 1):
                                    features_engineered[f'{feature}_lag{lag}'] = df_ml[feature].shift(lag)
                                
                                # Rolling means
                                features_engineered[f'{feature}_ma3'] = df_ml[feature].rolling(3).mean()
                                features_engineered[f'{feature}_ma6'] = df_ml[feature].rolling(6).mean()
                                
                                # Momentum
                                features_engineered[f'{feature}_mom'] = df_ml[feature].pct_change(3)
                            
                            # Target (future value)
                            target = df_ml[ml_target].shift(-ml_horizon)
                            
                            # Combiner
                            data_ml = pd.concat([features_engineered, target.rename('target')], axis=1)
                            data_ml = data_ml.dropna()
                            
                            if len(data_ml) > 30:
                                X = data_ml.drop('target', axis=1)
                                y = data_ml['target']
                                
                                # Split train/test
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, shuffle=False
                                )
                                
                                # Entraîner le modèle
                                if ml_model_type == "Random Forest":
                                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                                elif ml_model_type == "Linear Regression":
                                    model = LinearRegression()
                                else:  # XGBoost
                                    try:
                                        import xgboost as xgb
                                        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                                    except ImportError:
                                        st.warning("XGBoost not installed, using Random Forest instead")
                                        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                                
                                with st.spinner("Training model..."):
                                    model.fit(X_train, y_train)
                                
                                # Prédictions
                                y_pred_train = model.predict(X_train)
                                y_pred_test = model.predict(X_test)
                                
                                # Métriques
                                st.markdown("#### 📊 MODEL PERFORMANCE")
                                
                                col_perf1, col_perf2 = st.columns(2)
                                
                                with col_perf1:
                                    st.markdown("**Training Set:**")
                                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                    train_mae = mean_absolute_error(y_train, y_pred_train)
                                    train_r2 = r2_score(y_train, y_pred_train)
                                    
                                    st.metric("RMSE", f"{train_rmse:.4f}")
                                    st.metric("MAE", f"{train_mae:.4f}")
                                    st.metric("R²", f"{train_r2:.4f}")
                                
                                with col_perf2:
                                    st.markdown("**Test Set:**")
                                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                                    test_mae = mean_absolute_error(y_test, y_pred_test)
                                    test_r2 = r2_score(y_test, y_pred_test)
                                    
                                    st.metric("RMSE", f"{test_rmse:.4f}")
                                    st.metric("MAE", f"{test_mae:.4f}")
                                    st.metric("R²", f"{test_r2:.4f}")
                                
                                # Feature importance
                                if hasattr(model, 'feature_importances_'):
                                    st.markdown("#### 📊 FEATURE IMPORTANCE")
                                    
                                    importance_df = pd.DataFrame({
                                        'Feature': X.columns,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False).head(10)
                                    
                                    fig_imp = go.Figure()
                                    fig_imp.add_trace(go.Bar(
                                        x=importance_df['Importance'],
                                        y=importance_df['Feature'],
                                        orientation='h',
                                        marker=dict(color='#FFAA00')
                                    ))
                                    
                                    fig_imp.update_layout(
                                        title="Top 10 Most Important Features",
                                        paper_bgcolor='#000',
                                        plot_bgcolor='#111',
                                        font=dict(color='#FFAA00', size=10),
                                        xaxis=dict(gridcolor='#333', title="Importance"),
                                        yaxis=dict(gridcolor='#333'),
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_imp, use_container_width=True)
                                
                                # Graphique prédictions vs réel
                                st.markdown("#### 📈 PREDICTIONS VS ACTUAL")
                                
                                fig_pred = go.Figure()
                                
                                # Test set only
                                test_indices = y_test.index
                                test_dates = df_ml.loc[test_indices, 'date']
                                
                                fig_pred.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=y_test.values,
                                    mode='lines+markers',
                                    name='Actual',
                                    line=dict(color='#FFAA00', width=2)
                                ))
                                
                                fig_pred.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=y_pred_test,
                                    mode='lines+markers',
                                    name='Predicted',
                                    line=dict(color='#00FF00', width=2, dash='dash')
                                ))
                                
                                fig_pred.update_layout(
                                    title=f"{ml_target} - Actual vs Predicted (Test Set)",
                                    paper_bgcolor='#000',
                                    plot_bgcolor='#111',
                                    font=dict(color='#FFAA00', size=10),
                                    xaxis=dict(gridcolor='#333', title="Date"),
                                    yaxis=dict(gridcolor='#333', title="Value"),
                                    height=400
                                )
                                
                                st.plotly_chart(fig_pred, use_container_width=True)
                                
                                # Résiduals analysis
                                st.markdown("#### 📊 RESIDUALS ANALYSIS")
                                
                                residuals = y_test - y_pred_test
                                
                                col_res1, col_res2 = st.columns(2)
                                
                                with col_res1:
                                    fig_res = go.Figure()
                                    fig_res.add_trace(go.Scatter(
                                        x=test_dates,
                                        y=residuals,
                                        mode='markers',
                                        marker=dict(color='#FF6600', size=6)
                                    ))
                                    fig_res.add_hline(y=0, line_dash="dash", line_color="#FFAA00")
                                    
                                    fig_res.update_layout(
                                        title="Residuals over Time",
                                        paper_bgcolor='#000',
                                        plot_bgcolor='#111',
                                        font=dict(color='#FFAA00', size=10),
                                        xaxis=dict(gridcolor='#333'),
                                        yaxis=dict(gridcolor='#333'),
                                        height=300
                                    )
                                    
                                    st.plotly_chart(fig_res, use_container_width=True)
                                
                                with col_res2:
                                    fig_hist = go.Figure()
                                    fig_hist.add_trace(go.Histogram(
                                        x=residuals,
                                        nbinsx=20,
                                        marker=dict(color='#FFAA00')
                                    ))
                                    
                                    fig_hist.update_layout(
                                        title="Residuals Distribution",
                                        paper_bgcolor='#000',
                                        plot_bgcolor='#111',
                                        font=dict(color='#FFAA00', size=10),
                                        xaxis=dict(gridcolor='#333', title="Residual"),
                                        yaxis=dict(gridcolor='#333', title="Frequency"),
                                        height=300
                                    )
                                    
                                    st.plotly_chart(fig_hist, use_container_width=True)
                            
                            else:
                                st.warning("⚠️ Not enough data after feature engineering")
                        else:
                            st.warning("⚠️ Need at least 50 overlapping observations")
                    else:
                        st.error("❌ Could not retrieve data for selected series")
                
                except ImportError as ie:
                    st.error(f"❌ Missing library: {ie}. Install with: pip install scikit-learn xgboost")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please select at least one feature variable")
                        
# TAB 7: NOTES & EXPLICATIONS
with tab7:
    st.markdown("### 📝 TECHNICAL NOTES & METHODOLOGY")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #FFAA00;">
        📚 Référence rapide des concepts économétriques et statistiques utilisés dans ce terminal.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sous-onglets pour organiser
    notes_tab1, notes_tab2, notes_tab3, notes_tab4 = st.tabs([
        "📊 INDICATORS", 
        "🔬 TESTS", 
        "📈 MODELS",
        "📊 METRICS"
    ])
    
    # ===== NOTES TAB 1: INDICATEURS =====
    with notes_tab1:
        st.markdown("#### 📊 CONSTRUCTED INDICATORS")
        
        with st.expander("💰 INFLATION YoY", expanded=False):
            st.markdown("""
            **Formule:** `100 * (CPI_t / CPI_{t-12} - 1)`
            
            **À quoi ça sert :**
            - Mesure la variation des prix sur 12 mois
            - Indicateur clé pour la politique monétaire
            - Fed cible 2% d'inflation annuelle
            
            **Interprétation :**
            - \> 2% : Inflation au-dessus de la cible (politique restrictive probable)
            - ≈ 2% : Cible atteinte (politique neutre)
            - < 2% : Inflation faible (risque de déflation, politique accommodante)
            """)
        
        with st.expander("📊 REAL YIELD (10Y)", expanded=False):
            st.markdown("""
            **Formule:** `Nominal 10Y Treasury - CPI YoY`
            
            **À quoi ça sert :**
            - Mesure le rendement réel ajusté de l'inflation
            - Indicateur d'attractivité des obligations
            - Signal pour les investisseurs
            
            **Interprétation :**
            - Positif : Rendement réel attractif → incite à l'épargne
            - Négatif : Perte de pouvoir d'achat → favorise la consommation/investissement
            - Historiquement : moyenne ~2%
            """)
        
        with st.expander("📉 YIELD CURVE (10Y-2Y)", expanded=False):
            st.markdown("""
            **Formule:** `DGS10 - DGS2`
            
            **À quoi ça sert :**
            - **Indicateur #1 de récession** (fiabilité historique 80%+)
            - Reflète les anticipations économiques du marché
            
            **Interprétation :**
            - \> 0 : Courbe normale (croissance attendue)
            - ≈ 0 : Courbe plate (incertitude)
            - < 0 : **INVERSION** → Récession probable dans 12-18 mois
            
            **Pourquoi ça marche :**  
            Inversion = marché anticipe baisse des taux futurs (Fed combat récession)
            """)
        
        with st.expander("📉 YIELD CURVE (10Y-3M)", expanded=False):
            st.markdown("""
            **Formule:** `DGS10 - DGS3MO`
            
            **À quoi ça sert :**
            - Meilleur indicateur avancé que 10Y-2Y
            - Plus sensible aux changements de politique monétaire
            
            **Délai typique inversion → récession :** 6-12 mois
            
            **Avantage vs 10Y-2Y :**  
            3M reflète directement le taux directeur Fed
            """)
        
        with st.expander("💵 MONEY SUPPLY GROWTH (M2)", expanded=False):
            st.markdown("""
            **Formule:** `100 * (M2_t / M2_{t-12} - 1)`
            
            **À quoi ça sert :**
            - Mesure l'expansion/contraction monétaire
            - Lié à l'inflation (théorie quantitative de la monnaie)
            - Indicateur de liquidité dans l'économie
            
            **Interprétation :**
            - Croissance forte (>10%) : Risque inflationniste
            - Croissance modérée (2-6%) : Normal
            - **Contraction (<0%)** : Signal déflationniste très rare et dangereux
            
            **Note :** Contraction M2 a précédé la Grande Dépression (1929-1933)
            """)
        
        with st.expander("📈 REAL GDP QoQ ANNUALIZED", expanded=False):
            st.markdown("""
            **Formule:** `400 * (GDP_t / GDP_{t-1} - 1)`
            
            **À quoi ça sert :**
            - Mesure la croissance économique trimestrielle
            - Multipliée par 4 pour annualiser
            
            **Interprétation :**
            - \> 3% : Croissance forte
            - 2-3% : Croissance tendancielle US
            - 0-2% : Croissance faible
            - < 0 : Contraction
            - **2 trimestres négatifs consécutifs = Récession technique**
            
            **Limite :** Définition NBER de récession est plus large
            """)
        
        with st.expander("🎯 TAYLOR RULE", expanded=False):
            st.markdown("""
            **Formule complète:**  
            `i = r* + π + 0.5*(π - π*) + 0.5*y`
            
            Où :
            - `i` = taux directeur optimal
            - `r*` = taux neutre (~2%)
            - `π` = inflation actuelle
            - `π*` = cible inflation (2%)
            - `y` = output gap
            
            **Simplifiée (si y=0):**  
            `Taylor Rate = 2 + inflation + 0.5*(inflation - 2)`
            
            **À quoi ça sert :**
            - Règle normative pour la politique monétaire
            - Compare taux Fed vs taux "optimal"
            
            **Policy Gap = Fed Funds - Taylor Rate**
            - Gap > 0 : Politique **restrictive**
            - Gap < 0 : Politique **accommodante**
            """)
    
    # ===== NOTES TAB 2: TESTS =====
    with notes_tab2:
        st.markdown("#### 🔬 ECONOMETRIC TESTS")
        
        with st.expander("📊 ADF TEST (Augmented Dickey-Fuller)", expanded=False):
            st.markdown("""
            **Objectif :** Tester la stationnarité d'une série temporelle
            
            **Hypothèses :**
            - H₀ : La série a une racine unitaire (= NON stationnaire)
            - H₁ : La série est stationnaire
            
            **Interprétation :**
            - p-value < 0.05 → **Rejet H₀** → Série STATIONNAIRE ✅
            - p-value ≥ 0.05 → Série NON STATIONNAIRE ⚠️
            
            **Pourquoi c'est important :**
            - Régression sur séries non-stationnaires → régression fallacieuse
            - Besoin de différencier ou utiliser cointégration
            
            **Règle pratique :**
            - Série I(0) : stationnaire en niveau
            - Série I(1) : stationnaire en première différence
            """)
        
        with st.expander("📊 KPSS TEST", expanded=False):
            st.markdown("""
            **Objectif :** Test de stationnarité (complémentaire à ADF)
            
            **Hypothèses :**
            - H₀ : La série EST stationnaire (inverse de ADF !)
            - H₁ : La série a une racine unitaire
            
            **Interprétation :**
            - p-value > 0.05 → Série STATIONNAIRE ✅
            - p-value ≤ 0.05 → Série NON STATIONNAIRE ⚠️
            
            **Pourquoi utiliser ADF ET KPSS :**
            
            | ADF | KPSS | Conclusion |
            |-----|------|-----------|
            | Stationnaire | Stationnaire | ✅ Clairement stationnaire |
            | Non-stat | Non-stat | ⚠️ Incertain (tester différences) |
            | Stationnaire | Non-stat | 🔄 Trend-stationary |
            | Non-stat | Stationnaire | ⚠️ Résultats contradictoires |
            """)
        
        with st.expander("🔗 COINTEGRATION (Engle-Granger)", expanded=False):
            st.markdown("""
            **Objectif :** Tester si 2+ séries non-stationnaires partagent une relation long-terme
            
            **Méthode :**
            1. Régression : `Y = α + βX + ε`
            2. Test ADF sur les résidus `ε`
            
            **Hypothèses :**
            - H₀ : Pas de cointégration
            - H₁ : Cointégration existe
            
            **Interprétation :**
            - p < 0.05 → **Cointégrées** → Utiliser VECM ✅
            - p ≥ 0.05 → Pas cointégrées → Utiliser VAR en différences
            
            **Exemple classique :**
            - Rendement nominal 10Y et anticipations inflation
            - Prix spot et futures (arbitrage)
            
            **Implication :** Même si séries individuelles non-stationnaires,  
            leur combinaison linéaire est stationnaire
            """)
        
        with st.expander("🔗 COINTEGRATION (Johansen)", expanded=False):
            st.markdown("""
            **Objectif :** Test de cointégration pour systèmes multivariés
            
            **Avantages vs Engle-Granger :**
            - Plusieurs relations de cointégration possibles
            - Pas besoin de choisir variable dépendante
            - Plus puissant pour 3+ variables
            
            **Statistiques :**
            - **Trace statistic** : teste nombre de relations
            - **Max eigenvalue** : teste chaque relation individuellement
            
            **Interprétation Trace :**
            - r = 0 : Aucune relation
            - r ≤ 1 : Au moins 1 relation
            - r ≤ 2 : Au moins 2 relations
            
            Si Trace Stat > Critical Value → rejeter H₀
            
            **Utilisation :** Systèmes de taux (courbe des taux), devises, matières premières
            """)
        
        with st.expander("🎯 GRANGER CAUSALITY", expanded=False):
            st.markdown("""
            **Objectif :** X "cause" Y au sens de Granger si X passé améliore prédiction de Y
            
            **Ce n'est PAS une causalité vraie !**  
            C'est une "précédence temporelle prédictive"
            
            **Hypothèses :**
            - H₀ : X ne Granger-cause pas Y
            - H₁ : X Granger-cause Y
            
            **Interprétation :**
            - p < 0.05 → X améliore significativement prédiction de Y
            
            **Exemples classiques :**
            - Fed Funds → CPI ? (politique monétaire → inflation)
            - Consommation → PIB ?
            - VIX → S&P 500 ?
            
            **Piège :** "A Granger-cause B" ≠ "A cause B"  
            Peut juste refléter anticipations communes d'un 3e facteur
            """)
        
        with st.expander("📈 VAR (Vector AutoRegression)", expanded=False):
            st.markdown("""
            **Objectif :** Modéliser interactions dynamiques entre plusieurs variables
            
            **Structure :**  
            Chaque variable = f(ses propres lags + lags des autres variables)
            
            **Exemple VAR(2) à 2 variables :**
```
            Y₁ₜ = c₁ + α₁₁Y₁ₜ₋₁ + α₁₂Y₁ₜ₋₂ + β₁₁Y₂ₜ₋₁ + β₁₂Y₂ₜ₋₂ + ε₁ₜ
            Y₂ₜ = c₂ + α₂₁Y₁ₜ₋₁ + α₂₂Y₁ₜ₋₂ + β₂₁Y₂ₜ₋₁ + β₂₂Y₂ₜ₋₂ + ε₂ₜ
```
            
            **Applications :**
            - Impulse Response Functions (IRF) : choc sur X → effet sur Y
            - Forecast Error Variance Decomposition (FEVD)
            - Granger causality (test F joint)
            
            **Condition :** Variables doivent être stationnaires (ou cointégrées → VECM)
            
            **Choix du lag :** AIC, BIC, ou tests séquentiels
            """)
        
        with st.expander("📈 VECM (Vector Error Correction Model)", expanded=False):
            st.markdown("""
            **Objectif :** VAR pour variables I(1) cointégrées
            
            **Différence avec VAR :**  
            VECM = VAR en différences + **terme de correction d'erreur**
            
            **Structure :**
```
            ΔYₜ = α(βYₜ₋₁) + Γ₁ΔYₜ₋₁ + ... + εₜ
```
            
            Où :
            - `β` = vecteur de cointégration (relation long-terme)
            - `α` = vitesse d'ajustement
            - `Γᵢ` = dynamiques court-terme
            
            **Interprétation α :**
            - α = -0.2 → 20% de l'écart LT corrigé chaque période
            - |α| proche de 1 → ajustement rapide
            - α proche de 0 → ajustement lent
            
            **Usage :** Taux d'intérêt, spreads, relations arbitrage
            """)
    
    # ===== NOTES TAB 3: MODÈLES =====
    with notes_tab3:
        st.markdown("#### 📈 FORECASTING MODELS")
        
        with st.expander("📊 ARIMA (AutoRegressive Integrated Moving Average)", expanded=False):
            st.markdown("""
            **Structure :** ARIMA(p, d, q)
            
            **Composantes :**
            - **AR(p)** : Autoregressive = régression sur p valeurs passées
            - **I(d)** : Integrated = différenciation d ordre d
            - **MA(q)** : Moving Average = moyenne mobile des erreurs passées
            
            **Formule ARIMA(1,1,1) :**
```
            ΔYₜ = c + φ₁ΔYₜ₋₁ + θ₁εₜ₋₁ + εₜ
```
            
            **Identification :**
            - ACF (autocorrelation) → ordre MA
            - PACF (partial autocorrelation) → ordre AR
            - Ou utiliser Auto ARIMA (optimise AIC/BIC)
            
            **Quand l'utiliser :**
            - Série univariée
            - Dépendance temporelle claire
            - Pas de saisonnalité forte (sinon SARIMA)
            """)
        
        with st.expander("📊 SARIMAX (Seasonal ARIMA with eXogenous)", expanded=False):
            st.markdown("""
            **Structure :** SARIMAX(p,d,q)(P,D,Q,s)
            
            **Nouveauté vs ARIMA :**
            - **(P,D,Q,s)** : composante saisonnière de période s
            - **X** : variables exogènes (régresseurs)
            
            **Exemple SARIMAX(1,1,1)(1,1,1,12) :**
            - (1,1,1) : ARIMA standard
            - (1,1,1,12) : composante saisonnière mensuelle
            
            **Applications :**
            - CPI, ventes retail (saisonnalité)
            - Unemployment (cycles)
            - Variables exogènes : vacances, politique monétaire
            
            **Avantage :** Capture patterns saisonniers + effets exogènes
            """)
        
        with st.expander("📊 EXPONENTIAL SMOOTHING (Holt-Winters)", expanded=False):
            st.markdown("""
            **Principe :** Pondération exponentielle décroissante du passé
            
            **3 types :**
            1. **Simple** : niveau seulement
            2. **Holt** : niveau + tendance
            3. **Holt-Winters** : niveau + tendance + saisonnalité
            
            **Formules (Holt-Winters additif) :**
```
            Niveau    : Lₜ = α(Yₜ - Sₜ₋ₛ) + (1-α)(Lₜ₋₁ + Tₜ₋₁)
            Tendance  : Tₜ = β(Lₜ - Lₜ₋₁) + (1-β)Tₜ₋₁
            Saison    : Sₜ = γ(Yₜ - Lₜ) + (1-γ)Sₜ₋ₛ
            Prévision : Ŷₜ₊ₕ = Lₜ + hTₜ + Sₜ₊ₕ₋ₛ
```
            
            **Quand l'utiliser :**
            - Simple à implémenter
            - Patterns saisonniers réguliers
            - Moins flexible qu'ARIMA mais plus rapide
            
            **Variante multiplicative :** pour saisonnalité proportionnelle au niveau
            """)
        
        with st.expander("🤖 RANDOM FOREST REGRESSION", expanded=False):
            st.markdown("""
            **Principe :** Ensemble de nombreux arbres de décision
            
            **Fonctionnement :**
            1. Créer N arbres avec bootstrap samples
            2. À chaque split, considérer sous-ensemble aléatoire de features
            3. Prédiction = moyenne des prédictions de tous les arbres
            
            **Avantages :**
            - ✅ Capture non-linéarités
            - ✅ Interactions automatiques
            - ✅ Robuste au surapprentissage
            - ✅ Feature importance
            
            **Inconvénients :**
            - ❌ Moins bon pour extrapolation
            - ❌ Boîte noire (interprétabilité limitée)
            
            **Hyperparamètres clés :**
            - `n_estimators` : nombre d'arbres
            - `max_depth` : profondeur max (contrôle overfitting)
            - `min_samples_split` : observations min pour split
            """)
        
        with st.expander("🤖 XGBOOST (Extreme Gradient Boosting)", expanded=False):
            st.markdown("""
            **Principe :** Boosting = arbres séquentiels corrigeant erreurs précédentes
            
            **Différence vs Random Forest :**
            - RF : arbres parallèles indépendants
            - XGBoost : arbres séquentiels, chacun corrige le précédent
            
            **Avantages :**
            - ✅ Souvent meilleure performance que RF
            - ✅ Gestion native des valeurs manquantes
            - ✅ Régularisation intégrée (L1, L2)
            - ✅ Très rapide (implémentation optimisée)
            
            **Hyperparamètres clés :**
            - `learning_rate` : taux d'apprentissage (0.01-0.3)
            - `n_estimators` : nombre d'arbres
            - `max_depth` : profondeur (3-10)
            - `subsample` : fraction des données par arbre
            
            **Risque :** Overfitting si mal réglé → validation croisée essentielle
            """)
        
        with st.expander("🤖 FEATURE ENGINEERING", expanded=False):
            st.markdown("""
            **Variables créées automatiquement dans l'onglet ML :**
            
            **1. Lags (retards) :**
            - `X_lag1`, `X_lag2`, ..., `X_lag12`
            - Capture dépendance temporelle
            
            **2. Moving Averages :**
            - `X_ma3` : moyenne mobile 3 périodes
            - `X_ma6` : moyenne mobile 6 périodes
            - Lisse bruit, capture tendance
            
            **3. Momentum :**
            - `X_mom` = variation sur 3 périodes
            - `X_mom = (Xₜ - Xₜ₋₃) / Xₜ₋₃`
            
            **Pourquoi ça marche :**
            - Transforme série temporelle en problème supervisé
            - ML capte patterns non-linéaires que ARIMA rate
            
            **Piège :** Attention au data leakage !  
            Ne jamais utiliser info du futur pour prédire le passé
            """)
    
    # ===== NOTES TAB 4: MÉTRIQUES =====
    with notes_tab4:
        st.markdown("#### 📊 PERFORMANCE METRICS")
        
        with st.expander("📊 RMSE (Root Mean Squared Error)", expanded=False):
            st.markdown("""
            **Formule :**  
            `RMSE = √(Σ(yᵢ - ŷᵢ)² / n)`
            
            **Unité :** Même que la variable Y
            
            **Interprétation :**
            - RMSE = 0 : Prédiction parfaite
            - Plus RMSE petit → meilleur modèle
            - Pénalise fortement les grandes erreurs (²)
            
            **Exemple :**
            - Prédire CPI (indice ~300)
            - RMSE = 2.5 → erreur moyenne de 2.5 points d'indice
            
            **Avantage :** Sensible aux outliers (utile si coûteux)
            
            **Comparaison :** RMSE toujours ≥ MAE (égalité ssi toutes erreurs identiques)
            """)
        
        with st.expander("📊 MAE (Mean Absolute Error)", expanded=False):
            st.markdown("""
            **Formule :**  
            `MAE = Σ|yᵢ - ŷᵢ| / n`
            
            **Unité :** Même que la variable Y
            
            **Interprétation :**
            - MAE = erreur moyenne en valeur absolue
            - Plus robuste aux outliers que RMSE
            - Interprétation plus intuitive
            
            **Exemple :**
            - Prédire taux chômage (%)
            - MAE = 0.3 → erreur moyenne de 0.3 points de %
            
            **Quand préférer MAE vs RMSE :**
            - MAE : si outliers pas plus graves que petites erreurs
            - RMSE : si grandes erreurs très coûteuses
            """)
        
        with st.expander("📊 R² (Coefficient of Determination)", expanded=False):
            st.markdown("""
            **Formule :**  
            `R² = 1 - (SS_res / SS_tot)`
            
            Où :
            - `SS_res = Σ(yᵢ - ŷᵢ)²` : somme carrés résidus
            - `SS_tot = Σ(yᵢ - ȳ)²` : variance totale
            
            **Interprétation :**
            - R² = 1 : Modèle parfait (100% variance expliquée)
            - R² = 0 : Modèle = prédire la moyenne
            - R² < 0 : Modèle pire que la moyenne (très mauvais !)
            
            **Exemple :**
            - R² = 0.85 → 85% de la variance de Y expliquée par le modèle
            
            **ATTENTION :**
            - ❌ R² seul ne suffit pas (vérifier résidus)
            - ❌ R² augmente toujours si on ajoute variables → utiliser R² ajusté
            - ❌ R² élevé ≠ causalité
            
            **Règles empiriques :**
            - R² > 0.9 : Excellent (attention overfitting)
            - R² 0.7-0.9 : Bon
            - R² 0.5-0.7 : Moyen
            - R² < 0.5 : Faible pouvoir prédictif
            """)
        
        with st.expander("📊 AIC / BIC (Information Criteria)", expanded=False):
            st.markdown("""
            **Formules :**
```
            AIC = 2k - 2ln(L)
            BIC = k·ln(n) - 2ln(L)
```
            
            Où :
            - `k` = nombre de paramètres
            - `n` = nombre d'observations
            - `L` = vraisemblance du modèle
            
            **Objectif :** Équilibre fit vs complexité (pénalise surparamétrage)
            
            **Utilisation :**
            - Comparer modèles (plus petit = meilleur)
            - Sélection ordre AR/MA dans ARIMA
            - Choix nombre de lags dans VAR
            
            **AIC vs BIC :**
            - **BIC** pénalise plus la complexité (terme `ln(n)`)
            - BIC → modèles plus parcimonieux
            - AIC → meilleur pour prédiction
            - BIC → meilleur pour théorie
            
            **Règle :** Différence de 10 points = très significative
            """)
        
        with st.expander("📊 P-VALUE", expanded=False):
            st.markdown("""
            **Définition :** Probabilité d'observer résultat aussi extrême si H₀ vraie
            
            **Interprétation :**
            - p < 0.01 : Très significatif (***) → forte évidence contre H₀
            - p < 0.05 : Significatif (**) → rejet H₀ (seuil standard)
            - p < 0.10 : Faiblement significatif (*) → évidence marginale
            - p ≥ 0.10 : Non significatif → pas d'évidence contre H₀
            
            **CE QUE P-VALUE N'EST PAS :**
            - ❌ Probabilité que H₀ soit vraie
            - ❌ Probabilité de faire une erreur
            - ❌ Importance de l'effet
            
            **Piège du p-hacking :**
            Tester 20 hypothèses → 1 sera significatif à 5% par hasard !
            
            **Bonnes pratiques :**
            - Préspécifier hypothèses
            - Correction Bonferroni si tests multiples
            - Regarder aussi magnitude de l'effet (pas que significativité)
            """)
        
        with st.expander("📊 CRITICAL VALUES", expanded=False):
            st.markdown("""
            **Définition :** Seuils de rejet pour tests statistiques
            
            **Niveaux standards :**
            - **1%** : Très conservateur (forte évidence requise)
            - **5%** : Standard en économie/finance
            - **10%** : Plus permissif (sciences sociales)
            
            **Exemple ADF Test :**
            Statistique ADF = -3.5
        Critical value 5% = -2.86
        → -3.5 < -2.86 → Rejet H₀ → Série stationnaire
            
            **Lien avec p-value :**
            - Stat test < Critical value ⟺ p-value < seuil
            - Critical values sont pré-tabulés
            - p-value est probabilité exacte
            
            **One-tail vs Two-tail :**
            - One-tail : test directionnel (> ou <)
            - Two-tail : test non-directionnel (≠)
            - Critical values différents !
            """)
        
        with st.expander("📊 CONFIDENCE INTERVALS", expanded=False):
            st.markdown("""
            **Définition :** Intervalle contenant vraie valeur avec probabilité donnée
            
            **IC 95% :**  
            `[Estimate - 1.96·SE, Estimate + 1.96·SE]`
            
            **Interprétation :**
            - "Si on répète expérience 100 fois, 95 IC contiendront vraie valeur"
            - Plus IC étroit → estimation précise
            - Plus IC large → incertitude élevée
            
            **Exemple forecast :**
```
            Prévision CPI : 315.5
            IC 95% : [312.3, 318.7]
```
            → 95% de confiance que vraie valeur entre 312.3 et 318.7
            
            **Facteurs affectant largeur IC :**
            - ✅ Plus de données → IC plus étroit
            - ✅ Moins de variance → IC plus étroit
            - ❌ Horizon long → IC plus large
            
            **IC 90% vs 95% vs 99% :**
            - 90% : Plus étroit mais moins confiant
            - 95% : Standard (compromis)
            - 99% : Plus large mais plus confiant
            """)
        
        with st.expander("📊 RESIDUALS ANALYSIS", expanded=False):
            st.markdown("""
            **Définition :** Résidus = Erreurs de prédiction = `Y - Ŷ`
            
            **Propriétés d'un BON modèle :**
            
            1. **Moyenne nulle :** `E(ε) = 0`
               - Sinon : biais systématique
            
            2. **Homoscédasticité :** Variance constante
               - Test visuel : plot résidus vs fitted
               - Si entonnoir → hétéroscédasticité
            
            3. **Pas d'autocorrélation :** `Corr(εₜ, εₜ₋₁) = 0`
               - Test Durbin-Watson ou Ljung-Box
               - Si autocorrélés → info non capturée
            
            4. **Normalité :** `ε ~ N(0, σ²)`
               - Test visuel : histogramme, Q-Q plot
               - Pas crucial pour grandes données (CLT)
            
            **Diagnostics graphiques :**
```
            1. Résidus vs Temps → détecter patterns
            2. Résidus vs Fitted → détecter hétéroscédasticité
            3. Histogram résidus → vérifier normalité
            4. Q-Q plot → normalité
            5. ACF résidus → autocorrélation
```
            
            **Si résidus mauvais :**
            - Ajouter variables omises
            - Transformer Y (log, Box-Cox)
            - Changer spécification modèle
            """)
        
        with st.expander("📊 OVERFITTING vs UNDERFITTING", expanded=False):
            st.markdown("""
            **OVERFITTING (surapprentissage) :**
            - Modèle trop complexe
            - Excellentes performances sur train
            - Mauvaises performances sur test
            - Capture le bruit au lieu du signal
            
            **Signes :**
            - R² train = 0.99, R² test = 0.50
            - Modèle avec 100 paramètres pour 120 observations
            
            **Solutions :**
            - ✅ Régularisation (Ridge, Lasso)
            - ✅ Cross-validation
            - ✅ Early stopping
            - ✅ Simplifier modèle
            - ✅ Plus de données
            
            **UNDERFITTING (sous-apprentissage) :**
            - Modèle trop simple
            - Mauvaises performances train ET test
            - Ne capture pas patterns importants
            
            **Signes :**
            - R² train = 0.40, R² test = 0.38
            - Modèle linéaire pour relation non-linéaire
            
            **Solutions :**
            - ✅ Ajouter features
            - ✅ Polynômes / interactions
            - ✅ Modèle plus complexe
            
            **Sweet spot :** Train > Test, mais pas trop
            
            **Bias-Variance Tradeoff :**
```
            Error = Bias² + Variance + Irreducible Error
            
            Simple model → High Bias, Low Variance
            Complex model → Low Bias, High Variance
```
            """)
        
        with st.expander("📊 CROSS-VALIDATION", expanded=False):
            st.markdown("""
            **Objectif :** Évaluer performance sans gaspiller données
            
            **K-Fold CV (standard) :**
            1. Diviser données en K folds
            2. Entraîner sur K-1 folds
            3. Tester sur fold restant
            4. Répéter K fois
            5. Performance = moyenne des K tests
            
            **Time Series CV (IMPORTANT !) :**
            ⚠️ **NE PAS utiliser K-Fold standard !**
            
            Problème : mélange passé/futur → data leakage
            
            **Solution : Walk-Forward / Rolling Window**
```
            Train: [1...100] → Test: [101...110]
            Train: [1...110] → Test: [111...120]
            Train: [1...120] → Test: [121...130]
            ...
```
            
            **Expanding Window :**
            - Train grandit à chaque étape
            - Utilise toute l'histoire
            
            **Rolling Window :**
            - Taille train fixe
            - Plus adaptatif aux changements
            
            **Règle d'or :** Ne JAMAIS entraîner sur données futures !
            """)
        
        with st.expander("📊 FEATURE IMPORTANCE", expanded=False):
            st.markdown("""
            **Objectif :** Identifier variables les plus prédictives
            
            **Méthodes selon modèle :**
            
            **1. Tree-based (RF, XGBoost) :**
            - Importance = réduction moyenne d'impureté (Gini/entropy)
            - Ou permutation importance
            
            **2. Linear models :**
            - Coefficients (si variables standardisées)
            - Ou coefficients régularisés (Lasso)
            
            **3. Permutation Importance (général) :**
            - Mélanger valeurs d'une feature
            - Mesurer baisse de performance
            - Grande baisse → feature importante
            
            **Interprétation :**
```
            Feature          | Importance
            -----------------|-----------
            FEDFUNDS_lag1   |   0.25     ← 25% de l'importance
            CPI_lag3        |   0.18
            M2_ma6          |   0.12
            ...
```
            
            **Attention :**
            - Variables corrélées → importance partagée
            - Importance ≠ causalité
            - Dépend du modèle
            
            **Usage :**
            - Feature selection
            - Interpréter "boîte noire"
            - Identifier drivers économiques
            """)
        
        with st.expander("📊 STATIONARITY (Pourquoi c'est crucial)", expanded=False):
            st.markdown("""
            **Série stationnaire :** Propriétés statistiques constantes dans le temps
            
            **Conditions :**
            1. Moyenne constante : `E(Yₜ) = μ` ∀t
            2. Variance constante : `Var(Yₜ) = σ²` ∀t
            3. Covariance dépend que du lag : `Cov(Yₜ, Yₜ₋ₖ) = γₖ`
            
            **Pourquoi c'est important :**
            
            ❌ **Sans stationnarité :**
            - Régression fallacieuse (spurious regression)
            - R² élevé mais relation inexistante
            - Tests statistiques invalides
            
            **Exemple célèbre (Yule 1926) :**
            Régression "Mortalité UK" sur "Mariages Église d'Angleterre"  
            → R² = 0.95 mais aucun lien causal !  
            (Les deux sont trending)
            
            ✅ **Avec stationnarité :**
            - Inférence statistique valide
            - Prévisions fiables
            - Relations vraies (pas artefacts)
            
            **Solutions si non-stationnaire :**
            1. **Différenciation :** `ΔYₜ = Yₜ - Yₜ₋₁`
            2. **Détrending :** Retirer tendance
            3. **Transformation :** Log, Box-Cox
            4. **Cointégration :** Si relation long-terme vraie
            
            **Types de non-stationnarité :**
            - **Trend-stationary :** Détrending suffit
            - **Difference-stationary (I(1))** : Différenciation nécessaire
            - **Structural breaks :** Changement de régime
            """)

        with st.expander("📚 GLOSSARY - QUICK REFERENCE", expanded=False):
            st.markdown("""
            | Terme | Définition |
            |-------|-----------|
            | **AR** | AutoRegressive - régression sur valeurs passées |
            | **MA** | Moving Average - moyenne des erreurs passées |
            | **I(d)** | Integrated of order d - différenciation d fois pour stationnarité |
            | **Lag** | Retard temporel (lag 1 = période précédente) |
            | **White noise** | Erreurs non corrélées, variance constante |
            | **Unit root** | Racine = 1 → série non-stationnaire |
            | **Spurious regression** | Régression fallacieuse sur séries non-stationnaires |
            | **Cointegration** | Relation long-terme entre séries I(1) |
            | **ECM** | Error Correction Model - ajustement vers équilibre |
            | **IRF** | Impulse Response Function - effet d'un choc |
            | **FEVD** | Forecast Error Variance Decomposition |
            | **Exogenous** | Variable externe non expliquée par le modèle |
            | **Endogenous** | Variable expliquée par le modèle |
            | **Heteroskedasticity** | Variance non constante |
            | **Autocorrelation** | Corrélation avec valeurs passées |
            | **OLS** | Ordinary Least Squares - MCO |
            | **MLE** | Maximum Likelihood Estimation |
            | **QQ-plot** | Quantile-Quantile plot - test normalité graphique |
            | **ACF** | AutoCorrelation Function |
            | **PACF** | Partial AutoCorrelation Function |
            | **Nowcasting** | Prévision temps présent (avec données incomplètes) |
            """)
        
        with st.expander("🔗 USEFUL RESOURCES", expanded=False):
            st.markdown("""
            **📚 Livres de référence :**
            
            **Économétrie séries temporelles :**
            - Hamilton (1994) - Time Series Analysis
            - Enders (2015) - Applied Econometric Time Series
            - Tsay (2010) - Analysis of Financial Time Series
            
            **Machine Learning :**
            - Hastie & Tibshirani - Elements of Statistical Learning
            - James et al. - Introduction to Statistical Learning
            
            **Macroéconomie appliquée :**
            - Stock & Watson - Introduction to Econometrics
            - Wooldridge - Econometric Analysis of Cross Section and Panel Data
            
            **📊 Bases de données :**
            - FRED (Federal Reserve) : https://fred.stlouisfed.org
            - BEA (Bureau of Economic Analysis)
            - BLS (Bureau of Labor Statistics)
            - World Bank Open Data
            - IMF Data
            
            **🛠️ Outils Python :**
            - `statsmodels` : Tests économétriques, ARIMA, VAR
            - `scikit-learn` : Machine learning
            - `prophet` : Forecasting (Facebook)
            - `pmdarima` : Auto ARIMA
            - `arch` : GARCH models
            
            **📖 Documentation :**
            - Statsmodels : https://www.statsmodels.org
            - Scikit-learn : https://scikit-learn.org
            - FRED API : https://fred.stlouisfed.org/docs/api/
            """)
        
        with st.expander("⚠️ COMMON PITFALLS", expanded=False):
            st.markdown("""
            **🚨 Erreurs fréquentes à éviter :**
            
            **1. Data Leakage**
            ❌ Utiliser données futures pour prédire passé
            ✅ Strict train/test split chronologique
            
            **2. Ignorer la stationnarité**
            ❌ Régression sur séries non-stationnaires
            ✅ Tester ADF/KPSS, différencier si besoin
            
            **3. P-hacking**
            ❌ Tester 50 variables, garder les 3 significatives
            ✅ Préspécifier hypothèses, correction tests multiples
            
            **4. Overfitting**
            ❌ Modèle parfait sur train, mauvais sur test
            ✅ Cross-validation, régularisation
            
            **5. Confondre corrélation et causalité**
            ❌ "R² = 0.9 donc X cause Y"
            ✅ Corrélation ≠ causalité (3e variable?)
            
            **6. Ignorer les résidus**
            ❌ Regarder que R², ignorer diagnostics
            ✅ Analyser résidus (autocorrélation, normalité)
            
            **7. Extrapolation naïve**
            ❌ Prédire 10 ans avec modèle ARIMA(1,1,1)
            ✅ Modèles structurels pour long-terme
            
            **8. Oublier l'incertitude**
            ❌ "Mon modèle prédit CPI = 315.5"
            ✅ "Prévision 315.5 ± 3.2 (IC 95%)"
            
            **9. Feature selection post-hoc**
            ❌ Regarder importance après coup → biais
            ✅ Domain knowledge + tests a priori
            
            **10. Ignorer structural breaks**
            ❌ Modèle sur 50 ans incluant 2008
            ✅ Tester breaks (Chow, Bai-Perron)
            """)
