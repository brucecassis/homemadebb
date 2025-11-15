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
FRED_API_KEY = "ce5dbb3d3fcd8669f2fe2cdd9c79a7da"  # ← METTEZ VOTRE CLÉ ICI

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
    
    .stTextInput input:focus {
        border-color: #FFF;
        box-shadow: 0 0 3px #FFAA00;
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
    
    .indicator-box {
        background-color: #0a0a0a;
        border: 1px solid #333;
        padding: 10px;
        margin: 5px 0;
    }
    
    .warning-box {
        background-color: #1a0a00;
        border-left: 3px solid #FF6600;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #111;
        color: #FFAA00;
        border: 1px solid #333;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFAA00;
        color: #000;
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
        url = f"https://api.stlouisfed.org/fred/series/observations"
        
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
        url = f"https://api.stlouisfed.org/fred/series"
        
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

def calculate_mom_change(df):
    """Calcule la variation Month-over-Month"""
    if df is None or len(df) < 2:
        return None
    
    df = df.copy()
    df['mom_change'] = df['value'].pct_change(1) * 100
    return df

# Dictionnaire des séries économiques importantes
ECONOMIC_SERIES = {
    'Taux d\'intérêt': {
        'FEDFUNDS': 'Fed Funds Rate',
        'DGS10': '10-Year Treasury',
        'DGS2': '2-Year Treasury',
        'DGS30': '30-Year Treasury',
        'SOFR': 'SOFR Rate',
        'MORTGAGE30US': '30Y Mortgage Rate'
    },
    'Inflation': {
        'CPIAUCSL': 'CPI (All Items)',
        'CPILFESL': 'Core CPI (ex food & energy)',
        'PCEPI': 'PCE Price Index',
        'PCEPILFE': 'Core PCE',
        'GASREGW': 'Gas Prices'
    },
    'Emploi': {
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'JTSJOL': 'JOLTS Job Openings',
        'ICSA': 'Initial Claims',
        'CIVPART': 'Labor Force Participation',
        'AHETPI': 'Average Hourly Earnings'
    },
    'Croissance & Production': {
        'GDP': 'GDP',
        'GDPC1': 'Real GDP',
        'INDPRO': 'Industrial Production',
        'UMCSENT': 'Consumer Sentiment',
        'RSXFS': 'Retail Sales',
        'HOUST': 'Housing Starts'
    },
    'Marchés': {
        'SP500': 'S&P 500',
        'VIXCLS': 'VIX',
        'DTWEXBGS': 'Dollar Index',
        'DEXUSEU': 'EUR/USD',
        'BAMLH0A0HYM2': 'High Yield Spread',
        'T10Y2Y': '10Y-2Y Spread'
    },
    'Monétaire': {
        'M2SL': 'M2 Money Supply',
        'WALCL': 'Fed Balance Sheet',
        'TOTRESNS': 'Total Reserves',
        'WSHOMCB': 'Bank Credit'
    }
}

# ===== ONGLETS PRINCIPAUX =====
tab1, tab2, tab3, tab4 = st.tabs(["📊 DASHBOARD", "📈 CUSTOM ANALYSIS", "🔍 SERIES SEARCH", "📥 DOWNLOAD DATA"])

# ===== TAB 1: DASHBOARD =====
with tab1:
    st.markdown("### 📊 ECONOMIC INDICATORS DASHBOARD")
    
    # Bouton refresh
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
    
    # EMPLOI
    st.markdown("#### 👔 EMPLOYMENT")
    cols_employment = st.columns(4)
    
    employment_series = [
        ('UNRATE', 'UNEMPLOYMENT', '%', False),
        ('PAYEMS', 'PAYROLLS', 'K', True),
        ('JTSJOL', 'JOB OPENINGS', 'K', True),
        ('ICSA', 'INIT. CLAIMS', 'K', True)
    ]
    
    for idx, (series_id, label, unit, is_thousands) in enumerate(employment_series):
        with cols_employment[idx]:
            df = get_fred_series(series_id)
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                
                if is_thousands:
                    display_value = f"{current_value/1000:.0f}{unit}"
                    change = (current_value - previous_value) / 1000
                    delta_display = f"{change:+.0f}{unit}"
                else:
                    display_value = f"{current_value:.1f}{unit}"
                    change = current_value - previous_value
                    delta_display = f"{change:+.1f}{unit}"
                
                st.metric(
                    label=label,
                    value=display_value,
                    delta=delta_display
                )
            else:
                st.metric(label=label, value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # YIELD CURVE & RECESSION INDICATOR
    st.markdown("#### 📉 YIELD CURVE & RECESSION INDICATORS")
    
    col_yc1, col_yc2 = st.columns(2)
    
    with col_yc1:
        # Graphique Yield Curve
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
                height=350
            )
            
            st.plotly_chart(fig_curve, use_container_width=True)
    
    with col_yc2:
        # Indicateurs de récession
        df_10y2y = get_fred_series('T10Y2Y')
        
        if df_10y2y is not None and len(df_10y2y) > 0:
            current_spread = df_10y2y['value'].iloc[-1]
            
            if current_spread < 0:
                st.markdown(f"""
                <div class="warning-box">
                    <h4 style="color: #FF6600; margin: 0;">⚠️ YIELD CURVE INVERTED</h4>
                    <p style="margin: 5px 0;">10Y-2Y Spread: <strong>{current_spread:.2f}%</strong></p>
                    <p style="margin: 5px 0; font-size: 10px;">Indicateur historique de récession</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="indicator-box">
                    <h4 style="color: #00FF00; margin: 0;">✅ YIELD CURVE NORMAL</h4>
                    <p style="margin: 5px 0;">10Y-2Y Spread: <strong>{current_spread:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Graphique historique du spread
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=df_10y2y['date'],
                y=df_10y2y['value'],
                mode='lines',
                line=dict(color='#FFAA00', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 170, 0, 0.1)'
            ))
            
            fig_spread.add_hline(y=0, line_dash="dash", line_color="#FF0000", line_width=1)
            
            fig_spread.update_layout(
                title="10Y-2Y Spread Historical",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333'),
                yaxis=dict(title="Spread (%)", gridcolor='#333'),
                height=250
            )
            
            st.plotly_chart(fig_spread, use_container_width=True)

# ===== TAB 2: ANALYSE PERSONNALISÉE =====
with tab2:
    st.markdown("### 📈 CUSTOM ECONOMIC ANALYSIS")
    
    col_analysis1, col_analysis2 = st.columns([2, 1])
    
    with col_analysis1:
        # Sélection de séries à comparer
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
            options=['1Y', '2Y', '5Y', '10Y', '20Y', 'MAX'],
            index=2,
            key="lookback_select"
        )
        
        chart_type = st.selectbox(
            "CHART TYPE",
            options=['Absolute Values', 'YoY Change (%)', 'Normalized (Base 100)'],
            key="chart_type_select"
        )
        
        show_correlation = st.checkbox("Show Correlation Matrix", value=False)
    
    if st.button("📊 GENERATE ANALYSIS", use_container_width=True, key="generate_analysis"):
        if selected_series:
            # Calculer la date de début
            if lookback_period == 'MAX':
                start_date = None
            else:
                years = int(lookback_period[:-1])
                start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
            
            # Récupérer les données
            series_data = {}
            for series_id in selected_series:
                df = get_fred_series(series_id, observation_start=start_date)
                if df is not None:
                    series_data[series_id] = df
            
            if series_data:
                # Créer le graphique
                fig = go.Figure()
                
                for series_id, df in series_data.items():
                    if chart_type == 'YoY Change (%)':
                        df = calculate_yoy_change(df)
                        y_data = df['yoy_change']
                        y_label = "YoY Change (%)"
                    elif chart_type == 'Normalized (Base 100)':
                        y_data = (df['value'] / df['value'].iloc[0]) * 100
                        y_label = "Normalized (Base 100)"
                    else:
                        y_data = df['value']
                        y_label = "Value"
                    
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=y_data,
                        mode='lines',
                        name=f"{series_id} - {series_in_category.get(series_id, series_id)}",
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"{selected_category} - {chart_type}",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title=y_label),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Matrice de corrélation
                if show_correlation and len(series_data) > 1:
                    st.markdown("#### 📊 CORRELATION MATRIX")
                    
                    # Créer un DataFrame avec toutes les séries
                    corr_df = pd.DataFrame()
                    for series_id, df in series_data.items():
                        corr_df[series_id] = df.set_index('date')['value']
                    
                    # Calculer la corrélation
                    corr_matrix = corr_df.corr()
                    
                    # Créer la heatmap
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=[f"{sid}" for sid in corr_matrix.columns],
                        y=[f"{sid}" for sid in corr_matrix.columns],
                        colorscale=[
                            [0, '#FF0000'],
                            [0.5, '#000000'],
                            [1, '#00FF00']
                        ],
                        zmid=0,
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10, "color": "#FFAA00"}
                    ))
                    
                    fig_corr.update_layout(
                        title="Correlation Matrix",
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        height=400
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Statistiques
                st.markdown("#### 📊 STATISTICS")
                
                stats_data = []
                for series_id, df in series_data.items():
                    stats_data.append({
                        'Series': f"{series_id} - {series_in_category.get(series_id, series_id)}",
                        'Current': f"{df['value'].iloc[-1]:.2f}",
                        'Mean': f"{df['value'].mean():.2f}",
                        'Std Dev': f"{df['value'].std():.2f}",
                        'Min': f"{df['value'].min():.2f}",
                        'Max': f"{df['value'].max():.2f}",
                        'Latest Date': df['date'].iloc[-1].strftime('%Y-%m-%d')
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            else:
                st.warning("⚠️ No data available for selected series")
        else:
            st.warning("⚠️ Please select at least one series")

# ===== TAB 3: RECHERCHE DE SÉRIES =====
# ===== TAB 3: RECHERCHE DE SÉRIES =====
with tab3:
    st.markdown("### 🔍 SEARCH FRED SERIES")
    
    st.markdown("#### 📚 AVAILABLE SERIES BY CATEGORY")
    
    for category, series in ECONOMIC_SERIES.items():
        with st.expander(f"📊 {category}", expanded=False):
            for series_id, description in series.items():
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.markdown(f"**`{series_id}`**")
                
                with col2:
                    st.markdown(f"{description}")
                
                with col3:
                    if st.button("📊 VIEW", key=f"view_{series_id}", use_container_width=True):
                        df = get_fred_series(series_id)
                        info = get_fred_series_info(series_id)
                        
                        if df is not None:
                            st.markdown(f"### {series_id} - {description}")
                            
                            if info:
                                st.caption(f"**Title:** {info.get('title', 'N/A')}")
                                st.caption(f"**Units:** {info.get('units', 'N/A')}")
                                st.caption(f"**Frequency:** {info.get('frequency', 'N/A')}")
                            
                            # Graphique
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df['date'],
                                y=df['value'],
                                mode='lines',
                                line=dict(color='#FFAA00', width=2)
                            ))
                            
                            fig.update_layout(
                                paper_bgcolor='#000',
                                plot_bgcolor='#111',
                                font=dict(color='#FFAA00', size=10),
                                xaxis=dict(gridcolor='#333'),
                                yaxis=dict(gridcolor='#333'),
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Stats
                            st.markdown("**Latest Values:**")
                            st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
    
    # Recherche personnalisée
    st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔍 CUSTOM SERIES SEARCH")
    
    custom_series_id = st.text_input(
        "Enter FRED Series ID",
        placeholder="Ex: GDP, UNRATE, CPIAUCSL...",
        key="custom_series_input"
    ).upper()
    
    if st.button("🔍 SEARCH CUSTOM SERIES", use_container_width=True, key="search_custom"):
        if custom_series_id:
            df = get_fred_series(custom_series_id)
            info = get_fred_series_info(custom_series_id)
            
            if df is not None:
                st.success(f"✅ Series found: {custom_series_id}")
                
                if info:
                    st.markdown(f"**Title:** {info.get('title', 'N/A')}")
                    st.markdown(f"**Units:** {info.get('units', 'N/A')}")
                    st.markdown(f"**Frequency:** {info.get('frequency', 'N/A')}")
                    st.markdown(f"**Seasonal Adjustment:** {info.get('seasonal_adjustment', 'N/A')}")
                
                # Graphique
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['value'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 170, 0, 0.1)'
                ))
                
                fig.update_layout(
                    title=f"{custom_series_id} - Historical Data",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title="Value"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Current Value", f"{df['value'].iloc[-1]:.2f}")
                
                with col_stat2:
                    st.metric("Mean", f"{df['value'].mean():.2f}")
                
                with col_stat3:
                    st.metric("Min", f"{df['value'].min():.2f}")
                
                with col_stat4:
                    st.metric("Max", f"{df['value'].max():.2f}")
                
                # Dernières valeurs
                st.markdown("**Latest 20 Observations:**")
                st.dataframe(df.tail(20).sort_values('date', ascending=False), use_container_width=True, hide_index=True)
            
            else:
                st.error(f"❌ Series '{custom_series_id}' not found. Check the series ID on FRED website.")
        else:
            st.warning("⚠️ Please enter a series ID")
st.markdown(f"**Frequency:** {info.get('frequency', 'N/A')}")
                    st.markdown(f"**Seasonal Adjustment:** {info.get('seasonal_adjustment', 'N/A')}")
                
                # Graphique
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['value'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 170, 0, 0.1)'
                ))
                
                fig.update_layout(
                    title=f"{custom_series_id} - Historical Data",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title="Value"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Current Value", f"{df['value'].iloc[-1]:.2f}")
                
                with col_stat2:
                    st.metric("Mean", f"{df['value'].mean():.2f}")
                
                with col_stat3:
                    st.metric("Min", f"{df['value'].min():.2f}")
                
                with col_stat4:
                    st.metric("Max", f"{df['value'].max():.2f}")
                
                # Dernières valeurs
                st.markdown("**Latest 20 Observations:**")
                st.dataframe(df.tail(20).sort_values('date', ascending=False), use_container_width=True, hide_index=True)
            
            else:
                st.error(f"❌ Series '{custom_series_id}' not found. Check the series ID on FRED website.")
        else:
            st.warning("⚠️ Please enter a series ID")

# ===== TAB 4: TÉLÉCHARGEMENT DE DONNÉES =====
with tab4:
    st.markdown("### 📥 DOWNLOAD ECONOMIC DATA")
    
    st.markdown("#### 📊 BULK DOWNLOAD")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
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
    
    with col_dl2:
        download_period = st.selectbox(
            "TIME PERIOD",
            options=['1Y', '2Y', '5Y', '10Y', '20Y', 'MAX'],
            index=3,
            key="download_period"
        )
        
        download_format = st.selectbox(
            "FILE FORMAT",
            options=['CSV', 'Excel', 'JSON'],
            key="download_format"
        )
    
    if st.button("📥 DOWNLOAD DATA", use_container_width=True, key="download_button"):
        if download_series:
            with st.spinner('Downloading data...'):
                # Calculer la date de début
                if download_period == 'MAX':
                    start_date = None
                else:
                    years = int(download_period[:-1])
                    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
                
                # Récupérer toutes les données
                all_data = {}
                for series_id in download_series:
                    df = get_fred_series(series_id, observation_start=start_date)
                    if df is not None:
                        all_data[series_id] = df
                
                if all_data:
                    # Créer un DataFrame combiné
                    combined_df = pd.DataFrame()
                    
                    for series_id, df in all_data.items():
                        df_temp = df[['date', 'value']].copy()
                        df_temp = df_temp.rename(columns={'value': series_id})
                        
                        if combined_df.empty:
                            combined_df = df_temp
                        else:
                            combined_df = pd.merge(combined_df, df_temp, on='date', how='outer')
                    
                    combined_df = combined_df.sort_values('date')
                    
                    # Téléchargement selon le format
                    if download_format == 'CSV':
                        csv_data = combined_df.to_csv(index=False)
                        st.download_button(
                            label="💾 DOWNLOAD CSV",
                            data=csv_data,
                            file_name=f"economic_data_{download_category}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    elif download_format == 'Excel':
                        # Créer un fichier Excel en mémoire
                        from io import BytesIO
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            combined_df.to_excel(writer, sheet_name='Data', index=False)
                            
                            # Ajouter une feuille avec les métadonnées
                            metadata = []
                            for series_id in download_series:
                                info = get_fred_series_info(series_id)
                                if info:
                                    metadata.append({
                                        'Series ID': series_id,
                                        'Title': info.get('title', 'N/A'),
                                        'Units': info.get('units', 'N/A'),
                                        'Frequency': info.get('frequency', 'N/A')
                                    })
                            
                            if metadata:
                                metadata_df = pd.DataFrame(metadata)
                                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                        
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="💾 DOWNLOAD EXCEL",
                            data=excel_data,
                            file_name=f"economic_data_{download_category}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    elif download_format == 'JSON':
                        json_data = combined_df.to_json(orient='records', date_format='iso')
                        st.download_button(
                            label="💾 DOWNLOAD JSON",
                            data=json_data,
                            file_name=f"economic_data_{download_category}_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    st.success(f"✅ Data prepared! {len(combined_df)} observations downloaded.")
                    
                    # Prévisualisation
                    st.markdown("**Data Preview:**")
                    st.dataframe(combined_df.head(20), use_container_width=True, hide_index=True)
                
                else:
                    st.error("❌ No data could be retrieved")
        else:
            st.warning("⚠️ Please select at least one series")
    
    st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # Section téléchargement personnalisé
    st.markdown("#### 📊 CUSTOM SERIES DOWNLOAD")
    
    custom_download_ids = st.text_area(
        "Enter FRED Series IDs (one per line or comma-separated)",
        placeholder="GDP\nUNRATE\nCPIAUCSL\n\nor\n\nGDP, UNRATE, CPIAUCSL",
        height=100,
        key="custom_download_ids"
    )
    
    col_custom1, col_custom2 = st.columns(2)
    
    with col_custom1:
        custom_period = st.selectbox(
            "TIME PERIOD",
            options=['1Y', '2Y', '5Y', '10Y', '20Y', 'MAX'],
            index=2,
            key="custom_period"
        )
    
    with col_custom2:
        custom_format = st.selectbox(
            "FILE FORMAT",
            options=['CSV', 'Excel', 'JSON'],
            key="custom_format"
        )
    
    if st.button("📥 DOWNLOAD CUSTOM DATA", use_container_width=True, key="download_custom_button"):
        if custom_download_ids:
            # Parser les IDs
            ids_list = []
            for line in custom_download_ids.split('\n'):
                for item in line.split(','):
                    item = item.strip().upper()
                    if item:
                        ids_list.append(item)
            
            if ids_list:
                with st.spinner(f'Downloading {len(ids_list)} series...'):
                    # Calculer la date de début
                    if custom_period == 'MAX':
                        start_date = None
                    else:
                        years = int(custom_period[:-1])
                        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
                    
                    # Récupérer les données
                    all_data = {}
                    failed_series = []
                    
                    for series_id in ids_list:
                        df = get_fred_series(series_id, observation_start=start_date)
                        if df is not None:
                            all_data[series_id] = df
                        else:
                            failed_series.append(series_id)
                    
                    if all_data:
                        # Créer un DataFrame combiné
                        combined_df = pd.DataFrame()
                        
                        for series_id, df in all_data.items():
                            df_temp = df[['date', 'value']].copy()
                            df_temp = df_temp.rename(columns={'value': series_id})
                            
                            if combined_df.empty:
                                combined_df = df_temp
                            else:
                                combined_df = pd.merge(combined_df, df_temp, on='date', how='outer')
                        
                        combined_df = combined_df.sort_values('date')
                        
                        # Téléchargement selon le format
                        if custom_format == 'CSV':
                            csv_data = combined_df.to_csv(index=False)
                            st.download_button(
                                label="💾 DOWNLOAD CSV",
                                data=csv_data,
                                file_name=f"custom_economic_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        elif custom_format == 'Excel':
                            from io import BytesIO
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                combined_df.to_excel(writer, sheet_name='Data', index=False)
                            
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="💾 DOWNLOAD EXCEL",
                                data=excel_data,
                                file_name=f"custom_economic_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        elif custom_format == 'JSON':
                            json_data = combined_df.to_json(orient='records', date_format='iso')
                            st.download_button(
                                label="💾 DOWNLOAD JSON",
                                data=json_data,
                                file_name=f"custom_economic_data_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        st.success(f"✅ {len(all_data)} series downloaded successfully! {len(combined_df)} observations.")
                        
                        if failed_series:
                            st.warning(f"⚠️ Failed to download: {', '.join(failed_series)}")
                        
                        # Prévisualisation
                        st.markdown("**Data Preview:**")
                        st.dataframe(combined_df.head(20), use_container_width=True, hide_index=True)
                    
                    else:
                        st.error("❌ No data could be retrieved. Check your series IDs.")
            else:
                st.warning("⚠️ No valid series IDs found")
        else:
            st.warning("⚠️ Please enter at least one series ID")

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | FEDERAL RESERVE ECONOMIC DATA (FRED) | LAST UPDATE: {last_update}
    <br>
    Data provided by Federal Reserve Bank of St. Louis
</div>
""", unsafe_allow_html=True)
