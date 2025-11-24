import streamlit as st
import streamlit.components.v1 as components
from openbb import obb
import pandas as pd
from datetime import datetime
import time

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - TradingView & Analytics",
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
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    .stApp {
        background-color: #000000;
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
    
    .stTextInput > div > div > input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    
    .dataframe {
        background-color: #111 !important;
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 10px !important;
    }
    
    .dataframe th {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        border: 1px solid #666 !important;
    }
    
    .dataframe td {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
    }
    
    .news-item {
        background: #111;
        border: 1px solid #333;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #FFAA00;
    }
    
    .news-title {
        color: #00FFFF;
        font-weight: bold;
        font-size: 11px;
        margin-bottom: 5px;
    }
    
    .news-meta {
        color: #666;
        font-size: 9px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - TRADINGVIEW & ANALYTICS</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">ACCUEIL</a>
    </div>
    <div>{current_time} UTC • LIVE DATA</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# SÉLECTION DU TICKER
# =============================================
st.markdown("### 🔍 TICKER SELECTION")

col_ticker, col_button = st.columns([4, 1])

with col_ticker:
    ticker = st.text_input(
        "TICKER",
        value="AAPL",
        placeholder="Ex: AAPL, MSFT, TSLA, BTC-USD...",
        label_visibility="collapsed"
    ).upper()

with col_button:
    analyze_button = st.button("📊 ANALYSER", use_container_width=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# GRAPHIQUE TRADINGVIEW
# =============================================
st.markdown("### 📈 TRADINGVIEW CHART")

# Widget TradingView avec thème sombre
tradingview_widget = f"""
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container" style="height:600px;width:100%">
  <div id="tradingview_chart" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
    "width": "100%",
    "height": "600",
    "symbol": "{ticker}",
    "interval": "D",
    "timezone": "Etc/UTC",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#000000",
    "enable_publishing": false,
    "backgroundColor": "#000000",
    "gridColor": "#333333",
    "hide_top_toolbar": false,
    "hide_legend": false,
    "save_image": false,
    "container_id": "tradingview_chart"
  }}
  );
  </script>
</div>
<!-- TradingView Widget END -->
"""

components.html(tradingview_widget, height=620)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# DONNÉES OPENBB
# =============================================

if analyze_button or ticker:
    try:
        with st.spinner(f'🔄 CHARGEMENT DES DONNÉES POUR {ticker}...'):
            
            # ===== INFORMATIONS GÉNÉRALES =====
            st.markdown("### 📊 COMPANY OVERVIEW")
            
            try:
                # Profile de l'entreprise
                profile = obb.equity.profile(ticker, provider="fmp")
                
                if profile and hasattr(profile, 'results'):
                    data = profile.results[0] if isinstance(profile.results, list) else profile.results
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("COMPANY", data.name if hasattr(data, 'name') else ticker)
                        st.metric("SECTOR", data.sector if hasattr(data, 'sector') else "N/A")
                    
                    with col2:
                        st.metric("INDUSTRY", data.industry if hasattr(data, 'industry') else "N/A")
                        st.metric("COUNTRY", data.country if hasattr(data, 'country') else "N/A")
                    
                    with col3:
                        if hasattr(data, 'market_cap'):
                            market_cap = f"${data.market_cap/1e9:.2f}B" if data.market_cap else "N/A"
                            st.metric("MARKET CAP", market_cap)
                        if hasattr(data, 'employees'):
                            st.metric("EMPLOYEES", f"{data.employees:,}" if data.employees else "N/A")
                    
                    with col4:
                        if hasattr(data, 'website'):
                            st.markdown(f"**WEBSITE:** [{data.website}]({data.website})")
                        if hasattr(data, 'ceo'):
                            st.metric("CEO", data.ceo if data.ceo else "N/A")
                    
                    if hasattr(data, 'description') and data.description:
                        with st.expander("📄 DESCRIPTION"):
                            st.write(data.description)
            
            except Exception as e:
                st.warning(f"⚠️ Profile non disponible: {str(e)}")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # ===== RATIOS FINANCIERS =====
            st.markdown("### 💹 FINANCIAL RATIOS")
            
            try:
                ratios = obb.equity.fundamental.ratios(ticker, provider="fmp", limit=1)
                
                if ratios and hasattr(ratios, 'results') and ratios.results:
                    ratio_data = ratios.results[0] if isinstance(ratios.results, list) else ratios.results
                    
                    col_ratio1, col_ratio2, col_ratio3, col_ratio4 = st.columns(4)
                    
                    with col_ratio1:
                        st.markdown("**VALUATION**")
                        pe = getattr(ratio_data, 'price_to_earnings_ratio', None) or getattr(ratio_data, 'pe_ratio', None)
                        st.metric("P/E RATIO", f"{pe:.2f}" if pe else "N/A")
                        
                        pb = getattr(ratio_data, 'price_to_book_ratio', None) or getattr(ratio_data, 'pb_ratio', None)
                        st.metric("P/B RATIO", f"{pb:.2f}" if pb else "N/A")
                        
                        ps = getattr(ratio_data, 'price_to_sales_ratio', None) or getattr(ratio_data, 'ps_ratio', None)
                        st.metric("P/S RATIO", f"{ps:.2f}" if ps else "N/A")
                    
                    with col_ratio2:
                        st.markdown("**PROFITABILITY**")
                        roe = getattr(ratio_data, 'return_on_equity', None) or getattr(ratio_data, 'roe', None)
                        st.metric("ROE", f"{roe*100:.2f}%" if roe else "N/A")
                        
                        roa = getattr(ratio_data, 'return_on_assets', None) or getattr(ratio_data, 'roa', None)
                        st.metric("ROA", f"{roa*100:.2f}%" if roa else "N/A")
                        
                        margin = getattr(ratio_data, 'net_profit_margin', None) or getattr(ratio_data, 'profit_margin', None)
                        st.metric("NET MARGIN", f"{margin*100:.2f}%" if margin else "N/A")
                    
                    with col_ratio3:
                        st.markdown("**LIQUIDITY**")
                        current = getattr(ratio_data, 'current_ratio', None)
                        st.metric("CURRENT RATIO", f"{current:.2f}" if current else "N/A")
                        
                        quick = getattr(ratio_data, 'quick_ratio', None)
                        st.metric("QUICK RATIO", f"{quick:.2f}" if quick else "N/A")
                        
                        cash = getattr(ratio_data, 'cash_ratio', None)
                        st.metric("CASH RATIO", f"{cash:.2f}" if cash else "N/A")
                    
                    with col_ratio4:
                        st.markdown("**LEVERAGE**")
                        debt_equity = getattr(ratio_data, 'debt_to_equity', None) or getattr(ratio_data, 'debt_equity_ratio', None)
                        st.metric("DEBT/EQUITY", f"{debt_equity:.2f}" if debt_equity else "N/A")
                        
                        debt_assets = getattr(ratio_data, 'debt_to_assets', None)
                        st.metric("DEBT/ASSETS", f"{debt_assets:.2f}" if debt_assets else "N/A")
                        
                        interest = getattr(ratio_data, 'interest_coverage', None)
                        st.metric("INT COVERAGE", f"{interest:.2f}" if interest else "N/A")
                
                else:
                    st.info("📊 Ratios financiers non disponibles pour ce ticker")
            
            except Exception as e:
                st.warning(f"⚠️ Ratios non disponibles: {str(e)}")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # ===== MÉTRIQUES CLÉS =====
            st.markdown("### 📈 KEY METRICS")
            
            try:
                metrics = obb.equity.fundamental.metrics(ticker, provider="fmp", limit=1)
                
                if metrics and hasattr(metrics, 'results') and metrics.results:
                    metric_data = metrics.results[0] if isinstance(metrics.results, list) else metrics.results
                    
                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                    
                    with col_m1:
                        revenue = getattr(metric_data, 'revenue_per_share', None)
                        st.metric("REVENUE/SHARE", f"${revenue:.2f}" if revenue else "N/A")
                    
                    with col_m2:
                        eps = getattr(metric_data, 'net_income_per_share', None) or getattr(metric_data, 'earnings_per_share', None)
                        st.metric("EPS", f"${eps:.2f}" if eps else "N/A")
                    
                    with col_m3:
                        book = getattr(metric_data, 'book_value_per_share', None)
                        st.metric("BOOK VALUE/SHARE", f"${book:.2f}" if book else "N/A")
                    
                    with col_m4:
                        fcf = getattr(metric_data, 'free_cash_flow_per_share', None)
                        st.metric("FCF/SHARE", f"${fcf:.2f}" if fcf else "N/A")
                    
                    with col_m5:
                        div = getattr(metric_data, 'dividend_yield', None)
                        st.metric("DIV YIELD", f"{div*100:.2f}%" if div else "N/A")
            
            except Exception as e:
                st.warning(f"⚠️ Métriques non disponibles: {str(e)}")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # ===== NEWS =====
            st.markdown("### 📰 LATEST NEWS")
            
            try:
                news = obb.news.company(ticker, provider="fmp", limit=10)
                
                if news and hasattr(news, 'results') and news.results:
                    for article in news.results[:10]:
                        title = getattr(article, 'title', 'No title')
                        published = getattr(article, 'published_date', None) or getattr(article, 'date', None)
                        url = getattr(article, 'url', None) or getattr(article, 'link', None)
                        source = getattr(article, 'source', None) or getattr(article, 'site', 'Unknown')
                        
                        st.markdown(f"""
                        <div class="news-item">
                            <div class="news-title">{'📌 ' + title}</div>
                            <div class="news-meta">
                                🕐 {published} | 📡 {source}
                                {f' | <a href="{url}" target="_blank" style="color:#00FFFF;">READ MORE →</a>' if url else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("📰 Aucune news disponible pour ce ticker")
            
            except Exception as e:
                st.warning(f"⚠️ News non disponibles: {str(e)}")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # ===== HISTORIQUE DES PRIX =====
            st.markdown("### 📊 PRICE HISTORY (1 YEAR)")
            
            try:
                historical = obb.equity.price.historical(ticker, provider="fmp", start_date="2024-01-01")
                
                if historical and hasattr(historical, 'results') and historical.results:
                    # Conversion en DataFrame
                    df = pd.DataFrame([vars(x) for x in historical.results])
                    
                    if not df.empty and 'date' in df.columns:
                        df = df.sort_values('date')
                        
                        # Stats rapides
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        if 'close' in df.columns:
                            current_price = df['close'].iloc[-1]
                            year_ago_price = df['close'].iloc[0]
                            change = ((current_price - year_ago_price) / year_ago_price) * 100
                            
                            with col_stat1:
                                st.metric("CURRENT PRICE", f"${current_price:.2f}")
                            with col_stat2:
                                st.metric("1Y CHANGE", f"{change:+.2f}%")
                            with col_stat3:
                                st.metric("52W HIGH", f"${df['close'].max():.2f}")
                            with col_stat4:
                                st.metric("52W LOW", f"${df['close'].min():.2f}")
                        
                        # Afficher le DataFrame
                        st.dataframe(
                            df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(20),
                            use_container_width=True,
                            hide_index=True
                        )
            
            except Exception as e:
                st.warning(f"⚠️ Historique non disponible: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ ERREUR: {str(e)}")
        st.info("💡 Vérifiez que le ticker est valide et que l'API OpenBB est configurée")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# INFO SYSTÈME
# =============================================
col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DONNÉES: TRADINGVIEW + OPENBB (FMP PROVIDER)<br>
        🔄 GRAPHIQUES EN TEMPS RÉEL • DONNÉES FINANCIÈRES ACTUALISÉES
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 DERNIÈRE MAJ: {last_update}<br>
        📍 SYSTÈME OPÉRATIONNEL • PARIS, FRANCE
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | TRADINGVIEW CHARTS • OPENBB ANALYTICS | FINANCIAL DATA PLATFORM<br>
    POWERED BY FMP • REAL-TIME CHARTING • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
