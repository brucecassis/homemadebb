import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import requests

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
            
            # Récupération des données yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # ===== INFORMATIONS GÉNÉRALES =====
            st.markdown("### 📊 COMPANY OVERVIEW")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("COMPANY", info.get('longName', ticker))
                st.metric("SECTOR", info.get('sector', 'N/A'))
            
            with col2:
                st.metric("INDUSTRY", info.get('industry', 'N/A'))
                st.metric("COUNTRY", info.get('country', 'N/A'))
            
            with col3:
                market_cap = info.get('marketCap')
                if market_cap:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                else:
                    market_cap_str = "N/A"
                st.metric("MARKET CAP", market_cap_str)
                
                employees = info.get('fullTimeEmployees')
                st.metric("EMPLOYEES", f"{employees:,}" if employees else "N/A")
            
            with col4:
                website = info.get('website')
                if website:
                    st.markdown(f"**WEBSITE:** [{website}]({website})")
                
                ceo = info.get('companyOfficers')
                if ceo and len(ceo) > 0:
                    st.metric("CEO", ceo[0].get('name', 'N/A'))
                else:
                    st.metric("CEO", "N/A")
            
            description = info.get('longBusinessSummary')
            if description:
                with st.expander("📄 DESCRIPTION"):
                    st.write(description)
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # ===== RATIOS FINANCIERS =====
            st.markdown("### 💹 FINANCIAL RATIOS")
            
            col_ratio1, col_ratio2, col_ratio3, col_ratio4 = st.columns(4)
            
            with col_ratio1:
                st.markdown("**VALUATION**")
                pe = info.get('trailingPE') or info.get('forwardPE')
                st.metric("P/E RATIO", f"{pe:.2f}" if pe else "N/A")
                
                pb = info.get('priceToBook')
                st.metric("P/B RATIO", f"{pb:.2f}" if pb else "N/A")
                
                ps = info.get('priceToSalesTrailing12Months')
                st.metric("P/S RATIO", f"{ps:.2f}" if ps else "N/A")
            
            with col_ratio2:
                st.markdown("**PROFITABILITY**")
                roe = info.get('returnOnEquity')
                st.metric("ROE", f"{roe*100:.2f}%" if roe else "N/A")
                
                roa = info.get('returnOnAssets')
                st.metric("ROA", f"{roa*100:.2f}%" if roa else "N/A")
                
                margin = info.get('profitMargins')
                st.metric("NET MARGIN", f"{margin*100:.2f}%" if margin else "N/A")
            
            with col_ratio3:
                st.markdown("**LIQUIDITY**")
                current = info.get('currentRatio')
                st.metric("CURRENT RATIO", f"{current:.2f}" if current else "N/A")
                
                quick = info.get('quickRatio')
                st.metric("QUICK RATIO", f"{quick:.2f}" if quick else "N/A")
                
                cash = info.get('cashRatio')
                st.metric("CASH RATIO", f"{cash:.2f}" if cash else "N/A")
            
            with col_ratio4:
                st.markdown("**LEVERAGE**")
                debt_equity = info.get('debtToEquity')
                st.metric("DEBT/EQUITY", f"{debt_equity:.2f}" if debt_equity else "N/A")
                
                total_debt = info.get('totalDebt')
                total_assets = info.get('totalAssets')
                if total_debt and total_assets and total_assets > 0:
                    debt_assets = total_debt / total_assets
                    st.metric("DEBT/ASSETS", f"{debt_assets:.2f}")
                else:
                    st.metric("DEBT/ASSETS", "N/A")
                
                interest = info.get('interestCoverage')
                st.metric("INT COVERAGE", f"{interest:.2f}" if interest else "N/A")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # ===== MÉTRIQUES CLÉS =====
            st.markdown("### 📈 KEY METRICS")
            
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            with col_m1:
                revenue_per_share = info.get('revenuePerShare')
                st.metric("REVENUE/SHARE", f"${revenue_per_share:.2f}" if revenue_per_share else "N/A")
            
            with col_m2:
                eps = info.get('trailingEps')
                st.metric("EPS", f"${eps:.2f}" if eps else "N/A")
            
            with col_m3:
                book_value = info.get('bookValue')
                st.metric("BOOK VALUE/SHARE", f"${book_value:.2f}" if book_value else "N/A")
            
            with col_m4:
                fcf = info.get('freeCashflow')
                shares = info.get('sharesOutstanding')
                if fcf and shares and shares > 0:
                    fcf_per_share = fcf / shares
                    st.metric("FCF/SHARE", f"${fcf_per_share:.2f}")
                else:
                    st.metric("FCF/SHARE", "N/A")
            
            with col_m5:
                div_yield = info.get('dividendYield')
                st.metric("DIV YIELD", f"{div_yield*100:.2f}%" if div_yield else "N/A")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # ===== NEWS =====
            st.markdown("### 📰 LATEST NEWS")
            
            try:
                news = stock.news
                
                if news and len(news) > 0:
                    for article in news[:10]:
                        title = article.get('title', 'No title')
                        published = article.get('providerPublishTime')
                        if published:
                            published = datetime.fromtimestamp(published).strftime('%Y-%m-%d %H:%M')
                        url = article.get('link')
                        source = article.get('publisher', 'Unknown')
                        
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
                df = stock.history(period="1y")
                
                if not df.empty:
                    # Stats rapides
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    current_price = df['Close'].iloc[-1]
                    year_ago_price = df['Close'].iloc[0]
                    change = ((current_price - year_ago_price) / year_ago_price) * 100
                    
                    with col_stat1:
                        st.metric("CURRENT PRICE", f"${current_price:.2f}")
                    with col_stat2:
                        st.metric("1Y CHANGE", f"{change:+.2f}%")
                    with col_stat3:
                        st.metric("52W HIGH", f"${df['Close'].max():.2f}")
                    with col_stat4:
                        st.metric("52W LOW", f"${df['Close'].min():.2f}")
                    
                    # Préparer le DataFrame pour affichage
                    df_display = df.reset_index()
                    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
                    df_display = df_display.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    # Afficher le DataFrame
                    st.dataframe(
                        df_display[['date', 'open', 'high', 'low', 'close', 'volume']].tail(20),
                        use_container_width=True,
                        hide_index=True
                    )
            
            except Exception as e:
                st.warning(f"⚠️ Historique non disponible: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ ERREUR: {str(e)}")
        st.info("💡 Vérifiez que le ticker est valide")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# INFO SYSTÈME
# =============================================
col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DONNÉES: TRADINGVIEW + YAHOO FINANCE<br>
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
    © 2025 BLOOMBERG ENS® | TRADINGVIEW CHARTS • YAHOO FINANCE ANALYTICS | FINANCIAL DATA PLATFORM<br>
    POWERED BY YFINANCE • REAL-TIME CHARTING • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
