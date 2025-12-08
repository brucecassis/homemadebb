# pages/NEWS.py
# Bloomberg Terminal - News Feed avec Finnhub API + Widgets Finlogix

import streamlit as st
import requests
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import streamlit.components.v1 as components

# =============================================
# CONFIGURATION APIs
# =============================================
FINNHUB_API_KEY = "d14re49r01qop9mf2algd14re49r01qop9mf2am0"
POLYGON_API_KEY = "F95xsJcPxjI5WHlyVfcWWTFy1mK9cfEi"

# =============================================
# AUTO-REFRESH TOUTES LES 60 SECONDES
# =============================================
count = st_autorefresh(interval=60000, limit=None, key="news_refresh")

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - News",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# STYLE BLOOMBERG TERMINAL
# =============================================
st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body, .main, .stApp {
        font-family: 'Courier New', monospace;
        background: #000 !important;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .block-container { padding: 0rem 1rem !important; }
    
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
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        border: 2px solid #FFAA00 !important;
        padding: 6px 12px !important;
        text-transform: uppercase !important;
        border-radius: 0px !important;
        font-size: 10px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00 !important;
        color: #000 !important;
    }
    
    .stTextInput > div > div > input, .stDateInput > div > div > input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 2px solid #FFAA00 !important;
        border-radius: 0 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #111;
        border-bottom: 2px solid #FFAA00;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #222;
        color: #FFAA00;
        border: 1px solid #333;
        border-bottom: none;
        padding: 10px 30px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 12px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFAA00 !important;
        color: #000 !important;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .news-card {
        background: #111;
        border: 1px solid #333;
        border-left: 4px solid #FFAA00;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s;
    }
    
    .news-card:hover {
        border-left-color: #00FF00;
        background: #1a1a1a;
    }
    
    .news-title {
        color: #FFAA00;
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    
    .news-title a {
        color: #FFAA00;
        text-decoration: none;
    }
    
    .news-title a:hover {
        color: #00FF00;
        text-decoration: underline;
    }
    
    .news-meta {
        color: #666;
        font-size: 10px;
        margin-bottom: 5px;
    }
    
    .news-source {
        color: #00FFFF;
        font-weight: bold;
    }
    
    .news-ticker {
        background: #FFAA00;
        color: #000;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: bold;
        margin-right: 8px;
    }
    
    .news-category {
        background: #00FFFF;
        color: #000;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: bold;
        margin-right: 8px;
    }
    
    .news-summary {
        color: #AAA;
        font-size: 11px;
        line-height: 1.5;
        margin-top: 8px;
    }
    
    .category-header {
        background: #FFAA00;
        color: #000;
        padding: 8px 15px;
        font-weight: bold;
        font-size: 12px;
        margin: 20px 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .widget-container {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 20px;
        margin: 20px 0;
        min-height: 600px;
    }
    
    .widget-header {
        background: #FFAA00;
        color: #000;
        padding: 10px 15px;
        font-weight: bold;
        font-size: 14px;
        margin: -20px -20px 20px -20px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    hr { border-color: #333; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# =============================================
# BARRE DE COMMANDE BLOOMBERG
# =============================================

# Style pour la barre de commande
st.markdown("""
<style>
    .command-container {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 10px 15px;
        margin: 10px 0 20px 0;
    }
    .command-prompt {
        color: #FFAA00;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Dictionnaire des commandes et leurs pages
COMMANDS = {
    "EDGAR": "pages/EDGAR.py",
    "NEWS": "pages/NEWS.py",
    "PRICE": "pages/PRICING.py",
    "CHAT": "pages/CHATBOT.py",
    "BT": "pages/BACKTESTING.py",
    "ANA": "pages/COMPANY_ANALYSIS.py",
    "CRYPTO":"pages/CRYPTO_SCRAPER.py",
    "ECO":"pages/ECONOMICS.py", 
    "EU":"pages/EUROPE.py",
    "SIMU":"pages/PORTFOLIO_SIMU.py",
    "PY":"pages/PYTHON_EDITOR.py",
    "SQL":"pages/SQL_EDITOR.py",
    "BONDS":"pages/BONDS.py",
    "HOME":"pages/HOME.py",
}

# Affichage de la barre de commande
st.markdown('<div class="command-container">', unsafe_allow_html=True)

col_prompt, col_input = st.columns([1, 11])

with col_prompt:
    st.markdown('<span class="command-prompt">BBG&gt;</span>', unsafe_allow_html=True)

with col_input:
    command_input = st.text_input(
        "Command",
        placeholder="Tapez une commande: EDGAR, NEWS, CHATBOT, PRICING, HELP...",
        label_visibility="collapsed",
        key="bloomberg_command"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Traitement de la commande
if command_input:
    cmd = command_input.upper().strip()
    
    if cmd == "HELP" or cmd == "H":
        st.info("""
        **📋 COMMANDES DISPONIBLES:**
        - `EDGAR` → SEC Filings & Documents
        - `NEWS` → Market News Feed
        - `CHAT` → AI Assistant
        - `PRICE` → Options Pricing
        - `HELP` → Afficher cette aide
        - `BT` → Backesting de strategies
        - `ANA` → Analyse financière de sociétés côtées
        - `CRYPTO` → Scrapping et backtest de strategies liées aux cryptos
        - `ECO` → Données économiques
        - `EU` → Données Européennes
        - `SIMU` → Simulation de portefeuille
        - `PY` → Editeur de code python 
        - `SQL` → Editeur de code SQL
        - `BONDS` → Screener d'obligation
        - `HOME` → Menu
        """)
    elif cmd in COMMANDS:
        st.switch_page(COMMANDS[cmd])
    else:
        st.warning(f"⚠️ Commande '{cmd}' non reconnue. Tapez HELP pour voir les commandes disponibles.")


# =============================================
# FONCTIONS FINNHUB - NEWS
# =============================================
@st.cache_data(ttl=60)
def get_market_news(category="general"):
    """Récupère les news générales du marché via Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Erreur Finnhub: {e}")
        return []

# =============================================
# FONCTIONS UTILITAIRES
# =============================================
def format_timestamp(timestamp):
    """Convertit un timestamp en date lisible"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return "Date inconnue"

def display_news_card(news_item, ticker="", show_summary=True):
    """Affiche une carte de news style Bloomberg"""
    headline = news_item.get('headline', 'Sans titre')
    url = news_item.get('url', '#')
    source = news_item.get('source', 'Source inconnue')
    timestamp = news_item.get('datetime', 0)
    summary = news_item.get('summary', '')
    category = news_item.get('category', '')
    image = news_item.get('image', '')
    
    # Image si disponible
    img_html = ""
    if image:
        img_html = f'<img src="{image}" style="width:120px;height:80px;object-fit:cover;float:right;margin-left:15px;border:1px solid #333;">'
    
    # Badge ticker ou catégorie
    badge = ""
    if ticker:
        badge = f'<span class="news-ticker">{ticker}</span>'
    elif category:
        badge = f'<span class="news-category">{category.upper()}</span>'
    
    # Summary
    summary_html = ""
    if show_summary and summary:
        short_summary = summary[:200] + "..." if len(summary) > 200 else summary
        summary_html = f'<div class="news-summary">{short_summary}</div>'
    
    st.markdown(f"""
    <div class="news-card">
        {img_html}
        <div>
            {badge}
            <span class="news-meta"><span class="news-source">{source}</span> • {format_timestamp(timestamp)}</span>
        </div>
        <div class="news-title"><a href="{url}" target="_blank">{headline}</a></div>
        {summary_html}
    </div>
    """, unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® | NEWS TERMINAL</div>
        <a href="accueil.html" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">ACCUEIL</a>
    </div>
    <div>{current_time} UTC • FINNHUB + FINLOGIX • AUTO-REFRESH: 60s</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# ONGLETS PRINCIPAUX
# =============================================
tab_global, tab_eco_calendar, tab_earnings = st.tabs([
    "📰 GLOBAL FEED", 
    "📅 ECONOMIC CALENDAR", 
    "💰 EARNINGS CALENDAR"
])

# =============================================
# ONGLET 1 : GLOBAL FEED (INCHANGÉ)
# =============================================
with tab_global:
    st.markdown("### 🌍 GLOBAL MARKET NEWS - FINNHUB")
    
    # Sélection de catégorie
    col_cat, col_info = st.columns([2, 4])
    
    with col_cat:
        category = st.selectbox(
            "Catégorie",
            options=["general", "forex", "crypto", "merger"],
            format_func=lambda x: {
                "general": "📊 GENERAL MARKET",
                "forex": "💱 FOREX",
                "crypto": "₿ CRYPTO",
                "merger": "🤝 MERGERS & ACQUISITIONS"
            }.get(x, x)
        )
    
    with col_info:
        st.markdown(f"""
        <div style="color:#666;font-size:10px;padding:15px 0;">
            📡 NEWS EN TEMPS RÉEL • FINNHUB API • RAFRAÎCHISSEMENT AUTO: 60 SEC
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Récupérer les news
    with st.spinner("📡 Chargement des news..."):
        market_news = get_market_news(category)
    
    if market_news:
        # Stats
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("ARTICLES", len(market_news))
        with col_s2:
            st.metric("CATÉGORIE", category.upper())
        with col_s3:
            st.metric("MAJ", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # Afficher les news
        st.markdown(f'<div class="category-header">🕐 LATEST NEWS - {len(market_news)} ARTICLES</div>', unsafe_allow_html=True)
        
        for news in market_news[:50]:
            display_news_card(news, show_summary=True)
    else:
        st.warning("⚠️ Aucune news disponible pour le moment")

# =============================================
# ONGLET 2 : ECONOMIC CALENDAR (WIDGET TRADINGVIEW)
# =============================================
with tab_eco_calendar:
    st.markdown("### 📅 ECONOMIC CALENDAR - TRADINGVIEW")
    
    st.markdown("""
    <div style="color:#666;font-size:10px;margin:10px 0 20px 0;">
        📊 Calendrier économique mondial en temps réel • Données macroéconomiques • Indicateurs clés
    </div>
    """, unsafe_allow_html=True)
    
    # Widget TradingView Economic Calendar
    economic_calendar_widget = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            html, body {
                background-color: #000 !important;
                height: 100%;
                width: 100%;
                overflow: hidden;
            }
            .tradingview-widget-container {
                width: 100% !important;
                height: 100vh !important;
                background-color: #000 !important;
            }
            #tradingview-widget {
                width: 100% !important;
                height: 100% !important;
            }
        </style>
    </head>
    <body>
        <div style="background:#000;border:2px solid #FFAA00;padding:20px;margin:0;height:100vh;display:flex;flex-direction:column;">
            <div style="background:#FFAA00;color:#000;padding:10px 15px;font-weight:bold;font-size:14px;margin:-20px -20px 20px -20px;text-transform:uppercase;letter-spacing:2px;flex-shrink:0;">
                📊 ECONOMIC CALENDAR WIDGET
            </div>
            
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container" style="flex-grow:1;">
                <div class="tradingview-widget-container__widget" id="tradingview-widget" style="height:100%;width:100%;"></div>
                <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
                {
                  "colorTheme": "dark",
                  "isTransparent": false,
                  "width": "100%",
                  "height": "100%",
                  "locale": "en",
                  "importanceFilter": "-1,0,1",
                  "countryFilter": "ar,au,br,ca,cn,fr,de,in,id,it,jp,kr,mx,ru,sa,za,tr,gb,us,eu"
                }
                </script>
            </div>
            <!-- TradingView Widget END -->
        </div>
    </body>
    </html>
    """
    
    components.html(economic_calendar_widget, height=1000, scrolling=False)
    
    st.markdown("""
    <div style="color:#666;font-size:9px;margin-top:15px;text-align:center;">
        💡 Widget fourni par TradingView • Actualisation automatique • 
        Données économiques mondiales en temps réel
    </div>
    """, unsafe_allow_html=True)

# =============================================
# ONGLET 3 : EARNINGS CALENDAR (WIDGET CUSTOM)
# =============================================
with tab_earnings:
    st.markdown("### 💰 EARNINGS CALENDAR")
    
    st.markdown("""
    <div style="color:#666;font-size:10px;margin:10px 0 20px 0;">
        📈 Calendrier des résultats trimestriels • Estimations vs Résultats réels • Sociétés cotées
    </div>
    """, unsafe_allow_html=True)
    
    # Widget Earnings Calendar - Yahoo Finance Screener Style
    earnings_calendar_widget = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            html, body {
                background-color: #000 !important;
                height: 100%;
                width: 100%;
                overflow: hidden;
            }
            .widget-container {
                width: 100% !important;
                height: 100vh !important;
                background-color: #000 !important;
            }
            iframe {
                width: 100% !important;
                height: 100% !important;
                border: none !important;
            }
        </style>
    </head>
    <body>
        <div style="background:#000;border:2px solid #FFAA00;padding:20px;margin:0;height:100vh;display:flex;flex-direction:column;">
            <div style="background:#FFAA00;color:#000;padding:10px 15px;font-weight:bold;font-size:14px;margin:-20px -20px 20px -20px;text-transform:uppercase;letter-spacing:2px;flex-shrink:0;">
                💰 EARNINGS CALENDAR WIDGET
            </div>
            
            <!-- Earnings Calendar Widget -->
            <div class="widget-container" style="flex-grow:1;">
                <iframe src="https://www.investing.com/earnings-calendar/" 
                        style="width:100%;height:100%;border:none;background:#000;">
                </iframe>
            </div>
            
            <div style="background:#111;color:#666;padding:8px;font-size:9px;text-align:center;flex-shrink:0;margin:-20px;margin-top:10px;">
                💡 Données fournies par Investing.com • Cliquez sur une entreprise pour voir les détails
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(earnings_calendar_widget, height=1000, scrolling=False)
    
    st.markdown("""
    <div style="color:#666;font-size:9px;margin-top:15px;text-align:center;">
        📊 Source: Investing.com • Actualisation automatique • 
        Consultez les résultats passés et à venir avec estimations EPS/Revenue
    </div>
    """, unsafe_allow_html=True)

# =============================================
# SECTION NEWSLETTER
# =============================================

import re

# Configuration Supabase
SUPABASE_URL = "https://eityroxwiryhupmjeqvp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVpdHlyb3h3aXJ5aHVwbWplcXZwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUxMDAzODcsImV4cCI6MjA4MDY3NjM4N30.pIYAmo1K4Y2XkRB4JWGdzsxfOwLdTf7hExNcFkoyzQM"

def is_valid_email(email):
    """Valide le format d'un email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def add_subscriber(email):
    """Ajoute un abonné à Supabase"""
    try:
        # Insérer dans Supabase
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/emails",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json={
                "email": email.lower(),
                "active": True
            }
        )
        
        if response.status_code == 201:
            return True, "Inscription réussie !"
        elif response.status_code == 409:
            return False, "Cet email est déjà inscrit"
        else:
            return False, f"Erreur serveur ({response.status_code})"
            
    except Exception as e:
        return False, f"Erreur: {str(e)}"

# AFFICHAGE DU FORMULAIRE
st.markdown('<hr style="border-color:#333;margin:30px 0;">', unsafe_allow_html=True)

st.markdown("""
<div style="background:#111;border:2px solid #FFAA00;padding:25px;margin:20px 0;">
    <div style="text-align:center;margin-bottom:20px;">
        <div style="color:#FFAA00;font-size:18px;font-weight:bold;letter-spacing:2px;margin-bottom:8px;">
            📧 BLOOMBERG ENS® NEWSLETTER
        </div>
        <div style="color:#AAA;font-size:11px;line-height:1.6;">
            Recevez chaque dimanche un condensé des actualités financières de la semaine<br>
            directement dans votre boîte mail • Format Bloomberg Terminal
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_news1, col_news2, col_news3 = st.columns([1, 2, 1])

with col_news2:
    with st.form("newsletter_subscription", clear_on_submit=True):
        newsletter_email = st.text_input(
            "Email",
            placeholder="votre-email@example.com",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            subscribe_btn = st.form_submit_button("🚀 S'INSCRIRE", use_container_width=True)
        
        if subscribe_btn:
            if newsletter_email:
                if is_valid_email(newsletter_email):
                    success, message = add_subscriber(newsletter_email)
                    if success:
                        st.success(f"✅ {message} Vous recevrez la newsletter chaque dimanche à 20h.")
                    else:
                        st.warning(f"⚠️ {message}")
                else:
                    st.error("❌ Format d'email invalide")
            else:
                st.error("❌ Veuillez entrer une adresse email")

st.markdown("""
<div style="text-align:center;color:#666;font-size:9px;margin-top:15px;">
    📅 Envoi automatique chaque dimanche • 
    📊 Top actualités de la semaine • 
    🔒 Vos données restent privées
</div>
""", unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | FINNHUB + FINLOGIX APIs | SYSTÈME OPÉRATIONNEL<br>
    AUTO-REFRESH: 60 SECONDES • DERNIÈRE MAJ: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
