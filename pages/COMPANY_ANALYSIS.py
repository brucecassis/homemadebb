import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from groq import Groq
import json
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(
    page_title="Company Analysis",
    page_icon="🏢",
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
    
    .stTextInput input {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .company-header {
        background-color: #111;
        border: 2px solid #FFAA00;
        padding: 15px;
        margin: 10px 0;
    }
    
    .info-box {
        background-color: #0a0a0a;
        border-left: 3px solid #00FF00;
        padding: 10px;
        margin: 5px 0;
    }
    
    .warning-box {
        background-color: #1a0a00;
        border-left: 3px solid #FF6600;
        padding: 10px;
        margin: 5px 0;
    }
    
    .section-divider {
        border-bottom: 2px solid #FFAA00;
        margin: 20px 0;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .dataframe {
        font-family: 'Courier New', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - COMPANY ANALYSIS</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# Fonctions pour récupérer les données
@st.cache_data(ttl=300)
@st.cache_resource(ttl=300)
def get_company_info(ticker):
    """Récupère les informations de l'entreprise via yfinance"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return stock, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None


@st.cache_resource(ttl=300)
def get_financial_statements(ticker):
    """Récupère les états financiers"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        return {
            'income_stmt': stock.income_stmt,
            'balance_sheet': stock.balance_sheet,
            'cash_flow': stock.cashflow,
            'quarterly_income': stock.quarterly_income_stmt,
            'quarterly_balance': stock.quarterly_balance_sheet,
            'quarterly_cashflow': stock.quarterly_cashflow
        }
    except Exception as e:
        st.error(f"Error fetching financials: {e}")
        return None

@st.cache_data(ttl=300)
def get_price_history(ticker, period='1y'):
    """Récupère l'historique des prix"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching price history: {e}")
        return None

def analyze_financials_with_ai(data_type, dataframe, company_name, ticker):
    """Analyse les données financières avec Groq AI"""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            return "❌ Error: GROQ_API_KEY not found in secrets."
        
        client = Groq(api_key=api_key)
        df_recent = dataframe.iloc[:, :3] if len(dataframe.columns) >= 3 else dataframe
        data_str = df_recent.to_string()
        
        prompts = {
            "Income Statement": f"""Analyze this Income Statement for {company_name} ({ticker}).
Data: {data_str}
Cover: Revenue Trends, Profitability, Strengths, Concerns, Recommendations.""",
            "Balance Sheet": f"""Analyze this Balance Sheet for {company_name} ({ticker}).
Data: {data_str}
Cover: Assets, Liabilities, Ratios, Liquidity, Risks, Recommendations.""",
            "Cash Flow": f"""Analyze this Cash Flow for {company_name} ({ticker}).
Data: {data_str}
Cover: Operating CF, Free CF, Investments, Financing, Recommendations."""
        }
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Expert financial analyst. Be concise."},
                {"role": "user", "content": prompts.get(data_type, prompts["Income Statement"])}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_tradingview_symbol(ticker):
    """Convertit un ticker Yahoo Finance en symbole TradingView"""
    exchange_mapping = {
        '.PA': 'EURONEXT:', '.DE': 'XETR:', '.SW': 'SIX:', '.L': 'LSE:',
        '.AS': 'EURONEXT:', '.BR': 'EURONEXT:', '.MI': 'MIL:', '.MC': 'BME:',
        '.TO': 'TSX:', '.HK': 'HKEX:', '.T': 'TSE:', '.AX': 'ASX:',
    }
    for suffix, exchange in exchange_mapping.items():
        if ticker.endswith(suffix):
            return exchange + ticker.replace(suffix, '')
    return ticker


def render_tradingview_chart(symbol, height=450):
    """Render TradingView chart - Bloomberg style"""
    return f'''
    <div class="tradingview-widget-container" style="height:{height}px;width:100%;">
      <div id="tv_chart" style="height:100%;width:100%;"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "D",
        "timezone": "Europe/Paris",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#000000",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": false,
        "container_id": "tv_chart",
        "studies": ["STD;SMA"],
        "overrides": {{
            "paneProperties.background": "#000000",
            "paneProperties.backgroundType": "solid",
            "paneProperties.vertGridProperties.color": "#262626",
            "paneProperties.horzGridProperties.color": "#262626",
            "paneProperties.crossHairProperties.color": "#FFAA00",
            "scalesProperties.textColor": "#FFAA00",
            "scalesProperties.lineColor": "#FFAA00",
            "mainSeriesProperties.candleStyle.upColor": "#00FF00",
            "mainSeriesProperties.candleStyle.downColor": "#FF0000",
            "mainSeriesProperties.candleStyle.wickUpColor": "#00FF00",
            "mainSeriesProperties.candleStyle.wickDownColor": "#FF0000",
            "mainSeriesProperties.candleStyle.borderUpColor": "#00FF00",
            "mainSeriesProperties.candleStyle.borderDownColor": "#FF0000",
            "mainSeriesProperties.candleStyle.drawBorder": true,
            "volumePaneSize": "medium"
        }}
      }});
      </script>
    </div>
    '''


# ===== BARRE DE RECHERCHE =====
st.markdown("### 🔍 COMPANY SEARCH")

col_search1, col_search2 = st.columns([4, 1])

with col_search1:
    ticker_input = st.text_input(
        "", placeholder="Enter ticker (AAPL, MSFT, MC.PA...)",
        key="ticker_search", label_visibility="collapsed",
        on_change=lambda: st.session_state.update({'current_ticker': st.session_state.ticker_search.upper()})
    ).upper()

with col_search2:
    search_button = st.button("🔍 ANALYZE", use_container_width=True)

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)

if search_button and ticker_input:
    st.session_state['current_ticker'] = ticker_input

display_ticker = st.session_state.get('current_ticker', None)

# ===== AFFICHAGE DES DONNÉES =====
if display_ticker:
    with st.spinner(f'🔍 Analyzing {display_ticker}...'):
        stock, info = get_company_info(display_ticker)
        
        if stock and info:
            company_name = info.get('longName', display_ticker)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            website = info.get('website', 'N/A')
            
            st.markdown(f"""
            <div class="company-header">
                <h1 style="font-size: 24px; margin: 0; color: #00FF00;">{company_name} ({display_ticker})</h1>
                <p style="margin: 5px 0; font-size: 12px; color: #FFAA00;">
                    <strong>Sector:</strong> {sector} | <strong>Industry:</strong> {industry}
                </p>
                <p style="margin: 5px 0; font-size: 10px; color: #999;">
                    <strong>Website:</strong> {website}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 OVERVIEW", "💰 FINANCIALS","🔬 ADVANCED ANALYSIS", "📊 VALUATION", "🔍 DETAILED INFO"
            ])
            
            # ===== TAB 1: OVERVIEW =====
            with tab1:
                st.markdown("### 📊 COMPANY OVERVIEW")
                
                col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
                
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                previous_close = info.get('previousClose', 0)
                change = current_price - previous_close if current_price and previous_close else 0
                change_pct = (change / previous_close * 100) if previous_close else 0
                
                with col_ov1:
                    st.metric("CURRENT PRICE", f"${current_price:.2f}" if current_price else "N/A",
                              delta=f"{change:+.2f} ({change_pct:+.2f}%)")
                
                with col_ov2:
                    market_cap = info.get('marketCap', 0)
                    if market_cap:
                        cap_display = f"${market_cap/1e12:.2f}T" if market_cap >= 1e12 else f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
                    else:
                        cap_display = "N/A"
                    st.metric("MARKET CAP", cap_display)
                
                with col_ov3:
                    pe_ratio = info.get('trailingPE', 0)
                    st.metric("P/E RATIO", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                
                with col_ov4:
                    dividend_yield = info.get('dividendYield', 0)
                    if dividend_yield:
                        st.metric("DIVIDEND YIELD", f"{dividend_yield*100:.2f}%" if dividend_yield < 1 else f"{dividend_yield:.2f}%")
                    else:
                        st.metric("DIVIDEND YIELD", "N/A")
                
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # ===== TRADINGVIEW CHART =====
                st.markdown("#### 📈 PRICE CHART")
                tv_symbol = get_tradingview_symbol(display_ticker)
                components.html(render_tradingview_chart(tv_symbol, height=450), height=470)
                
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # Trading Information
                st.markdown("#### 📊 TRADING INFORMATION")
                col_trade1, col_trade2, col_trade3, col_trade4 = st.columns(4)
                
                with col_trade1:
                    volume = info.get('volume', 0)
                    st.metric("VOLUME", f"{volume:,.0f}" if volume else "N/A")
                
                with col_trade2:
                    day_high, day_low = info.get('dayHigh', 0), info.get('dayLow', 0)
                    st.metric("DAY RANGE", f"${day_low:.2f} - ${day_high:.2f}" if day_low and day_high else "N/A")
                
                with col_trade3:
                    year_high, year_low = info.get('fiftyTwoWeekHigh', 0), info.get('fiftyTwoWeekLow', 0)
                    st.metric("52W RANGE", f"${year_low:.2f} - ${year_high:.2f}" if year_low and year_high else "N/A")
                
                with col_trade4:
                    beta = info.get('beta', 0)
                    st.metric("BETA", f"{beta:.2f}" if beta else "N/A")
                
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # Description
                st.markdown("#### 📝 COMPANY DESCRIPTION")
                description = info.get('longBusinessSummary', 'No description available.')
                st.markdown(f"""
                <div style="background-color: #111; padding: 15px; border-left: 3px solid #FFAA00;">
                    <p style="font-size: 11px; color: #999; line-height: 1.6;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Executives
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 👥 KEY EXECUTIVES")
                officers = info.get('companyOfficers', [])
                if officers:
                    exec_data = []
                    for officer in officers[:5]:
                        total_pay = officer.get('totalPay', None)
                        pay_display = f"${total_pay:,.0f}" if isinstance(total_pay, (int, float)) and total_pay else 'N/A'
                        exec_data.append({'Name': officer.get('name', 'N/A'), 'Title': officer.get('title', 'N/A'), 'Pay': pay_display})
                    st.dataframe(pd.DataFrame(exec_data), use_container_width=True, hide_index=True)
                else:
                    st.info("Executive information not available")
            
            # ===== TAB 2: FINANCIALS =====
            with tab2:
                st.markdown("### 💰 FINANCIAL STATEMENTS")
                financials = get_financial_statements(display_ticker)
                
                if financials:
                    fin_tab1, fin_tab2, fin_tab3 = st.tabs(["📊 INCOME STATEMENT", "📊 BALANCE SHEET", "📊 CASH FLOW"])
                    
                    with fin_tab1:
                        st.markdown("#### 📊 INCOME STATEMENT")
                        period_type = st.radio("Period", ["Annual", "Quarterly"], horizontal=True, key="income_period")
                        income_df = financials['income_stmt'] if period_type == "Annual" else financials['quarterly_income']
                        
                        if income_df is not None and not income_df.empty:
                            income_display = income_df.copy()
                            for col in income_display.columns:
                                income_display[col] = income_display[col].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "-")
                            st.dataframe(income_display, use_container_width=True)
                            
                            st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                            if st.button("🤖 ANALYZE WITH AI", key="analyze_income"):
                                with st.spinner('🤖 Analyzing...'):
                                    analysis = analyze_financials_with_ai("Income Statement", income_df, company_name, display_ticker)
                                    st.markdown(f'<div style="background-color: #0a0a0a; border-left: 3px solid #00FF00; padding: 15px;">{analysis}</div>', unsafe_allow_html=True)
                    
                    with fin_tab2:
                        st.markdown("#### 📊 BALANCE SHEET")
                        period_type_bs = st.radio("Period", ["Annual", "Quarterly"], horizontal=True, key="balance_period")
                        balance_df = financials['balance_sheet'] if period_type_bs == "Annual" else financials['quarterly_balance']
                        
                        if balance_df is not None and not balance_df.empty:
                            balance_display = balance_df.copy()
                            for col in balance_display.columns:
                                balance_display[col] = balance_display[col].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "-")
                            st.dataframe(balance_display, use_container_width=True)
                            
                            if st.button("🤖 ANALYZE WITH AI", key="analyze_balance"):
                                with st.spinner('🤖 Analyzing...'):
                                    analysis = analyze_financials_with_ai("Balance Sheet", balance_df, company_name, display_ticker)
                                    st.markdown(f'<div style="background-color: #0a0a0a; border-left: 3px solid #00FF00; padding: 15px;">{analysis}</div>', unsafe_allow_html=True)
                    
                    with fin_tab3:
                        st.markdown("#### 📊 CASH FLOW STATEMENT")
                        period_type_cf = st.radio("Period", ["Annual", "Quarterly"], horizontal=True, key="cashflow_period")
                        cashflow_df = financials['cash_flow'] if period_type_cf == "Annual" else financials['quarterly_cashflow']
                        
                        if cashflow_df is not None and not cashflow_df.empty:
                            cashflow_display = cashflow_df.copy()
                            for col in cashflow_display.columns:
                                cashflow_display[col] = cashflow_display[col].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "-")
                            st.dataframe(cashflow_display, use_container_width=True)
                            
                            if st.button("🤖 ANALYZE WITH AI", key="analyze_cashflow"):
                                with st.spinner('🤖 Analyzing...'):
                                    analysis = analyze_financials_with_ai("Cash Flow", cashflow_df, company_name, display_ticker)
                                    st.markdown(f'<div style="background-color: #0a0a0a; border-left: 3px solid #00FF00; padding: 15px;">{analysis}</div>', unsafe_allow_html=True)
            
            
            # ===== TAB 4: VALUATION =====
            with tab4:
                st.markdown("### 📊 VALUATION METRICS")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💰 VALUATION RATIOS")
                    val_metrics = {'P/E (TTM)': info.get('trailingPE'), 'Forward P/E': info.get('forwardPE'),
                                  'PEG': info.get('pegRatio'), 'P/S': info.get('priceToSalesTrailing12Months'),
                                  'P/B': info.get('priceToBook'), 'EV/EBITDA': info.get('enterpriseToEbitda')}
                    val_data = [{'Metric': k, 'Value': f"{v:.2f}" if isinstance(v, (int, float)) else 'N/A'} for k, v in val_metrics.items()]
                    st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### 📊 PROFITABILITY")
                    prof_metrics = {'Profit Margin': info.get('profitMargins'), 'Operating Margin': info.get('operatingMargins'),
                                   'ROA': info.get('returnOnAssets'), 'ROE': info.get('returnOnEquity')}
                    prof_data = [{'Metric': k, 'Value': f"{v*100:.2f}%" if isinstance(v, (int, float)) else 'N/A'} for k, v in prof_metrics.items()]
                    st.dataframe(pd.DataFrame(prof_data), use_container_width=True, hide_index=True)
                
                st.markdown("#### 🎯 ANALYST TARGETS")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("TARGET HIGH", f"${info.get('targetHighPrice', 0):.2f}" if info.get('targetHighPrice') else "N/A")
                with col2:
                    st.metric("TARGET MEAN", f"${info.get('targetMeanPrice', 0):.2f}" if info.get('targetMeanPrice') else "N/A")
                with col3:
                    st.metric("TARGET LOW", f"${info.get('targetLowPrice', 0):.2f}" if info.get('targetLowPrice') else "N/A")

            with tab4:
    st.markdown("### 🔬 ADVANCED ANALYSIS")
    
    # Sous-onglets pour organiser l'information
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
        "💰 VALUATION", "📊 FINANCIAL HEALTH", "📈 TECHNICAL", "👥 OWNERSHIP", "⚖️ PEER COMPARISON"
    ])
    
    # ===== SUB-TAB 1: VALUATION =====
    with analysis_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 VALUATION RATIOS")
            val_metrics = {
                'P/E (TTM)': info.get('trailingPE'),
                'Forward P/E': info.get('forwardPE'),
                'PEG': info.get('pegRatio'),
                'P/S': info.get('priceToSalesTrailing12Months'),
                'P/B': info.get('priceToBook'),
                'EV/EBITDA': info.get('enterpriseToEbitda'),
                'EV/Revenue': info.get('enterpriseToRevenue')
            }
            val_data = [{'Metric': k, 'Value': f"{v:.2f}" if isinstance(v, (int, float)) else 'N/A'} for k, v in val_metrics.items()]
            st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📊 PROFITABILITY")
            prof_metrics = {
                'Profit Margin': info.get('profitMargins'),
                'Operating Margin': info.get('operatingMargins'),
                'EBITDA Margin': info.get('ebitdaMargins'),
                'ROA': info.get('returnOnAssets'),
                'ROE': info.get('returnOnEquity'),
                'ROIC': info.get('returnOnCapital')
            }
            prof_data = [{'Metric': k, 'Value': f"{v*100:.2f}%" if isinstance(v, (int, float)) else 'N/A'} for k, v in prof_metrics.items()]
            st.dataframe(pd.DataFrame(prof_data), use_container_width=True, hide_index=True)
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        
        st.markdown("#### 🎯 ANALYST TARGETS & RECOMMENDATIONS")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("TARGET HIGH", f"${info.get('targetHighPrice', 0):.2f}" if info.get('targetHighPrice') else "N/A")
        with col2:
            st.metric("TARGET MEAN", f"${info.get('targetMeanPrice', 0):.2f}" if info.get('targetMeanPrice') else "N/A")
        with col3:
            st.metric("TARGET LOW", f"${info.get('targetLowPrice', 0):.2f}" if info.get('targetLowPrice') else "N/A")
        with col4:
            num_analysts = info.get('numberOfAnalystOpinions', 0)
            st.metric("ANALYSTS", f"{num_analysts}" if num_analysts else "N/A")
        
        # Recommendations
        recommendation = info.get('recommendationKey', 'N/A').upper()
        rec_color = '#00FF00' if recommendation in ['BUY', 'STRONG_BUY'] else '#FFAA00' if recommendation == 'HOLD' else '#FF0000'
        st.markdown(f"""
        <div style="background-color: #111; border-left: 3px solid {rec_color}; padding: 15px; margin: 10px 0;">
            <p style="font-size: 12px; color: #FFAA00; font-weight: bold;">CONSENSUS: <span style="color: {rec_color};">{recommendation}</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== SUB-TAB 2: FINANCIAL HEALTH =====
    with analysis_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💧 LIQUIDITY RATIOS")
            liquidity_metrics = {
                'Current Ratio': info.get('currentRatio'),
                'Quick Ratio': info.get('quickRatio'),
                'Cash Ratio': (info.get('totalCash', 0) / info.get('totalCurrentLiabilities', 1)) if info.get('totalCurrentLiabilities') else None,
                'Working Capital': info.get('workingCapital')
            }
            liq_data = []
            for k, v in liquidity_metrics.items():
                if k == 'Working Capital' and isinstance(v, (int, float)):
                    display = f"${v/1e9:.2f}B" if v >= 1e9 else f"${v/1e6:.2f}M"
                elif isinstance(v, (int, float)):
                    display = f"{v:.2f}"
                else:
                    display = 'N/A'
                liq_data.append({'Metric': k, 'Value': display})
            st.dataframe(pd.DataFrame(liq_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📊 LEVERAGE RATIOS")
            debt_metrics = {
                'Debt/Equity': info.get('debtToEquity'),
                'Total Debt': info.get('totalDebt'),
                'Net Debt': (info.get('totalDebt', 0) - info.get('totalCash', 0)),
                'Interest Coverage': (info.get('ebitda', 0) / info.get('interestExpense', 1)) if info.get('interestExpense') else None
            }
            debt_data = []
            for k, v in debt_metrics.items():
                if k in ['Total Debt', 'Net Debt'] and isinstance(v, (int, float)):
                    display = f"${v/1e9:.2f}B" if abs(v) >= 1e9 else f"${v/1e6:.2f}M"
                elif isinstance(v, (int, float)):
                    display = f"{v:.2f}"
                else:
                    display = 'N/A'
                debt_data.append({'Metric': k, 'Value': display})
            st.dataframe(pd.DataFrame(debt_data), use_container_width=True, hide_index=True)
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        
        st.markdown("#### 💰 CASH MANAGEMENT")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_cash = info.get('totalCash', 0)
            st.metric("TOTAL CASH", f"${total_cash/1e9:.2f}B" if total_cash >= 1e9 else f"${total_cash/1e6:.2f}M" if total_cash else "N/A")
        with col2:
            fcf = info.get('freeCashflow', 0)
            st.metric("FREE CASH FLOW", f"${fcf/1e9:.2f}B" if fcf >= 1e9 else f"${fcf/1e6:.2f}M" if fcf else "N/A")
        with col3:
            op_cf = info.get('operatingCashflow', 0)
            st.metric("OPERATING CF", f"${op_cf/1e9:.2f}B" if op_cf >= 1e9 else f"${op_cf/1e6:.2f}M" if op_cf else "N/A")
        with col4:
            fcf_margin = (fcf / info.get('totalRevenue', 1) * 100) if info.get('totalRevenue') and fcf else None
            st.metric("FCF MARGIN", f"{fcf_margin:.2f}%" if fcf_margin else "N/A")
    
    # ===== SUB-TAB 3: TECHNICAL ANALYSIS =====
    with analysis_tab3:
        st.markdown("#### 📈 TECHNICAL INDICATORS")
        
        price_hist = get_price_history(display_ticker, period='6mo')
        if price_hist is not None and not price_hist.empty:
            # Calcul des indicateurs techniques
            close = price_hist['Close']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else None
            
            # Moving Averages
            sma_20 = close.rolling(window=20).mean().iloc[-1]
            sma_50 = close.rolling(window=50).mean().iloc[-1]
            sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None
            
            # MACD
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_current = macd.iloc[-1] if not macd.empty else None
            signal_current = signal.iloc[-1] if not signal.empty else None
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                rsi_color = '#00FF00' if current_rsi and current_rsi < 30 else '#FF0000' if current_rsi and current_rsi > 70 else '#FFAA00'
                st.markdown(f'<p style="color: #FFAA00; font-size: 10px; font-weight: bold;">RSI (14)</p><p style="color: {rsi_color}; font-size: 20px; font-weight: bold;">{current_rsi:.2f}</p>', unsafe_allow_html=True)
            with col2:
                st.metric("SMA 20", f"${sma_20:.2f}" if pd.notna(sma_20) else "N/A")
            with col3:
                st.metric("SMA 50", f"${sma_50:.2f}" if pd.notna(sma_50) else "N/A")
            with col4:
                st.metric("SMA 200", f"${sma_200:.2f}" if pd.notna(sma_200) else "N/A")
            
            st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
            
            # Signal d'achat/vente
            current_price = info.get('currentPrice', close.iloc[-1])
            signals = []
            if current_rsi:
                if current_rsi < 30:
                    signals.append("🟢 RSI oversold - potential BUY signal")
                elif current_rsi > 70:
                    signals.append("🔴 RSI overbought - potential SELL signal")
            
            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    signals.append("🟢 Golden Cross (SMA20 > SMA50)")
                else:
                    signals.append("🔴 Death Cross (SMA20 < SMA50)")
            
            if macd_current and signal_current:
                if macd_current > signal_current:
                    signals.append("🟢 MACD Bullish")
                else:
                    signals.append("🔴 MACD Bearish")
            
            st.markdown("#### 📊 TECHNICAL SIGNALS")
            for signal in signals:
                color = '#00FF00' if '🟢' in signal else '#FF0000'
                st.markdown(f'<div style="background-color: #111; border-left: 3px solid {color}; padding: 10px; margin: 5px 0;"><p style="font-size: 11px; color: #FFAA00;">{signal}</p></div>', unsafe_allow_html=True)
        else:
            st.info("Not enough data for technical analysis")
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        
        st.markdown("#### 📊 SUPPORT & RESISTANCE")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("52W HIGH", f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A")
        with col2:
            st.metric("52W LOW", f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else "N/A")
        with col3:
            beta = info.get('beta', 0)
            st.metric("BETA", f"{beta:.2f}" if beta else "N/A")
    
    # ===== SUB-TAB 4: OWNERSHIP =====
    with analysis_tab4:
        st.markdown("#### 👥 OWNERSHIP STRUCTURE")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            insider = info.get('heldPercentInsiders', 0)
            st.metric("INSIDER OWNERSHIP", f"{insider*100:.2f}%" if isinstance(insider, (int, float)) else "N/A")
        with col2:
            inst = info.get('heldPercentInstitutions', 0)
            st.metric("INSTITUTIONAL", f"{inst*100:.2f}%" if isinstance(inst, (int, float)) else "N/A")
        with col3:
            float_shares = info.get('floatShares', 0)
            st.metric("FLOAT SHARES", f"{float_shares/1e9:.2f}B" if float_shares >= 1e9 else f"{float_shares/1e6:.2f}M" if float_shares else "N/A")
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        
        st.markdown("#### 📊 SHARE STRUCTURE")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            shares_out = info.get('sharesOutstanding', 0)
            st.metric("SHARES OUT", f"{shares_out/1e9:.2f}B" if shares_out >= 1e9 else f"{shares_out/1e6:.2f}M" if shares_out else "N/A")
        with col2:
            implied_shares = info.get('impliedSharesOutstanding', 0)
            st.metric("IMPLIED SHARES", f"{implied_shares/1e9:.2f}B" if implied_shares >= 1e9 else f"{implied_shares/1e6:.2f}M" if implied_shares else "N/A")
        with col3:
            short_percent = info.get('shortPercentOfFloat', 0)
            st.metric("SHORT % OF FLOAT", f"{short_percent*100:.2f}%" if isinstance(short_percent, (int, float)) else "N/A")
        with col4:
            short_ratio = info.get('shortRatio', 0)
            st.metric("SHORT RATIO", f"{short_ratio:.2f}" if short_ratio else "N/A")
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        
        # Top Holders (if available)
        st.markdown("#### 🏦 MAJOR HOLDERS")
        try:
            major_holders = stock.major_holders
            if major_holders is not None and not major_holders.empty:
                st.dataframe(major_holders, use_container_width=True, hide_index=True)
            else:
                st.info("Major holders data not available")
        except:
            st.info("Major holders data not available")
    
    # ===== SUB-TAB 5: PEER COMPARISON =====
    with analysis_tab5:
        st.markdown("#### ⚖️ PEER COMPARISON")
        
        # Note: Pour une vraie comparaison, il faudrait récupérer les données des concurrents
        # Ici on affiche les métriques clés de l'entreprise actuelle
        
        peers_text = info.get('sector', 'N/A')
        st.markdown(f"""
        <div style="background-color: #111; border-left: 3px solid #FFAA00; padding: 15px; margin: 10px 0;">
            <p style="font-size: 11px; color: #999;">Sector: <strong style="color: #FFAA00;">{peers_text}</strong></p>
            <p style="font-size: 10px; color: #666; margin-top: 5px;">For detailed peer comparison, consider comparing with other companies in the {peers_text} sector.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 COMPANY VS SECTOR METRICS")
        
        comparison_metrics = {
            'Market Cap': info.get('marketCap'),
            'P/E Ratio': info.get('trailingPE'),
            'Profit Margin': info.get('profitMargins'),
            'ROE': info.get('returnOnEquity'),
            'Debt/Equity': info.get('debtToEquity'),
            'Revenue Growth': info.get('revenueGrowth')
        }
        
        comp_data = []
        for k, v in comparison_metrics.items():
            if k == 'Market Cap' and isinstance(v, (int, float)):
                display = f"${v/1e12:.2f}T" if v >= 1e12 else f"${v/1e9:.2f}B"
            elif k in ['Profit Margin', 'ROE', 'Revenue Growth'] and isinstance(v, (int, float)):
                display = f"{v*100:.2f}%"
            elif isinstance(v, (int, float)):
                display = f"{v:.2f}"
            else:
                display = 'N/A'
            comp_data.append({'Metric': k, 'Value': display, 'Sector Avg': 'N/A'})
        
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        
        st.info("💡 Tip: For detailed peer analysis, search for competitor tickers in the same sector.")
            
            # ===== TAB 5: DETAILED INFO =====
            with tab5:
                st.markdown("### 🔍 DETAILED INFO")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    insider = info.get('heldPercentInsiders', 0)
                    st.metric("INSIDER %", f"{insider*100:.2f}%" if isinstance(insider, (int, float)) else "N/A")
                with col2:
                    inst = info.get('heldPercentInstitutions', 0)
                    st.metric("INSTITUTIONAL %", f"{inst*100:.2f}%" if isinstance(inst, (int, float)) else "N/A")
                with col3:
                    float_shares = info.get('floatShares', 0)
                    st.metric("FLOAT", f"{float_shares/1e9:.2f}B" if float_shares >= 1e9 else f"{float_shares/1e6:.2f}M" if float_shares else "N/A")
                
                with st.expander("🔍 RAW DATA"):
                    info_items = [{'Field': k, 'Value': str(v)} for k, v in info.items() if not isinstance(v, (dict, list))]
                    st.dataframe(pd.DataFrame(info_items), use_container_width=True, hide_index=True)
        
        else:
            st.error(f"❌ Could not find data for '{display_ticker}'")

elif not ticker_input and search_button:
    st.warning("⚠️ Please enter a ticker symbol")

# Info section when no ticker
if not display_ticker:
    st.markdown("""
    <div style="background-color: #111; border: 2px solid #FFAA00; padding: 20px; margin: 20px 0;">
        <h3 style="color: #FFAA00;">📊 COMPANY ANALYSIS</h3>
        <p style="font-size: 11px; color: #999;">Enter any stock ticker to analyze:</p>
        <ul style="font-size: 10px; color: #999;">
            <li>🇺🇸 US: AAPL, MSFT, GOOGL, TSLA</li>
            <li>🇫🇷 France: MC.PA, AIR.PA</li>
            <li>🇨🇭 Switzerland: NESN.SW, NOVN.SW</li>
            <li>🇬🇧 UK: SHEL.L, BP.L</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; padding: 10px; border-top: 1px solid #333; margin-top: 20px;'>
    © 2025 BLOOMBERG ENS® | Data: Yahoo Finance & TradingView | {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
