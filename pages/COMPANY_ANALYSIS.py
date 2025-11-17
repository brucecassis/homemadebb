import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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

# ===== BARRE DE RECHERCHE =====
st.markdown("### 🔍 COMPANY SEARCH")

col_search1, col_search2 = st.columns([4, 1])

with col_search1:
    ticker_input = st.text_input(
        "",
        placeholder="Enter ticker symbol (e.g., AAPL, MSFT, TSLA, MC.PA...)",
        key="ticker_search",
        label_visibility="collapsed"
    ).upper()

with col_search2:
    search_button = st.button("🔍 ANALYZE", use_container_width=True, key="search_company")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)

# ===== AFFICHAGE DES DONNÉES =====
if search_button and ticker_input:
    with st.spinner(f'🔍 Analyzing {ticker_input}...'):
        stock, info = get_company_info(ticker_input)
        
        if stock and info:
            # ===== COMPANY HEADER =====
            company_name = info.get('longName', ticker_input)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            website = info.get('website', 'N/A')
            
            st.markdown(f"""
            <div class="company-header">
                <h1 style="font-size: 24px; margin: 0; color: #00FF00;">{company_name} ({ticker_input})</h1>
                <p style="margin: 5px 0; font-size: 12px; color: #FFAA00;">
                    <strong>Sector:</strong> {sector} | <strong>Industry:</strong> {industry}
                </p>
                <p style="margin: 5px 0; font-size: 10px; color: #999;">
                    <strong>Website:</strong> {website}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # ===== ONGLETS PRINCIPAUX =====
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 OVERVIEW",
                "💰 FINANCIALS",
                "📈 PRICE & PERFORMANCE",
                "📊 VALUATION",
                "📰 NEWS & EVENTS",
                "🔍 DETAILED INFO"
            ])
            
            # ===== TAB 1: OVERVIEW =====
            with tab1:
                st.markdown("### 📊 COMPANY OVERVIEW")
                
                # Prix actuel et métriques clés
                col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
                
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                previous_close = info.get('previousClose', 0)
                change = current_price - previous_close if current_price and previous_close else 0
                change_pct = (change / previous_close * 100) if previous_close else 0
                
                with col_ov1:
                    st.metric(
                        "CURRENT PRICE",
                        f"${current_price:.2f}" if current_price else "N/A",
                        delta=f"{change:+.2f} ({change_pct:+.2f}%)"
                    )
                
                with col_ov2:
                    market_cap = info.get('marketCap', 0)
                    if market_cap:
                        if market_cap >= 1e12:
                            cap_display = f"${market_cap/1e12:.2f}T"
                        elif market_cap >= 1e9:
                            cap_display = f"${market_cap/1e9:.2f}B"
                        else:
                            cap_display = f"${market_cap/1e6:.2f}M"
                    else:
                        cap_display = "N/A"
                    
                    st.metric("MARKET CAP", cap_display)
                
                with col_ov3:
                    pe_ratio = info.get('trailingPE', 0)
                    st.metric("P/E RATIO", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                
                with col_ov4:
                    dividend_yield = info.get('dividendYield', 0)
                    if dividend_yield:
                        # Yahoo Finance retourne généralement en décimal (0.0034 = 0.34%)
                        # mais parfois peut varier selon l'action
                        if dividend_yield < 1:  # Si < 1, c'est un décimal
                            st.metric("DIVIDEND YIELD", f"{dividend_yield*100:.2f}%")
                        else:  # Si >= 1, c'est déjà en pourcentage
                            st.metric("DIVIDEND YIELD", f"{dividend_yield:.2f}%")
                    else:
                        st.metric("DIVIDEND YIELD", "N/A")
                
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # Trading Information
                st.markdown("#### 📊 TRADING INFORMATION")
                
                col_trade1, col_trade2, col_trade3, col_trade4 = st.columns(4)
                
                with col_trade1:
                    volume = info.get('volume', 0)
                    avg_volume = info.get('averageVolume', 0)
                    st.metric("VOLUME", f"{volume:,.0f}" if volume else "N/A")
                    st.caption(f"Avg: {avg_volume:,.0f}" if avg_volume else "")
                
                with col_trade2:
                    day_high = info.get('dayHigh', 0)
                    day_low = info.get('dayLow', 0)
                    st.metric("DAY RANGE", f"${day_low:.2f} - ${day_high:.2f}" if day_low and day_high else "N/A")
                
                with col_trade3:
                    year_high = info.get('fiftyTwoWeekHigh', 0)
                    year_low = info.get('fiftyTwoWeekLow', 0)
                    st.metric("52W RANGE", f"${year_low:.2f} - ${year_high:.2f}" if year_low and year_high else "N/A")
                
                with col_trade4:
                    beta = info.get('beta', 0)
                    st.metric("BETA", f"{beta:.2f}" if beta else "N/A")
                
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # Company Description
                st.markdown("#### 📝 COMPANY DESCRIPTION")
                
                description = info.get('longBusinessSummary', 'No description available.')
                st.markdown(f"""
                <div style="background-color: #111; padding: 15px; border-left: 3px solid #FFAA00;">
                    <p style="font-size: 11px; color: #999; line-height: 1.6;">
                        {description}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key People
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 👥 KEY EXECUTIVES")
                
                officers = info.get('companyOfficers', [])
                
                if officers:
                    exec_data = []
                    for officer in officers[:5]:  # Top 5
                        # Correction : totalPay peut être un int directement ou un dict
                        total_pay = officer.get('totalPay', None)
                        
                        if isinstance(total_pay, dict):
                            pay_display = f"${total_pay.get('fmt', 'N/A')}"
                        elif isinstance(total_pay, (int, float)):
                            pay_display = f"${total_pay:,.0f}" if total_pay else 'N/A'
                        else:
                            pay_display = 'N/A'
                        
                        exec_data.append({
                            'Name': officer.get('name', 'N/A'),
                            'Title': officer.get('title', 'N/A'),
                            'Pay': pay_display
                        })
                    
                    exec_df = pd.DataFrame(exec_data)
                    st.dataframe(exec_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Executive information not available")


            
            # ===== TAB 2: FINANCIALS =====
            with tab2:
                st.markdown("### 💰 FINANCIAL STATEMENTS")
                
                financials = get_financial_statements(ticker_input)
                
                if financials:
                    # Sous-onglets pour les états financiers
                    fin_tab1, fin_tab2, fin_tab3 = st.tabs([
                        "📊 INCOME STATEMENT",
                        "📊 BALANCE SHEET",
                        "📊 CASH FLOW"
                    ])
                    
                    # Income Statement
                    with fin_tab1:
                        st.markdown("#### 📊 INCOME STATEMENT")
                        
                        period_type = st.radio(
                            "Period",
                            options=["Annual", "Quarterly"],
                            horizontal=True,
                            key="income_period"
                        )
                        
                        if period_type == "Annual":
                            income_df = financials['income_stmt']
                        else:
                            income_df = financials['quarterly_income']
                        
                        if income_df is not None and not income_df.empty:
                            # Formatter les nombres
                            income_display = income_df.copy()
                            
                            # Convertir en millions
                            for col in income_display.columns:
                                income_display[col] = income_display[col].apply(
                                    lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "-"
                                )
                            
                            st.dataframe(income_display, use_container_width=True)
                            
                            # Key metrics from income statement
                            st.markdown("#### 📊 KEY METRICS")
                            
                            col_inc1, col_inc2, col_inc3, col_inc4 = st.columns(4)
                            
                            try:
                                revenue = income_df.loc['Total Revenue'].iloc[0]
                                gross_profit = income_df.loc['Gross Profit'].iloc[0] if 'Gross Profit' in income_df.index else 0
                                operating_income = income_df.loc['Operating Income'].iloc[0] if 'Operating Income' in income_df.index else 0
                                net_income = income_df.loc['Net Income'].iloc[0] if 'Net Income' in income_df.index else 0
                                
                                with col_inc1:
                                    st.metric("REVENUE", f"${revenue/1e9:.2f}B" if revenue else "N/A")
                                
                                with col_inc2:
                                    gross_margin = (gross_profit / revenue * 100) if revenue and gross_profit else 0
                                    st.metric("GROSS MARGIN", f"{gross_margin:.1f}%" if gross_margin else "N/A")
                                
                                with col_inc3:
                                    operating_margin = (operating_income / revenue * 100) if revenue and operating_income else 0
                                    st.metric("OPERATING MARGIN", f"{operating_margin:.1f}%" if operating_margin else "N/A")
                                
                                with col_inc4:
                                    net_margin = (net_income / revenue * 100) if revenue and net_income else 0
                                    st.metric("NET MARGIN", f"{net_margin:.1f}%" if net_margin else "N/A")
                            
                            except:
                                st.warning("Could not calculate key metrics")
                        
                        else:
                            st.warning("Income statement data not available")
                    
                    # Balance Sheet
                    with fin_tab2:
                        st.markdown("#### 📊 BALANCE SHEET")
                        
                        period_type_bs = st.radio(
                            "Period",
                            options=["Annual", "Quarterly"],
                            horizontal=True,
                            key="balance_period"
                        )
                        
                        if period_type_bs == "Annual":
                            balance_df = financials['balance_sheet']
                        else:
                            balance_df = financials['quarterly_balance']
                        
                        if balance_df is not None and not balance_df.empty:
                            # Formatter
                            balance_display = balance_df.copy()
                            
                            for col in balance_display.columns:
                                balance_display[col] = balance_display[col].apply(
                                    lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "-"
                                )
                            
                            st.dataframe(balance_display, use_container_width=True)
                            
                            # Key ratios
                            st.markdown("#### 📊 KEY RATIOS")
                            
                            col_bs1, col_bs2, col_bs3, col_bs4 = st.columns(4)
                            
                            try:
                                total_assets = balance_df.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_df.index else 0
                                total_liabilities = balance_df.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_df.index else 0
                                stockholder_equity = balance_df.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_df.index else 0
                                current_assets = balance_df.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_df.index else 0
                                current_liabilities = balance_df.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_df.index else 0
                                
                                with col_bs1:
                                    st.metric("TOTAL ASSETS", f"${total_assets/1e9:.2f}B" if total_assets else "N/A")
                                
                                with col_bs2:
                                    debt_to_equity = (total_liabilities / stockholder_equity) if stockholder_equity else 0
                                    st.metric("DEBT/EQUITY", f"{debt_to_equity:.2f}" if debt_to_equity else "N/A")
                                
                                with col_bs3:
                                    current_ratio = (current_assets / current_liabilities) if current_liabilities else 0
                                    st.metric("CURRENT RATIO", f"{current_ratio:.2f}" if current_ratio else "N/A")
                                
                                with col_bs4:
                                    st.metric("EQUITY", f"${stockholder_equity/1e9:.2f}B" if stockholder_equity else "N/A")
                            
                            except:
                                st.warning("Could not calculate key ratios")
                        
                        else:
                            st.warning("Balance sheet data not available")
                    
                    # Cash Flow
                    with fin_tab3:
                        st.markdown("#### 📊 CASH FLOW STATEMENT")
                        
                        period_type_cf = st.radio(
                            "Period",
                            options=["Annual", "Quarterly"],
                            horizontal=True,
                            key="cashflow_period"
                        )
                        
                        if period_type_cf == "Annual":
                            cashflow_df = financials['cash_flow']
                        else:
                            cashflow_df = financials['quarterly_cashflow']
                        
                        if cashflow_df is not None and not cashflow_df.empty:
                            # Formatter
                            cashflow_display = cashflow_df.copy()
                            
                            for col in cashflow_display.columns:
                                cashflow_display[col] = cashflow_display[col].apply(
                                    lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "-"
                                )
                            
                            st.dataframe(cashflow_display, use_container_width=True)
                            
                            # Key cash flow metrics
                            st.markdown("#### 📊 CASH FLOW METRICS")
                            
                            col_cf1, col_cf2, col_cf3 = st.columns(3)
                            
                            try:
                                operating_cf = cashflow_df.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow_df.index else 0
                                investing_cf = cashflow_df.loc['Investing Cash Flow'].iloc[0] if 'Investing Cash Flow' in cashflow_df.index else 0
                                financing_cf = cashflow_df.loc['Financing Cash Flow'].iloc[0] if 'Financing Cash Flow' in cashflow_df.index else 0
                                free_cf = cashflow_df.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cashflow_df.index else 0
                                
                                with col_cf1:
                                    st.metric("OPERATING CF", f"${operating_cf/1e9:.2f}B" if operating_cf else "N/A")
                                
                                with col_cf2:
                                    st.metric("FREE CF", f"${free_cf/1e9:.2f}B" if free_cf else "N/A")
                                
                                with col_cf3:
                                    fcf_margin = (free_cf / revenue * 100) if 'revenue' in locals() and revenue and free_cf else 0
                                    st.metric("FCF MARGIN", f"{fcf_margin:.1f}%" if fcf_margin else "N/A")
                            
                            except:
                                st.warning("Could not calculate cash flow metrics")
                        
                        else:
                            st.warning("Cash flow data not available")
                
                else:
                    st.error("Could not retrieve financial statements")
            
            # ===== TAB 3: PRICE & PERFORMANCE =====
            with tab3:
                st.markdown("### 📈 PRICE & PERFORMANCE")
                
                # Sélection de période
                col_perf1, col_perf2 = st.columns([3, 1])
                
                with col_perf1:
                    time_period = st.selectbox(
                        "TIME PERIOD",
                        options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
                        index=3,
                        key="price_period"
                    )
                
                with col_perf2:
                    if st.button("📊 UPDATE CHART", use_container_width=True):
                        st.cache_data.clear()
                
                # Récupérer données de prix
                price_hist = get_price_history(ticker_input, period=time_period)
                
                if price_hist is not None and not price_hist.empty:
                    # Graphique prix + volume
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Price', 'Volume'),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=price_hist.index,
                            open=price_hist['Open'],
                            high=price_hist['High'],
                            low=price_hist['Low'],
                            close=price_hist['Close'],
                            name='Price',
                            increasing_line_color='#00FF00',
                            decreasing_line_color='#FF0000'
                        ),
                        row=1, col=1
                    )
                    
                    # Volume
                    colors = ['#00FF00' if price_hist['Close'].iloc[i] >= price_hist['Open'].iloc[i] 
                             else '#FF0000' for i in range(len(price_hist))]
                    
                    fig.add_trace(
                        go.Bar(
                            x=price_hist.index,
                            y=price_hist['Volume'],
                            name='Volume',
                            marker_color=colors,
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f"{ticker_input} Price Chart",
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        xaxis_rangeslider_visible=False,
                        height=600,
                        hovermode='x unified'
                    )
                    
                    fig.update_xaxes(gridcolor='#333')
                    fig.update_yaxes(gridcolor='#333')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance Statistics
                    st.markdown("#### 📊 PERFORMANCE STATISTICS")
                    
                    # Calculer returns
                    returns = price_hist['Close'].pct_change()
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        total_return = ((price_hist['Close'].iloc[-1] / price_hist['Close'].iloc[0]) - 1) * 100
                        st.metric("TOTAL RETURN", f"{total_return:+.2f}%")
                    
                    with col_stat2:
                        volatility = returns.std() * np.sqrt(252) * 100
                        st.metric("VOLATILITY (Ann.)", f"{volatility:.2f}%")
                    
                    with col_stat3:
                        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                        st.metric("SHARPE RATIO", f"{sharpe:.2f}")
                    
                    with col_stat4:
                        max_drawdown = ((price_hist['Close'] / price_hist['Close'].cummax()) - 1).min() * 100
                        st.metric("MAX DRAWDOWN", f"{max_drawdown:.2f}%")
                    
                    # Moving Averages
                    st.markdown("#### 📊 TECHNICAL INDICATORS")
                    
                    price_hist['MA50'] = price_hist['Close'].rolling(50).mean()
                    price_hist['MA200'] = price_hist['Close'].rolling(200).mean()
                    
                    fig_ma = go.Figure()
                    
                    fig_ma.add_trace(go.Scatter(
                        x=price_hist.index,
                        y=price_hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#FFAA00', width=2)
                    ))
                    
                    fig_ma.add_trace(go.Scatter(
                        x=price_hist.index,
                        y=price_hist['MA50'],
                        mode='lines',
                        name='MA 50',
                        line=dict(color='#00FF00', width=1, dash='dash')
                    ))
                    
                    fig_ma.add_trace(go.Scatter(
                        x=price_hist.index,
                        y=price_hist['MA200'],
                        mode='lines',
                        name='MA 200',
                        line=dict(color='#FF0000', width=1, dash='dash')
                    ))
                    
                    fig_ma.update_layout(
                        title="Price with Moving Averages",
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        xaxis=dict(gridcolor='#333'),
                        yaxis=dict(gridcolor='#333'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_ma, use_container_width=True)
                    
                    # Current position vs MAs
                    current_price_val = price_hist['Close'].iloc[-1]
                    ma50_val = price_hist['MA50'].iloc[-1]
                    ma200_val = price_hist['MA200'].iloc[-1]
                    
                    col_ma1, col_ma2, col_ma3 = st.columns(3)
                    
                    with col_ma1:
                        st.metric("CURRENT PRICE", f"${current_price_val:.2f}")
                    
                    with col_ma2:
                        ma50_diff = ((current_price_val / ma50_val) - 1) * 100 if pd.notna(ma50_val) else 0
                        st.metric("vs MA50", f"{ma50_diff:+.2f}%")
                    
                    with col_ma3:
                        ma200_diff = ((current_price_val / ma200_val) - 1) * 100 if pd.notna(ma200_val) else 0
                        st.metric("vs MA200", f"{ma200_diff:+.2f}%")
                
                else:
                    st.error("Could not retrieve price history")
            
            # ===== TAB 4: VALUATION =====
            with tab4:
                st.markdown("### 📊 VALUATION METRICS")
                
                col_val1, col_val2 = st.columns(2)
                
                with col_val1:
                    st.markdown("#### 💰 VALUATION RATIOS")
                    
                    valuation_metrics = {
                        'P/E Ratio (TTM)': info.get('trailingPE', 'N/A'),
                        'Forward P/E': info.get('forwardPE', 'N/A'),
                        'PEG Ratio': info.get('pegRatio', 'N/A'),
                        'Price/Sales (TTM)': info.get('priceToSalesTrailing12Months', 'N/A'),
                        'Price/Book': info.get('priceToBook', 'N/A'),
                        'EV/Revenue': info.get('enterpriseToRevenue', 'N/A'),
                        'EV/EBITDA': info.get('enterpriseToEbitda', 'N/A')
                    }
                    
                    val_data = []
                    for metric, value in valuation_metrics.items():
                        if isinstance(value, (int, float)):
                            val_data.append({'Metric': metric, 'Value': f"{value:.2f}"})
                        else:
                            val_data.append({'Metric': metric, 'Value': str(value)})
                    
                    val_df = pd.DataFrame(val_data)
                    st.dataframe(val_df, use_container_width=True, hide_index=True)
                
                with col_val2:
                    st.markdown("#### 📊 PROFITABILITY METRICS")
                    
                    profitability_metrics = {
                        'Profit Margin': info.get('profitMargins', 'N/A'),
                        'Operating Margin': info.get('operatingMargins', 'N/A'),
                        'ROA (Return on Assets)': info.get('returnOnAssets', 'N/A'),
                        'ROE (Return on Equity)': info.get('returnOnEquity', 'N/A'),
                        'ROIC': info.get('returnOnCapital', 'N/A')
                    }
                    
                    prof_data = []
                    for metric, value in profitability_metrics.items():
                        if isinstance(value, (int, float)):
                            prof_data.append({'Metric': metric, 'Value': f"{value*100:.2f}%"})
                        else:
                            prof_data.append({'Metric': metric, 'Value': str(value)})
                    
                    prof_df = pd.DataFrame(prof_data)
                    st.dataframe(prof_df, use_container_width=True, hide_index=True)
                
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # Growth Metrics
                st.markdown("#### 📈 GROWTH METRICS")
                
                col_growth1, col_growth2, col_growth3, col_growth4 = st.columns(4)
                
                with col_growth1:
                    revenue_growth = info.get('revenueGrowth', 0)
                    if isinstance(revenue_growth, (int, float)):
                        st.metric("REVENUE GROWTH", f"{revenue_growth*100:.2f}%")
                    else:
                        st.metric("REVENUE GROWTH", "N/A")
                
                with col_growth2:
                    earnings_growth = info.get('earningsGrowth', 0)
                    if isinstance(earnings_growth, (int, float)):
                        st.metric("EARNINGS GROWTH", f"{earnings_growth*100:.2f}%")
                    else:
                        st.metric("EARNINGS GROWTH", "N/A")
                
                with col_growth3:
                    earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', 0)
                    if isinstance(earnings_quarterly_growth, (int, float)):
                        st.metric("EARNINGS GROWTH (Q)", f"{earnings_quarterly_growth*100:.2f}%")
                    else:
                        st.metric("EARNINGS GROWTH (Q)", "N/A")
                
                with col_growth4:
                    revenue_per_share = info.get('revenuePerShare', 0)
                    if isinstance(revenue_per_share, (int, float)):
                        st.metric("REVENUE/SHARE", f"${revenue_per_share:.2f}")
                    else:
                        st.metric("REVENUE/SHARE", "N/A")
                
                # Analyst Targets
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 🎯 ANALYST TARGETS")
                
                col_target1, col_target2, col_target3 = st.columns(3)
                
                with col_target1:
                    target_high = info.get('targetHighPrice', 0)
                    st.metric("TARGET HIGH", f"${target_high:.2f}" if target_high else "N/A")
                
                with col_target2:
                    target_mean = info.get('targetMeanPrice', 0)
                    st.metric("TARGET MEAN", f"${target_mean:.2f}" if target_mean else "N/A")
                
                with col_target3:
                    target_low = info.get('targetLowPrice', 0)
                    st.metric("TARGET LOW", f"${target_low:.2f}" if target_low else "N/A")
                
                # Upside/Downside
                if target_mean and current_price:
                    upside = ((target_mean / current_price) - 1) * 100
                    
                    if upside > 0:
                        st.markdown(f"""
                        <div class="info-box">
                            <p style="margin: 0; font-size: 12px; color: #00FF00; font-weight: bold;">
                            📈 UPSIDE POTENTIAL: +{upside:.2f}%
                            </p>
                            <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
                            Based on analyst consensus target
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <p style="margin: 0; font-size: 12px; color: #FF6600; font-weight: bold;">
                            📉 DOWNSIDE RISK: {upside:.2f}%
                            </p>
                            <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
                            Based on analyst consensus target
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 📊 ANALYST RECOMMENDATIONS")
                
                recommendation = info.get('recommendationKey', 'N/A')
                num_analysts = info.get('numberOfAnalystOpinions', 0)
                
                col_rec1, col_rec2 = st.columns(2)
                
                with col_rec1:
                    st.metric("CONSENSUS", recommendation.upper() if recommendation else "N/A")
                
                with col_rec2:
                    st.metric("# ANALYSTS", f"{num_analysts}" if num_analysts else "N/A")
            
            # ===== TAB 5: NEWS & EVENTS =====
            with tab5:
                st.markdown("### 📰 NEWS & EVENTS")
                
                # Recent News
                try:
                    news = stock.news
                    
                    if news:
                        st.markdown("#### 📰 RECENT NEWS")
                        
                        for idx, article in enumerate(news[:10]):
                            title = article.get('title', 'No title')
                            publisher = article.get('publisher', 'Unknown')
                            link = article.get('link', '#')
                            publish_time = article.get('providerPublishTime', 0)
                            
                            if publish_time:
                                date_str = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                            else:
                                date_str = 'Unknown date'
                            
                            st.markdown(f"""
                            <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 5px 0;">
                                <p style="margin: 0; font-size: 12px; color: #FFAA00; font-weight: bold;">
                                    {title}
                                </p>
                                <p style="margin: 5px 0 0 0; font-size: 9px; color: #666;">
                                    {publisher} | {date_str}
                                </p>
                                <a href="{link}" target="_blank" style="font-size: 9px; color: #00FFFF;">
                                    Read more →
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No recent news available")
                
                except:
                    st.warning("Could not retrieve news")
                
                # Calendar Events
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 📅 CALENDAR EVENTS")
                
                try:
                    calendar = stock.calendar
                    
                    if calendar is not None and not calendar.empty:
                        st.dataframe(calendar, use_container_width=True)
                    else:
                        st.info("No calendar events available")
                
                except:
                    st.info("Calendar data not available")
                
                # Earnings History
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 📊 EARNINGS HISTORY")
                
                try:
                    earnings_dates = stock.earnings_dates
                    
                    if earnings_dates is not None and not earnings_dates.empty:
                        # Limiter aux 8 derniers
                        recent_earnings = earnings_dates.head(8)
                        st.dataframe(recent_earnings, use_container_width=True)
                    else:
                        st.info("No earnings history available")
                
                except:
                    st.info("Earnings history not available")
            
            # ===== TAB 6: DETAILED INFO =====
            with tab6:
                st.markdown("### 🔍 DETAILED COMPANY INFORMATION")
                
                # Ownership & Institutional Holdings
                st.markdown("#### 👥 OWNERSHIP STRUCTURE")
                
                col_own1, col_own2, col_own3 = st.columns(3)
                
                with col_own1:
                    insider_pct = info.get('heldPercentInsiders', 0)
                    if isinstance(insider_pct, (int, float)):
                        st.metric("INSIDER OWNERSHIP", f"{insider_pct*100:.2f}%")
                    else:
                        st.metric("INSIDER OWNERSHIP", "N/A")
                
                with col_own2:
                    institution_pct = info.get('heldPercentInstitutions', 0)
                    if isinstance(institution_pct, (int, float)):
                        st.metric("INSTITUTIONAL", f"{institution_pct*100:.2f}%")
                    else:
                        st.metric("INSTITUTIONAL", "N/A")
                
                with col_own3:
                    float_shares = info.get('floatShares', 0)
                    if float_shares:
                        if float_shares >= 1e9:
                            float_display = f"{float_shares/1e9:.2f}B"
                        elif float_shares >= 1e6:
                            float_display = f"{float_shares/1e6:.2f}M"
                        else:
                            float_display = f"{float_shares:,.0f}"
                        st.metric("FLOAT SHARES", float_display)
                    else:
                        st.metric("FLOAT SHARES", "N/A")
                
                # Major Holders
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 🏦 MAJOR HOLDERS")
                
                try:
                    major_holders = stock.major_holders
                    
                    if major_holders is not None and not major_holders.empty:
                        st.dataframe(major_holders, use_container_width=True, hide_index=True)
                    else:
                        st.info("Major holders data not available")
                
                except:
                    st.info("Major holders data not available")
                
                # Institutional Holders
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 🏢 TOP INSTITUTIONAL HOLDERS")
                
                try:
                    institutional_holders = stock.institutional_holders
                    
                    if institutional_holders is not None and not institutional_holders.empty:
                        st.dataframe(institutional_holders.head(10), use_container_width=True, hide_index=True)
                    else:
                        st.info("Institutional holders data not available")
                
                except:
                    st.info("Institutional holders data not available")
                
                # Sustainability & ESG
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 🌱 ESG & SUSTAINABILITY")
                
                esg_scores = {
                    'ESG Score': info.get('esgScores', {}).get('totalEsg', 'N/A') if isinstance(info.get('esgScores'), dict) else 'N/A',
                    'Environment Score': info.get('esgScores', {}).get('environmentScore', 'N/A') if isinstance(info.get('esgScores'), dict) else 'N/A',
                    'Social Score': info.get('esgScores', {}).get('socialScore', 'N/A') if isinstance(info.get('esgScores'), dict) else 'N/A',
                    'Governance Score': info.get('esgScores', {}).get('governanceScore', 'N/A') if isinstance(info.get('esgScores'), dict) else 'N/A'
                }
                
                esg_available = any(v != 'N/A' for v in esg_scores.values())
                
                if esg_available:
                    col_esg1, col_esg2, col_esg3, col_esg4 = st.columns(4)
                    
                    cols = [col_esg1, col_esg2, col_esg3, col_esg4]
                    
                    for idx, (label, value) in enumerate(esg_scores.items()):
                        with cols[idx]:
                            if isinstance(value, (int, float)):
                                st.metric(label, f"{value:.1f}")
                            else:
                                st.metric(label, value)
                else:
                    st.info("ESG data not available for this company")
                
                # All Info (Raw Data)
                st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### 📊 RAW DATA (ALL AVAILABLE INFO)")
                
                with st.expander("🔍 VIEW RAW DATA", expanded=False):
                    # Convertir info dict en DataFrame pour affichage
                    info_items = []
                    for key, value in info.items():
                        if not isinstance(value, (dict, list)):
                            info_items.append({'Field': key, 'Value': str(value)})
                    
                    info_df = pd.DataFrame(info_items)
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        else:
            st.error(f"❌ Could not find data for ticker '{ticker_input}'. Please check the ticker symbol.")

elif not ticker_input and search_button:
    st.warning("⚠️ Please enter a ticker symbol")

# ===== INFO SECTION (si aucun ticker recherché) =====
if not search_button or not ticker_input:
    st.markdown("""
    <div style="background-color: #111; border: 2px solid #FFAA00; padding: 20px; margin: 20px 0;">
        <h3 style="margin: 0 0 15px 0; color: #FFAA00;">📊 COMPREHENSIVE COMPANY ANALYSIS</h3>
        <p style="font-size: 11px; color: #999; margin: 10px 0;">
        Enter any stock ticker to get instant access to:
        </p>
        <ul style="font-size: 10px; color: #999; margin: 10px 0 10px 20px;">
            <li><strong>Company Overview:</strong> Key metrics, description, executives</li>
            <li><strong>Financial Statements:</strong> Income statement, balance sheet, cash flow (annual & quarterly)</li>
            <li><strong>Price & Performance:</strong> Interactive charts, technical indicators, moving averages</li>
            <li><strong>Valuation:</strong> P/E, PEG, EV/EBITDA, analyst targets</li>
            <li><strong>News & Events:</strong> Latest news, earnings calendar, company events</li>
            <li><strong>Detailed Info:</strong> Ownership structure, institutional holders, ESG scores</li>
        </ul>
        <p style="font-size: 10px; color: #FFAA00; margin: 15px 0 0 0;">
        <strong>📌 SUPPORTED MARKETS:</strong>
        </p>
        <ul style="font-size: 9px; color: #999; margin: 5px 0 0 20px;">
            <li><strong>US:</strong> AAPL, MSFT, GOOGL, TSLA, NVDA...</li>
            <li><strong>Europe:</strong> MC.PA (LVMH), AIR.PA (Airbus), SAP.DE (SAP)...</li>
            <li><strong>Switzerland:</strong> NESN.SW (Nestlé), NOVN.SW (Novartis), ROG.SW (Roche)...</li>
            <li><strong>UK:</strong> SHEL.L (Shell), BP.L (BP), HSBA.L (HSBC)...</li>
        </ul>
        <p style="font-size: 10px; color: #00FFFF; margin: 15px 0 0 0; font-weight: bold;">
        💡 TIP: Use Yahoo Finance ticker format (add exchange suffix for non-US stocks)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Examples - Simple display
    st.markdown("#### 🔍 POPULAR TICKERS")
    st.markdown("""
    <div style="background-color: #0a0a0a; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0 0 10px 0; font-size: 10px; color: #FFAA00;">
        Try these popular tickers:
        </p>
        <ul style="margin: 0; font-size: 11px; color: #999; list-style: none; padding: 0;">
            <li style="margin: 5px 0;">🇺🇸 <strong style="color: #00FFFF;">AAPL</strong> - Apple Inc.</li>
            <li style="margin: 5px 0;">🇺🇸 <strong style="color: #00FFFF;">MSFT</strong> - Microsoft Corp.</li>
            <li style="margin: 5px 0;">🇺🇸 <strong style="color: #00FFFF;">TSLA</strong> - Tesla Inc.</li>
            <li style="margin: 5px 0;">🇫🇷 <strong style="color: #00FFFF;">MC.PA</strong> - LVMH</li>
            <li style="margin: 5px 0;">🇫🇷 <strong style="color: #00FFFF;">AIR.PA</strong> - Airbus</li>
            <li style="margin: 5px 0;">🇨🇭 <strong style="color: #00FFFF;">NESN.SW</strong> - Nestlé</li>
            <li style="margin: 5px 0;">🇨🇭 <strong style="color: #00FFFF;">NOVN.SW</strong> - Novartis</li>
            <li style="margin: 5px 0;">🇬🇧 <strong style="color: #00FFFF;">SHEL.L</strong> - Shell</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)    
    # Examples
    st.markdown("#### 🔍 EXAMPLE TICKERS")
    
    example_cols = st.columns(5)
    
    examples = [
        ('AAPL', 'Apple'),
        ('MSFT', 'Microsoft'),
        ('MC.PA', 'LVMH'),
        ('NESN.SW', 'Nestlé'),
        ('SHEL.L', 'Shell')
    ]
    
    for idx, (ticker, name) in enumerate(examples):
        with example_cols[idx]:
            if st.button(f"{ticker}\n{name}", use_container_width=True, key=f"example_{ticker}"):
                st.session_state['ticker_search'] = ticker
                st.rerun()

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | COMPANY ANALYSIS | LAST UPDATE: {last_update}
    <br>
    Data provided by Yahoo Finance
</div>
""", unsafe_allow_html=True)
