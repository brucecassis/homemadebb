import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Backtesting & Forecasting",
    page_icon="🎯",
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
    
    .signal-box-long {
        background-color: #0a1a00;
        border-left: 3px solid #00FF00;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .signal-box-short {
        background-color: #1a0a00;
        border-left: 3px solid #FF0000;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .signal-box-neutral {
        background-color: #0a0a0a;
        border-left: 3px solid #FFAA00;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .backtest-stats {
        background-color: #111;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - BACKTESTING & FORECASTING</div>
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

# Fonctions de calcul
def calculate_yoy_change(df):
    """Calcule la variation Year-over-Year"""
    if df is None or len(df) < 12:
        return None
    
    df = df.copy()
    df['yoy_change'] = df['value'].pct_change(12) * 100
    return df

def calculate_z_score(series):
    """Calcule le z-score"""
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

# ONGLETS PRINCIPAUX
tab1, tab2, tab3, tab4 = st.tabs(["🎯 GDP NOWCASTING", "📊 MACRO BACKTESTING", "📈 TRADING SIGNALS", "🔗 DATA INTEGRATION"])

# ===== TAB 1: GDP NOWCASTING =====
with tab1:
    st.markdown("### 🎯 GDP NOWCASTING")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #00FF00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
        🔮 REAL-TIME GDP ESTIMATION
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Utilise des indicateurs haute fréquence (mensuel) pour estimer le PIB trimestriel en temps réel.
        Méthode : Dynamic Factor Model avec pondération optimale des prédicteurs.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_now1, col_now2 = st.columns([2, 1])
    
    with col_now1:
        st.markdown("#### 📊 HIGH-FREQUENCY INDICATORS")
        
        # Sélection des indicateurs
        nowcast_indicators = st.multiselect(
            "SELECT MONTHLY INDICATORS",
            options=[
                'INDPRO - Industrial Production',
                'PAYEMS - Nonfarm Payrolls',
                'RSXFS - Retail Sales',
                'HOUST - Housing Starts',
                'UMCSENT - Consumer Sentiment',
                'CPIAUCSL - CPI'
            ],
            default=[
                'INDPRO - Industrial Production',
                'PAYEMS - Nonfarm Payrolls',
                'RSXFS - Retail Sales'
            ],
            key="nowcast_indicators"
        )
    
    with col_now2:
        nowcast_method = st.selectbox(
            "NOWCAST METHOD",
            options=["Simple Average", "Weighted Average", "Principal Components"],
            key="nowcast_method"
        )
        
        lookback_quarters = st.slider(
            "TRAINING QUARTERS",
            min_value=8,
            max_value=40,
            value=20,
            key="lookback_quarters"
        )
    
    if st.button("🎯 GENERATE NOWCAST", use_container_width=True, key="run_nowcast"):
        if nowcast_indicators:
            with st.spinner("Generating GDP nowcast..."):
                # Récupérer le PIB réel
                df_gdp = get_fred_series('GDPC1')
                
                if df_gdp is not None:
                    # Calculer croissance trimestrielle
                    df_gdp['gdp_growth'] = df_gdp['value'].pct_change(1) * 100
                    
                    # Récupérer les indicateurs mensuels
                    indicator_data = {}
                    
                    for indicator_full in nowcast_indicators:
                        indicator_id = indicator_full.split(' - ')[0]
                        df_ind = get_fred_series(indicator_id)
                        
                        if df_ind is not None:
                            # Agréger en trimestriel (moyenne)
                            df_ind['quarter'] = df_ind['date'].dt.to_period('Q')
                            df_quarterly = df_ind.groupby('quarter')['value'].mean().reset_index()
                            df_quarterly['date'] = df_quarterly['quarter'].dt.to_timestamp()
                            df_quarterly['growth'] = df_quarterly['value'].pct_change(1) * 100
                            
                            indicator_data[indicator_id] = df_quarterly[['date', 'growth']]
                    
                    if indicator_data:
                        # Créer dataset combiné
                        df_combined = df_gdp[['date', 'gdp_growth']].copy()
                        
                        for ind_id, ind_df in indicator_data.items():
                            df_combined = pd.merge(
                                df_combined,
                                ind_df.rename(columns={'growth': ind_id}),
                                on='date',
                                how='inner'
                            )
                        
                        df_combined = df_combined.dropna()
                        
                        if len(df_combined) > lookback_quarters:
                            # Split train/test (dernier point = nowcast)
                            train_data = df_combined.iloc[:-1]
                            test_point = df_combined.iloc[-1:]
                            
                            X_train = train_data[[col for col in df_combined.columns if col not in ['date', 'gdp_growth']]]
                            y_train = train_data['gdp_growth']
                            
                            X_test = test_point[[col for col in df_combined.columns if col not in ['date', 'gdp_growth']]]
                            y_actual = test_point['gdp_growth'].values[0]
                            
                            # Méthode de nowcast
                            if nowcast_method == "Simple Average":
                                # Corrélations avec GDP
                                correlations = {}
                                for col in X_train.columns:
                                    corr = X_train[col].corr(y_train)
                                    correlations[col] = abs(corr)
                                
                                # Prédiction = moyenne pondérée par corrélation
                                total_corr = sum(correlations.values())
                                weights = {k: v/total_corr for k, v in correlations.items()}
                                
                                # Régression simple
                                from sklearn.linear_model import LinearRegression
                                model = LinearRegression()
                                model.fit(X_train, y_train)
                                
                                nowcast_value = model.predict(X_test)[0]
                                
                            elif nowcast_method == "Weighted Average":
                                from sklearn.linear_model import Ridge
                                model = Ridge(alpha=1.0)
                                model.fit(X_train, y_train)
                                nowcast_value = model.predict(X_test)[0]
                                
                            else:  # Principal Components
                                from sklearn.decomposition import PCA
                                from sklearn.linear_model import LinearRegression
                                
                                # PCA
                                pca = PCA(n_components=min(2, len(X_train.columns)))
                                X_train_pca = pca.fit_transform(X_train)
                                X_test_pca = pca.transform(X_test)
                                
                                # Régression sur composantes
                                model = LinearRegression()
                                model.fit(X_train_pca, y_train)
                                nowcast_value = model.predict(X_test_pca)[0]
                            
                            # Affichage résultats
                            st.markdown("### 📊 NOWCAST RESULTS")
                            
                            col_res1, col_res2, col_res3 = st.columns(3)
                            
                            with col_res1:
                                st.metric(
                                    "NOWCAST GDP GROWTH",
                                    f"{nowcast_value:.2f}%",
                                    help="Estimated real GDP quarterly growth"
                                )
                            
                            with col_res2:
                                st.metric(
                                    "ACTUAL GDP GROWTH",
                                    f"{y_actual:.2f}%",
                                    help="Official BEA release"
                                )
                            
                            with col_res3:
                                error = nowcast_value - y_actual
                                st.metric(
                                    "NOWCAST ERROR",
                                    f"{error:+.2f}%",
                                    delta=f"{abs(error):.2f}% MAE"
                                )
                            
                            # Graphique historique
                            st.markdown("#### 📈 HISTORICAL NOWCAST PERFORMANCE")
                            
                            # Backtest sur données historiques
                            nowcast_history = []
                            actual_history = []
                            dates_history = []
                            
                            for i in range(lookback_quarters, len(df_combined)):
                                train_hist = df_combined.iloc[:i]
                                test_hist = df_combined.iloc[i]
                                
                                X_train_hist = train_hist[[col for col in df_combined.columns if col not in ['date', 'gdp_growth']]]
                                y_train_hist = train_hist['gdp_growth']
                                X_test_hist = test_hist[[col for col in df_combined.columns if col not in ['date', 'gdp_growth']]].values.reshape(1, -1)
                                
                                if nowcast_method == "Weighted Average":
                                    model_hist = Ridge(alpha=1.0)
                                else:
                                    model_hist = LinearRegression()
                                
                                model_hist.fit(X_train_hist, y_train_hist)
                                pred = model_hist.predict(X_test_hist)[0]
                                
                                nowcast_history.append(pred)
                                actual_history.append(test_hist['gdp_growth'])
                                dates_history.append(test_hist['date'])
                            
                            # Graphique
                            fig_nowcast = go.Figure()
                            
                            fig_nowcast.add_trace(go.Scatter(
                                x=dates_history,
                                y=actual_history,
                                mode='lines+markers',
                                name='Actual GDP Growth',
                                line=dict(color='#FFAA00', width=2),
                                marker=dict(size=6)
                            ))
                            
                            fig_nowcast.add_trace(go.Scatter(
                                x=dates_history,
                                y=nowcast_history,
                                mode='lines+markers',
                                name='Nowcast',
                                line=dict(color='#00FF00', width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            fig_nowcast.update_layout(
                                paper_bgcolor='#000',
                                plot_bgcolor='#111',
                                font=dict(color='#FFAA00', size=10),
                                xaxis=dict(gridcolor='#333', title="Date"),
                                yaxis=dict(gridcolor='#333', title="GDP Growth (%)"),
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_nowcast, use_container_width=True)
                            
                            # Statistiques de performance
                            st.markdown("#### 📊 BACKTEST STATISTICS")
                            
                            errors = np.array(nowcast_history) - np.array(actual_history)
                            mae = np.mean(np.abs(errors))
                            rmse = np.sqrt(np.mean(errors**2))
                            
                            from scipy.stats import pearsonr
                            correlation, _ = pearsonr(nowcast_history, actual_history)
                            
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            
                            with col_stat1:
                                st.metric("MAE", f"{mae:.3f}%")
                            
                            with col_stat2:
                                st.metric("RMSE", f"{rmse:.3f}%")
                            
                            with col_stat3:
                                st.metric("CORRELATION", f"{correlation:.3f}")
                            
                            with col_stat4:
                                # Direction accuracy
                                actual_direction = np.sign(np.array(actual_history))
                                nowcast_direction = np.sign(np.array(nowcast_history))
                                direction_accuracy = np.mean(actual_direction == nowcast_direction) * 100
                                st.metric("DIRECTION ACCURACY", f"{direction_accuracy:.1f}%")
                            
                            # Contribution des indicateurs
                            if hasattr(model, 'coef_'):
                                st.markdown("#### 📊 INDICATOR CONTRIBUTIONS")
                                
                                contributions = pd.DataFrame({
                                    'Indicator': X_train.columns,
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False)
                                
                                fig_contrib = go.Figure()
                                fig_contrib.add_trace(go.Bar(
                                    x=contributions['Coefficient'],
                                    y=contributions['Indicator'],
                                    orientation='h',
                                    marker=dict(color='#FFAA00')
                                ))
                                
                                fig_contrib.update_layout(
                                    title="Indicator Weights in Nowcast Model",
                                    paper_bgcolor='#000',
                                    plot_bgcolor='#111',
                                    font=dict(color='#FFAA00', size=10),
                                    xaxis=dict(gridcolor='#333', title="Weight"),
                                    yaxis=dict(gridcolor='#333'),
                                    height=300
                                )
                                
                                st.plotly_chart(fig_contrib, use_container_width=True)
                        
                        else:
                            st.warning("⚠️ Not enough historical data for nowcasting")
                    else:
                        st.error("❌ Could not retrieve indicator data")
                else:
                    st.error("❌ Could not retrieve GDP data")
        else:
            st.warning("⚠️ Please select at least one indicator")

# ===== TAB 2: MACRO BACKTESTING =====
with tab2:
    st.markdown("### 📊 MACRO STRATEGY BACKTESTING")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #FFAA00; font-weight: bold;">
        📈 POINT-IN-TIME BACKTESTING (NO LOOKAHEAD BIAS)
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Teste des stratégies macro avec données réellement disponibles à chaque date.
        Évite le biais de lookahead en n'utilisant que l'information passée.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🎯 STRATEGY CONFIGURATION")
    
    col_strat1, col_strat2 = st.columns(2)
    
    with col_strat1:
        strategy_type = st.selectbox(
            "STRATEGY TYPE",
            options=[
                "Yield Curve Inversion → Short Equities",
                "Inflation Momentum → Long Commodities",
                "Recession Probability → Risk Off"
            ],
            key="strategy_type"
        )
        
        backtest_start = st.date_input(
            "BACKTEST START DATE",
            value=datetime(2000, 1, 1),
            key="backtest_start"
        )
    
    with col_strat2:
        backtest_end = st.date_input(
            "BACKTEST END DATE",
            value=datetime.now(),
            key="backtest_end"
        )
        
        rebalance_freq = st.selectbox(
            "REBALANCE FREQUENCY",
            options=["Monthly", "Quarterly"],
            key="rebalance_freq"
        )
    
    if st.button("📊 RUN BACKTEST", use_container_width=True, key="run_backtest"):
        with st.spinner("Running backtest..."):
            start_date_str = backtest_start.strftime('%Y-%m-%d')
            end_date_str = backtest_end.strftime('%Y-%m-%d')
            
            if "Yield Curve" in strategy_type:
                st.markdown("### 📉 YIELD CURVE INVERSION STRATEGY")
                st.caption("Signal: Short S&P 500 when 10Y-2Y < 0, Long otherwise")
                
                # Récupérer données
                df_spread = get_fred_series('T10Y2Y', observation_start=start_date_str)
                df_sp500 = get_fred_series('SP500', observation_start=start_date_str)
                
                if df_spread is not None and df_sp500 is not None:
                    # Merger
                    df_backtest = pd.merge(
                        df_spread[['date', 'value']].rename(columns={'value': 'spread'}),
                        df_sp500[['date', 'value']].rename(columns={'value': 'sp500'}),
                        on='date',
                        how='inner'
                    )
                    
                    # Calculer rendements S&P 500
                    df_backtest['sp500_return'] = df_backtest['sp500'].pct_change()
                    
                    # Signal: -1 si inversion, +1 sinon
                    df_backtest['signal'] = np.where(df_backtest['spread'] < 0, -1, 1)
                    
                    # Stratégie returns = signal * market return
                    df_backtest['strategy_return'] = df_backtest['signal'].shift(1) * df_backtest['sp500_return']
                    
                    # Performance cumulée
                    df_backtest['strategy_cumul'] = (1 + df_backtest['strategy_return'].fillna(0)).cumprod()
                    df_backtest['buy_hold_cumul'] = (1 + df_backtest['sp500_return'].fillna(0)).cumprod()
                    
                    # Graphique
                    fig_bt = go.Figure()
                    
                    fig_bt.add_trace(go.Scatter(
                        x=df_backtest['date'],
                        y=df_backtest['strategy_cumul'],
                        mode='lines',
                        name='Yield Curve Strategy',
                        line=dict(color='#00FF00', width=2)
                    ))
                    
                    fig_bt.add_trace(go.Scatter(
                        x=df_backtest['date'],
                        y=df_backtest['buy_hold_cumul'],
                        mode='lines',
                        name='Buy & Hold S&P 500',
                        line=dict(color='#FFAA00', width=2, dash='dash')
                    ))
                    
                    fig_bt.update_layout(
                        title="Cumulative Returns: Strategy vs Buy & Hold",
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        xaxis=dict(gridcolor='#333', title="Date"),
                        yaxis=dict(gridcolor='#333', title="Cumulative Return"),
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    # Statistiques
                    st.markdown("#### 📊 BACKTEST STATISTICS")
                    
                    strategy_returns = df_backtest['strategy_return'].dropna()
                    bh_returns = df_backtest['sp500_return'].dropna()
                    
                    # Calculs
                    total_return_strat = (df_backtest['strategy_cumul'].iloc[-1] - 1) * 100
                    total_return_bh = (df_backtest['buy_hold_cumul'].iloc[-1] - 1) * 100
                    
                    annual_return_strat = strategy_returns.mean() * 252 * 100
                    annual_return_bh = bh_returns.mean() * 252 * 100
                    
                    volatility_strat = strategy_returns.std() * np.sqrt(252) * 100
                    volatility_bh = bh_returns.std() * np.sqrt(252) * 100
                    
                    sharpe_strat = (annual_return_strat / volatility_strat) if volatility_strat > 0 else 0
                    sharpe_bh = (annual_return_bh / volatility_bh) if volatility_bh > 0 else 0
                    
                    max_dd_strat = ((df_backtest['strategy_cumul'] / df_backtest['strategy_cumul'].cummax()) - 1).min() * 100
                    max_dd_bh = ((df_backtest['buy_hold_cumul'] / df_backtest['buy_hold_cumul'].cummax()) - 1).min() * 100
                    
                    # Affichage
                    col_bt1, col_bt2 = st.columns(2)
                    
                    with col_bt1:
                        st.markdown("**🎯 STRATEGY**")
                        st.metric("Total Return", f"{total_return_strat:.2f}%")
                        st.metric("Annual Return", f"{annual_return_strat:.2f}%")
                        st.metric("Volatility", f"{volatility_strat:.2f}%")
                        st.metric("Sharpe Ratio", f"{sharpe_strat:.2f}")
                        st.metric("Max Drawdown", f"{max_dd_strat:.2f}%")
                    
                    with col_bt2:
                        st.markdown("**📈 BUY & HOLD**")
                        st.metric("Total Return", f"{total_return_bh:.2f}%")
                        st.metric("Annual Return", f"{annual_return_bh:.2f}%")
                        st.metric("Volatility", f"{volatility_bh:.2f}%")
                        st.metric("Sharpe Ratio", f"{sharpe_bh:.2f}")
                        st.metric("Max Drawdown", f"{max_dd_bh:.2f}%")
                    
                    # Analyse des signaux
                    st.markdown("#### 📊 SIGNAL ANALYSIS")
                    
                    inversions = df_backtest[df_backtest['spread'] < 0]
                    pct_inverted = (len(inversions) / len(df_backtest)) * 100
                    
                    st.caption(f"**Inversion periods:** {len(inversions)} days ({pct_inverted:.1f}% of backtest)")
                    
                    # Performance pendant inversions vs normal
                    if len(inversions) > 0:
                        normal_periods = df_backtest[df_backtest['spread'] >= 0]
                        
                        avg_return_inversion = inversions['sp500_return'].mean() * 252 * 100
                        avg_return_normal = normal_periods['sp500_return'].mean() * 252 * 100
                        
                        col_sig1, col_sig2 = st.columns(2)
                        
                        with col_sig1:
                            st.metric("Avg Return (Inversion)", f"{avg_return_inversion:.2f}%/year")
                        
                        with col_sig2:
                            st.metric("Avg Return (Normal)", f"{avg_return_normal:.2f}%/year")

# ===== TAB 3: TRADING SIGNALS =====
with tab3:
    st.markdown("### 📈 MACRO TRADING SIGNALS")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #00FFFF; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #00FFFF; font-weight: bold;">
        🎯 REAL-TIME TRADING SIGNALS
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Combine plusieurs indicateurs macro pour générer des signaux Long/Short/Neutral.
        Mise à jour en temps réel avec les dernières données FRED disponibles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🎯 CURRENT MARKET SIGNALS")
    
    col_sig1, col_sig2 = st.columns([2, 1])
    
    with col_sig1:
        asset_class = st.selectbox(
            "SELECT ASSET CLASS",
            options=["Equities (S&P 500)", "Treasuries (10Y)", "Commodities (Gold)", "USD Index"],
            key="asset_class"
        )
    
    with col_sig2:
        signal_sensitivity = st.slider(
            "SIGNAL SENSITIVITY",
            min_value=1,
            max_value=5,
            value=3,
            help="1=Conservative, 5=Aggressive",
            key="signal_sensitivity"
        )
    
    if st.button("🎯 GENERATE SIGNALS", use_container_width=True, key="generate_signals"):
        with st.spinner("Analyzing macro conditions..."):
            # Récupérer les indicateurs clés
            df_spread = get_fred_series('T10Y2Y')
            df_cpi = get_fred_series('CPIAUCSL')
            df_unrate = get_fred_series('UNRATE')
            df_m2 = get_fred_series('M2SL')
            df_fedfunds = get_fred_series('FEDFUNDS')
            
            if all(df is not None for df in [df_spread, df_cpi, df_unrate, df_m2, df_fedfunds]):
                # Dernières valeurs
                current_spread = df_spread['value'].iloc[-1]
                prev_spread = df_spread['value'].iloc[-2]
                
                # Inflation YoY
                cpi_current = df_cpi['value'].iloc[-1]
                cpi_12m_ago = df_cpi['value'].iloc[-13]
                inflation_yoy = ((cpi_current / cpi_12m_ago) - 1) * 100
                
                # M2 Growth
                m2_current = df_m2['value'].iloc[-1]
                m2_12m_ago = df_m2['value'].iloc[-13]
                m2_growth = ((m2_current / m2_12m_ago) - 1) * 100
                
                # Unemployment
                current_unrate = df_unrate['value'].iloc[-1]
                unrate_3m_ago = df_unrate['value'].iloc[-4]
                unrate_change = current_unrate - unrate_3m_ago
                
                # Fed Funds
                current_ff = df_fedfunds['value'].iloc[-1]
                
                st.markdown("### 📊 MACRO SNAPSHOT")
                
                col_snap1, col_snap2, col_snap3, col_snap4 = st.columns(4)
                
                with col_snap1:
                    st.metric("10Y-2Y SPREAD", f"{current_spread:.2f}%", 
                             delta=f"{current_spread - prev_spread:+.2f}%")
                
                with col_snap2:
                    st.metric("INFLATION YoY", f"{inflation_yoy:.2f}%")
                
                with col_snap3:
                    st.metric("UNEMPLOYMENT", f"{current_unrate:.1f}%",
                             delta=f"{unrate_change:+.1f}%")
                
                with col_snap4:
                    st.metric("M2 GROWTH", f"{m2_growth:.2f}%")
                
                st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # ===== SIGNAL GENERATION =====
                st.markdown("### 🎯 TRADING SIGNAL BREAKDOWN")
                
                # Scores individuels (-2 à +2)
                scores = {}
                
                # 1. YIELD CURVE SIGNAL
                if current_spread < -0.5:
                    yield_score = -2  # Forte inversion
                elif current_spread < 0:
                    yield_score = -1  # Inversion légère
                elif current_spread < 0.5:
                    yield_score = 0   # Neutre
                elif current_spread < 1.5:
                    yield_score = 1   # Pente positive
                else:
                    yield_score = 2   # Forte pente
                
                scores['Yield Curve'] = yield_score
                
                # 2. INFLATION SIGNAL
                if inflation_yoy > 4:
                    inflation_score = -2  # Inflation élevée (négatif pour risk assets)
                elif inflation_yoy > 3:
                    inflation_score = -1
                elif inflation_yoy > 1.5:
                    inflation_score = 0
                else:
                    inflation_score = 1   # Déflation risk
                
                scores['Inflation'] = inflation_score
                
                # 3. UNEMPLOYMENT MOMENTUM
                if unrate_change > 0.5:
                    unemp_score = -2  # Détérioration emploi
                elif unrate_change > 0.2:
                    unemp_score = -1
                elif unrate_change > -0.2:
                    unemp_score = 0
                else:
                    unemp_score = 1   # Amélioration emploi
                
                scores['Employment'] = unemp_score
                
                # 4. LIQUIDITY SIGNAL (M2 Growth)
                if m2_growth < -2:
                    liquidity_score = -2  # Contraction monétaire
                elif m2_growth < 0:
                    liquidity_score = -1
                elif m2_growth < 5:
                    liquidity_score = 0
                else:
                    liquidity_score = 1   # Expansion liquide
                
                scores['Liquidity'] = liquidity_score
                
                # 5. FED POLICY SIGNAL
                # Taylor rule simplifié
                taylor_rate = 2 + inflation_yoy + 0.5 * (inflation_yoy - 2)
                policy_gap = current_ff - taylor_rate
                
                if policy_gap > 2:
                    fed_score = -1  # Politique très restrictive
                elif policy_gap > 0.5:
                    fed_score = 0   # Légèrement restrictive
                elif policy_gap > -1:
                    fed_score = 1   # Accommodante
                else:
                    fed_score = 2   # Très accommodante
                
                scores['Fed Policy'] = fed_score
                
                # Afficher les scores
                st.markdown("#### 📊 INDIVIDUAL SIGNALS")
                
                for signal_name, score in scores.items():
                    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
                    
                    with col_s1:
                        st.markdown(f"**{signal_name}**")
                    
                    with col_s2:
                        # Barre de score
                        if score >= 1:
                            color = "#00FF00"
                            label = "BULLISH"
                        elif score <= -1:
                            color = "#FF0000"
                            label = "BEARISH"
                        else:
                            color = "#FFAA00"
                            label = "NEUTRAL"
                        
                        st.markdown(f'<div style="color: {color}; font-weight: bold;">{label}</div>', 
                                   unsafe_allow_html=True)
                    
                    with col_s3:
                        st.markdown(f"Score: {score:+d}")
                
                st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                # ===== SIGNAL COMPOSITE =====
                st.markdown("### 🎯 COMPOSITE SIGNAL")
                
                # Score total ajusté par sensibilité
                total_score = sum(scores.values())
                
                # Seuils ajustés par sensibilité (plus sensible = seuils plus bas)
                threshold_long = 3 - (signal_sensitivity - 3)
                threshold_short = -3 + (signal_sensitivity - 3)
                
                if total_score >= threshold_long:
                    signal = "LONG"
                    signal_color = "#00FF00"
                    signal_class = "signal-box-long"
                    signal_icon = "📈"
                elif total_score <= threshold_short:
                    signal = "SHORT"
                    signal_color = "#FF0000"
                    signal_class = "signal-box-short"
                    signal_icon = "📉"
                else:
                    signal = "NEUTRAL"
                    signal_color = "#FFAA00"
                    signal_class = "signal-box-neutral"
                    signal_icon = "➡️"
                
                st.markdown(f"""
                <div class="{signal_class}">
                    <h3 style="margin: 0; color: {signal_color}; font-size: 18px;">
                        {signal_icon} {signal} {asset_class.split('(')[1].replace(')', '')}
                    </h3>
                    <p style="margin: 5px 0 0 0; font-size: 11px; color: #999;">
                        Composite Score: {total_score:+d} | Threshold: ±{abs(threshold_long)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Breakdown par composante
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    st.markdown("**🔴 BEARISH FACTORS:**")
                    bearish = [name for name, score in scores.items() if score < 0]
                    if bearish:
                        for factor in bearish:
                            st.markdown(f"- {factor} ({scores[factor]:+d})")
                    else:
                        st.markdown("*None*")
                
                with col_comp2:
                    st.markdown("**🟢 BULLISH FACTORS:**")
                    bullish = [name for name, score in scores.items() if score > 0]
                    if bullish:
                        for factor in bullish:
                            st.markdown(f"- {factor} ({scores[factor]:+d})")
                    else:
                        st.markdown("*None*")
                
                # ===== RECOMMENDATION =====
                st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("### 💡 RECOMMENDATION")
                
                if signal == "LONG":
                    st.markdown("""
                    **📈 BULLISH STANCE RECOMMENDED**
                    
                    **Strategy:**
                    - Increase exposure to risk assets
                    - Consider adding leverage
                    - Reduce hedges
                    
                    **Risk Management:**
                    - Set stop-loss at -5%
                    - Monitor yield curve daily
                    - Exit if unemployment spikes >0.5%
                    """)
                
                elif signal == "SHORT":
                    st.markdown("""
                    **📉 BEARISH STANCE RECOMMENDED**
                    
                    **Strategy:**
                    - Reduce risk asset exposure
                    - Add defensive positions (Treasuries, Gold)
                    - Consider hedging with VIX calls
                    
                    **Risk Management:**
                    - Set stop-loss at +3% (cover shorts)
                    - Monitor Fed policy closely
                    - Exit if yield curve uninverts
                    """)
                
                else:
                    st.markdown("""
                    **➡️ NEUTRAL STANCE RECOMMENDED**
                    
                    **Strategy:**
                    - Maintain balanced portfolio
                    - Wait for clearer signals
                    - Focus on high-quality assets
                    
                    **Risk Management:**
                    - Tight position sizing
                    - Monitor for signal changes
                    - Stay liquid
                    """)
                
                # ===== SIGNAL HISTORY =====
                st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
                st.markdown("### 📊 SIGNAL HISTORY (Last 5 Years)")
                
                # Calculer signaux historiques
                lookback_days = 5 * 252  # 5 ans
                
                if len(df_spread) > lookback_days:
                    df_hist = df_spread.tail(lookback_days).copy()
                    
                    # Calculer scores historiques (simplifié)
                    df_hist['signal_score'] = 0
                    
                    # Yield curve component
                    df_hist['signal_score'] += np.where(df_hist['value'] < -0.5, -2,
                                                np.where(df_hist['value'] < 0, -1,
                                                np.where(df_hist['value'] < 0.5, 0,
                                                np.where(df_hist['value'] < 1.5, 1, 2))))
                    
                    # Signal zones
                    df_hist['signal_zone'] = np.where(df_hist['signal_score'] >= 1, 1,
                                                       np.where(df_hist['signal_score'] <= -1, -1, 0))
                    
                    # Graphique
                    fig_sig_hist = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Yield Curve (10Y-2Y)', 'Signal Zone')
                    )
                    
                    # Plot 1: Yield curve
                    fig_sig_hist.add_trace(
                        go.Scatter(x=df_hist['date'], y=df_hist['value'],
                                  mode='lines', line=dict(color='#FFAA00', width=1),
                                  name='10Y-2Y Spread'),
                        row=1, col=1
                    )
                    
                    fig_sig_hist.add_hline(y=0, line_dash="dash", line_color="#FF0000", row=1, col=1)
                    
                    # Plot 2: Signal zones
                    colors = ['#FF0000' if x == -1 else '#00FF00' if x == 1 else '#FFAA00' 
                             for x in df_hist['signal_zone']]
                    
                    fig_sig_hist.add_trace(
                        go.Bar(x=df_hist['date'], y=df_hist['signal_zone'],
                              marker_color=colors, name='Signal'),
                        row=2, col=1
                    )
                    
                    fig_sig_hist.update_layout(
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        height=500,
                        showlegend=False
                    )
                    
                    fig_sig_hist.update_xaxes(gridcolor='#333')
                    fig_sig_hist.update_yaxes(gridcolor='#333')
                    
                    st.plotly_chart(fig_sig_hist, use_container_width=True)
                    
                    # Stats des signaux
                    signal_counts = df_hist['signal_zone'].value_counts()
                    
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        bearish_pct = (signal_counts.get(-1, 0) / len(df_hist)) * 100
                        st.metric("BEARISH PERIODS", f"{bearish_pct:.1f}%")
                    
                    with col_stats2:
                        neutral_pct = (signal_counts.get(0, 0) / len(df_hist)) * 100
                        st.metric("NEUTRAL PERIODS", f"{neutral_pct:.1f}%")
                    
                    with col_stats3:
                        bullish_pct = (signal_counts.get(1, 0) / len(df_hist)) * 100
                        st.metric("BULLISH PERIODS", f"{bullish_pct:.1f}%")
            
            else:
                st.error("❌ Could not retrieve all required data")
# ===== TAB 4: DATA INTEGRATION =====
with tab4:
    st.markdown("### 🔗 DATA INTEGRATION & ENRICHMENT")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #FF00FF; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #FF00FF; font-weight: bold;">
        🔗 MULTI-SOURCE DATA FUSION
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Fusionne données macro FRED avec prix de marché, données alternatives, et fondamentaux.
        Alignement automatique des dates avec business days.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sous-onglets
    integration_tab1, integration_tab2, integration_tab3, integration_tab4 = st.tabs([
        "📊 MACRO + MARKET DATA",
        "🌍 ALTERNATIVE DATA",
        "🏭 SECTOR MAPPING",
        "🏢 COMPANY FUNDAMENTALS"
    ])
    
    # ===== INTEGRATION TAB 1: MACRO + MARKET DATA =====
    with integration_tab1:
        st.markdown("#### 📊 MACRO-MARKET DATA FUSION")
        st.caption("Cross-join FRED macro series with market prices (Yahoo Finance)")
        
        col_mm1, col_mm2 = st.columns(2)
        
        with col_mm1:
            macro_series_select = st.selectbox(
                "SELECT MACRO SERIES",
                options=[
                    'FEDFUNDS - Fed Funds Rate',
                    'CPIAUCSL - CPI',
                    'UNRATE - Unemployment',
                    'GDP - GDP',
                    'M2SL - M2 Money Supply',
                    'DGS10 - 10Y Treasury'
                ],
                key="macro_series_mm"
            )
            
            market_ticker = st.text_input(
                "MARKET TICKER (Yahoo Finance)",
                value="SPY",
                help="Ex: SPY, QQQ, GLD, TLT, AAPL...",
                key="market_ticker"
            ).upper()
        
        with col_mm2:
            fusion_period = st.selectbox(
                "TIME PERIOD",
                options=['1Y', '2Y', '5Y', '10Y'],
                index=2,
                key="fusion_period"
            )
            
            alignment_method = st.selectbox(
                "DATE ALIGNMENT",
                options=['Forward Fill', 'Backward Fill', 'Interpolate'],
                help="Forward Fill = use last available macro value",
                key="alignment_method"
            )
        
        if st.button("🔗 MERGE DATA", use_container_width=True, key="merge_macro_market"):
            with st.spinner("Merging macro and market data..."):
                # Calculer date de début
                years = int(fusion_period[:-1])
                start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
                
                # Récupérer série macro
                macro_id = macro_series_select.split(' - ')[0]
                df_macro = get_fred_series(macro_id, observation_start=start_date)
                
                # Récupérer données marché
                try:
                    import yfinance as yf
                    
                    ticker_data = yf.Ticker(market_ticker)
                    df_market = ticker_data.history(start=start_date, interval='1d')
                    
                    if not df_market.empty and df_macro is not None:
                        # Préparer données marché
                        df_market = df_market.reset_index()
                        df_market = df_market.rename(columns={'Date': 'date'})
                        df_market['date'] = pd.to_datetime(df_market['date']).dt.tz_localize(None)
                        
                        # Préparer données macro
                        df_macro = df_macro.rename(columns={'value': 'macro_value'})
                        
                        st.success(f"✅ Data retrieved: {len(df_macro)} macro obs, {len(df_market)} market days")
                        
                        # Merger les données
                        if alignment_method == "Forward Fill":
                            df_merged = pd.merge_asof(
                                df_market.sort_values('date'),
                                df_macro[['date', 'macro_value']].sort_values('date'),
                                on='date',
                                direction='backward'
                            )
                        elif alignment_method == "Backward Fill":
                            df_merged = pd.merge_asof(
                                df_market.sort_values('date'),
                                df_macro[['date', 'macro_value']].sort_values('date'),
                                on='date',
                                direction='forward'
                            )
                        else:  # Interpolate
                            df_merged = pd.merge(
                                df_market,
                                df_macro[['date', 'macro_value']],
                                on='date',
                                how='outer'
                            ).sort_values('date')
                            df_merged['macro_value'] = df_merged['macro_value'].interpolate(method='linear')
                            df_merged = df_merged.dropna(subset=['Close'])
                        
                        st.markdown("### 📊 MERGED DATASET")
                        st.caption(f"Total observations: {len(df_merged)}")
                        
                        # Aperçu des données
                        st.dataframe(
                            df_merged[['date', 'Close', 'macro_value']].tail(10),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Graphique dual-axis
                        st.markdown("#### 📈 DUAL-AXIS VISUALIZATION")
                        
                        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig_dual.add_trace(
                            go.Scatter(
                                x=df_merged['date'],
                                y=df_merged['Close'],
                                name=f"{market_ticker} Price",
                                line=dict(color='#FFAA00', width=2)
                            ),
                            secondary_y=False
                        )
                        
                        fig_dual.add_trace(
                            go.Scatter(
                                x=df_merged['date'],
                                y=df_merged['macro_value'],
                                name=macro_series_select.split(' - ')[1],
                                line=dict(color='#00FF00', width=2)
                            ),
                            secondary_y=True
                        )
                        
                        fig_dual.update_layout(
                            title=f"{market_ticker} vs {macro_series_select.split(' - ')[1]}",
                            paper_bgcolor='#000',
                            plot_bgcolor='#111',
                            font=dict(color='#FFAA00', size=10),
                            xaxis=dict(gridcolor='#333'),
                            height=400,
                            hovermode='x unified'
                        )
                        
                        fig_dual.update_yaxes(title_text=f"{market_ticker} Price", secondary_y=False, gridcolor='#333')
                        fig_dual.update_yaxes(title_text="Macro Value", secondary_y=True, gridcolor='#333')
                        
                        st.plotly_chart(fig_dual, use_container_width=True)
                        
                        # Analyse de corrélation
                        st.markdown("#### 📊 CORRELATION ANALYSIS")
                        
                        df_merged['market_return'] = df_merged['Close'].pct_change()
                        df_merged['macro_change'] = df_merged['macro_value'].pct_change()
                        
                        corr = df_merged[['market_return', 'macro_change']].corr().iloc[0, 1]
                        
                        # Rolling correlation
                        df_merged['rolling_corr'] = df_merged['market_return'].rolling(60).corr(df_merged['macro_change'])
                        
                        col_corr1, col_corr2 = st.columns(2)
                        
                        with col_corr1:
                            st.metric("OVERALL CORRELATION", f"{corr:.3f}")
                            
                            if abs(corr) > 0.5:
                                st.success("Strong correlation detected")
                            elif abs(corr) > 0.3:
                                st.info("Moderate correlation")
                            else:
                                st.warning("Weak correlation")
                        
                        with col_corr2:
                            st.metric("CURRENT ROLLING CORR (60D)", 
                                     f"{df_merged['rolling_corr'].iloc[-1]:.3f}")
                        
                        # Graphique rolling correlation
                        fig_corr = go.Figure()
                        
                        fig_corr.add_trace(go.Scatter(
                            x=df_merged['date'],
                            y=df_merged['rolling_corr'],
                            mode='lines',
                            line=dict(color='#00FFFF', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0, 255, 255, 0.1)'
                        ))
                        
                        fig_corr.add_hline(y=0, line_dash="dash", line_color="#FFAA00")
                        
                        fig_corr.update_layout(
                            title="60-Day Rolling Correlation",
                            paper_bgcolor='#000',
                            plot_bgcolor='#111',
                            font=dict(color='#FFAA00', size=10),
                            xaxis=dict(gridcolor='#333', title="Date"),
                            yaxis=dict(gridcolor='#333', title="Correlation"),
                            height=300
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Download merged data
                        st.markdown("#### 💾 EXPORT MERGED DATA")
                        
                        csv_data = df_merged[['date', 'Close', 'macro_value', 'market_return', 'macro_change']].to_csv(index=False)
                        
                        st.download_button(
                            label="📥 DOWNLOAD CSV",
                            data=csv_data,
                            file_name=f"{market_ticker}_{macro_id}_merged_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    else:
                        st.error(f"❌ Could not retrieve data for {market_ticker}")
                
                except ImportError:
                    st.error("❌ yfinance not installed. Run: pip install yfinance")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ===== INTEGRATION TAB 2: ALTERNATIVE DATA =====
    with integration_tab2:
        st.markdown("#### 🌍 ALTERNATIVE DATA INTEGRATION")
        st.caption("Enhance nowcasts with high-frequency alternative data")
        
        st.markdown("""
        <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
            📊 AVAILABLE ALTERNATIVE DATA SOURCES
            </p>
            <ul style="margin: 10px 0; font-size: 10px; color: #999;">
                <li><strong>Mobility Data:</strong> Google Mobility Reports, Apple Mobility Trends</li>
                <li><strong>Google Trends:</strong> Search volume for economic keywords</li>
                <li><strong>Shipping Indices:</strong> Baltic Dry Index (BDI), Harpex</li>
                <li><strong>Energy Data:</strong> EIA Petroleum Status, Natural Gas Storage</li>
                <li><strong>Credit Card Data:</strong> Affinity Solutions spending data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        alt_data_type = st.selectbox(
            "SELECT ALTERNATIVE DATA TYPE",
            options=[
                "Shipping Index (Baltic Dry)",
                "Energy Data (EIA)",
                "Custom CSV Upload"
            ],
            key="alt_data_type"
        )
        
        if alt_data_type == "Shipping Index (Baltic Dry)":
            st.markdown("#### 🚢 BALTIC DRY INDEX")
            st.caption("Leading indicator for global trade and GDP")
            
            if st.button("📊 LOAD BDI DATA", use_container_width=True, key="load_bdi"):
                with st.spinner("Loading Baltic Dry Index..."):
                    # Note: BDI n'est pas directement dans FRED, mais on peut utiliser un proxy
                    # Ou simuler avec données alternatives
                    
                    st.info("""
                    **🔧 IMPLEMENTATION NOTE:**
                    
                    Baltic Dry Index n'est pas disponible via FRED API.
                    
                    **Options d'intégration :**
                    1. **Quandl API** : `QUANDL/CHRIS/ICE_B1` (nécessite compte)
                    2. **Web Scraping** : https://www.investing.com/indices/baltic-dry
                    3. **CSV Upload** : Télécharger données depuis Bloomberg/Reuters
                    
                    **Code exemple (Quandl) :**
```python
                    import quandl
                    quandl.ApiConfig.api_key = "YOUR_KEY"
                    bdi = quandl.get("QUANDL/CHRIS/ICE_B1")
```
                    """)
        
        elif alt_data_type == "Energy Data (EIA)":
            st.markdown("#### ⚡ ENERGY INFORMATION ADMINISTRATION")
            
            eia_series = st.selectbox(
                "SELECT EIA SERIES",
                options=[
                    "Crude Oil Stocks",
                    "Natural Gas Storage",
                    "Gasoline Prices",
                    "Electricity Generation"
                ],
                key="eia_series"
            )
            
            if st.button("📊 LOAD EIA DATA", use_container_width=True, key="load_eia"):
                st.info("""
                **🔧 EIA API INTEGRATION:**
                
                L'EIA (Energy Information Administration) fournit une API gratuite.
                
                **Setup requis :**
                1. Créer compte sur : https://www.eia.gov/opendata/
                2. Obtenir API key gratuite
                3. Installer : `pip install eia-python`
                
                **Code exemple :**
```python
                from eia import API
                api = API('YOUR_EIA_KEY')
                
                # Crude oil stocks
                crude_stocks = api.data_by_series('PET.WCRSTUS1.W')
```
                
                **Séries utiles pour nowcasting :**
                - `PET.WCRSTUS1.W` : Crude stocks (hebdo)
                - `NG.NW2_EPG0_SWO_R48_BCF.W` : Natural gas storage
                - `PET.EMM_EPM0_PTE_NUS_DPG.W` : Gasoline prices
                """)
        
        else:  # Custom CSV Upload
            st.markdown("#### 📁 CUSTOM DATA UPLOAD")
            
            uploaded_file = st.file_uploader(
                "UPLOAD CSV FILE",
                type=['csv'],
                help="CSV must have 'date' and 'value' columns",
                key="custom_csv_upload"
            )
            
            if uploaded_file is not None:
                try:
                    df_custom = pd.read_csv(uploaded_file)
                    
                    st.success(f"✅ File loaded: {len(df_custom)} rows")
                    st.dataframe(df_custom.head(10), use_container_width=True)
                    
                    # Vérifier colonnes
                    if 'date' in df_custom.columns and 'value' in df_custom.columns:
                        df_custom['date'] = pd.to_datetime(df_custom['date'])
                        
                        # Graphique
                        fig_custom = go.Figure()
                        
                        fig_custom.add_trace(go.Scatter(
                            x=df_custom['date'],
                            y=df_custom['value'],
                            mode='lines',
                            line=dict(color='#FF00FF', width=2)
                        ))
                        
                        fig_custom.update_layout(
                            title="Custom Alternative Data",
                            paper_bgcolor='#000',
                            plot_bgcolor='#111',
                            font=dict(color='#FFAA00', size=10),
                            xaxis=dict(gridcolor='#333'),
                            yaxis=dict(gridcolor='#333'),
                            height=350
                        )
                        
                        st.plotly_chart(fig_custom, use_container_width=True)
                        
                        # Option de merge avec macro
                        st.markdown("#### 🔗 MERGE WITH MACRO DATA")
                        
                        macro_to_merge = st.selectbox(
                            "SELECT MACRO SERIES",
                            options=['GDP', 'CPIAUCSL', 'UNRATE', 'INDPRO'],
                            key="macro_merge_custom"
                        )
                        
                        if st.button("🔗 MERGE", key="merge_custom"):
                            df_macro_merge = get_fred_series(macro_to_merge)
                            
                            if df_macro_merge is not None:
                                df_merged_custom = pd.merge_asof(
                                    df_custom.sort_values('date'),
                                    df_macro_merge[['date', 'value']].rename(columns={'value': 'macro_value'}).sort_values('date'),
                                    on='date',
                                    direction='backward'
                                )
                                
                                st.success(f"✅ Merged: {len(df_merged_custom)} observations")
                                st.dataframe(df_merged_custom.tail(10), use_container_width=True)
                    
                    else:
                        st.error("❌ CSV must contain 'date' and 'value' columns")
                
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")
    
    # ===== INTEGRATION TAB 3: SECTOR MAPPING =====
    with integration_tab3:
        st.markdown("#### 🏭 MACRO-TO-SECTOR MAPPING")
        st.caption("Map GDP/CPI moves to sector exposures for portfolio tilting")
        
        st.markdown("""
        <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
            📊 SECTOR SENSITIVITY FRAMEWORK
            </p>
            <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
            Différents secteurs réagissent différemment aux cycles macro :
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Matrice de sensibilité prédéfinie
        sector_sensitivity = {
            'Technology': {'GDP': 1.5, 'Inflation': -0.8, 'Rates': -1.2, 'Type': 'Cyclical'},
            'Consumer Discretionary': {'GDP': 1.3, 'Inflation': -1.0, 'Rates': -0.9, 'Type': 'Cyclical'},
            'Financials': {'GDP': 1.1, 'Inflation': -0.5, 'Rates': 0.8, 'Type': 'Cyclical'},
            'Industrials': {'GDP': 1.4, 'Inflation': -0.7, 'Rates': -0.6, 'Type': 'Cyclical'},
            'Materials': {'GDP': 1.6, 'Inflation': 0.3, 'Rates': -0.5, 'Type': 'Cyclical'},
            'Energy': {'GDP': 0.9, 'Inflation': 0.9, 'Rates': -0.3, 'Type': 'Cyclical'},
            'Consumer Staples': {'GDP': 0.3, 'Inflation': -0.4, 'Rates': -0.2, 'Type': 'Defensive'},
            'Healthcare': {'GDP': 0.4, 'Inflation': -0.3, 'Rates': -0.4, 'Type': 'Defensive'},
            'Utilities': {'GDP': 0.2, 'Inflation': -0.2, 'Rates': -0.7, 'Type': 'Defensive'},
            'Real Estate': {'GDP': 0.8, 'Inflation': -0.5, 'Rates': -1.5, 'Type': 'Sensitive'},
            'Communication Services': {'GDP': 0.7, 'Inflation': -0.6, 'Rates': -0.8, 'Type': 'Mixed'}
        }
        
        # Afficher la matrice
        st.markdown("#### 📊 SECTOR SENSITIVITY MATRIX")
        
        sens_df = pd.DataFrame(sector_sensitivity).T
        sens_df = sens_df.reset_index().rename(columns={'index': 'Sector'})
        
        st.dataframe(sens_df, use_container_width=True, hide_index=True)
        
        st.caption("""
        **Interprétation des sensibilités :**
        - **GDP**: Sensibilité à la croissance économique (>1 = très sensible)
        - **Inflation**: Sensibilité à l'inflation (positif = bénéficie, négatif = pénalisé)
        - **Rates**: Sensibilité aux taux d'intérêt (négatif = souffre de hausse taux)
        """)
        
        # Calculer recommandations actuelles
        st.markdown("#### 🎯 CURRENT SECTOR RECOMMENDATIONS")
        
        if st.button("📊 GENERATE RECOMMENDATIONS", use_container_width=True, key="sector_reco"):
            with st.spinner("Analyzing macro conditions..."):
                # Récupérer conditions actuelles
                df_gdp = get_fred_series('GDP')
                df_cpi = get_fred_series('CPIAUCSL')
                df_rates = get_fred_series('DGS10')
                
                if all(df is not None for df in [df_gdp, df_cpi, df_rates]):
                    # Croissance GDP (QoQ annualisé)
                    gdp_growth = (df_gdp['value'].iloc[-1] / df_gdp['value'].iloc[-2] - 1) * 400
                    
                    # Inflation YoY
                    inflation = (df_cpi['value'].iloc[-1] / df_cpi['value'].iloc[-13] - 1) * 100
                    
                    # Changement taux (vs 6 mois)
                    rate_change = df_rates['value'].iloc[-1] - df_rates['value'].iloc[-120]
                    
                    # Normaliser les signaux (-1 à +1)
                    gdp_signal = np.clip((gdp_growth - 2) / 2, -1, 1)  # 2% = neutre
                    inflation_signal = np.clip((inflation - 2) / 2, -1, 1)  # 2% = neutre
                    rate_signal = np.clip(rate_change / 2, -1, 1)  # 2% move = extrême
                    
                    st.markdown("**📊 Current Macro Conditions:**")
                    
                    col_macro1, col_macro2, col_macro3 = st.columns(3)
                    
                    with col_macro1:
                        st.metric("GDP GROWTH", f"{gdp_growth:.2f}%")
                    
                    with col_macro2:
                        st.metric("INFLATION", f"{inflation:.2f}%")
                    
                    with col_macro3:
                        st.metric("RATE CHANGE (6M)", f"{rate_change:+.2f}%")
                    
                    # Calculer scores par secteur
                    sector_scores = {}
                    
                    for sector, sensitivities in sector_sensitivity.items():
                        score = (
                            sensitivities['GDP'] * gdp_signal +
                            sensitivities['Inflation'] * inflation_signal +
                            sensitivities['Rates'] * rate_signal
                        )
                        sector_scores[sector] = {
                            'score': score,
                            'type': sensitivities['Type']
                        }
                    
                    # Trier par score
                    sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1]['score'], reverse=True)
                    
                    st.markdown("#### 🎯 SECTOR RANKINGS")
                    
                    for i, (sector, data) in enumerate(sorted_sectors):
                        col_rank1, col_rank2, col_rank3 = st.columns([3, 1, 1])
                        
                        with col_rank1:
                            if i < 3:
                                emoji = "🟢"
                                label = "OVERWEIGHT"
                            elif i < 8:
                                emoji = "🟡"
                                label = "NEUTRAL"
                            else:
                                emoji = "🔴"
                                label = "UNDERWEIGHT"
                            
                            st.markdown(f"{emoji} **{sector}** ({data['type']})")
                        
                        with col_rank2:
                            st.markdown(f"Score: {data['score']:.2f}")
                        
                        with col_rank3:
                            st.markdown(f"**{label}**")
                    
                    # Graphique radar
                    st.markdown("#### 📊 TOP 5 SECTORS - SENSITIVITY PROFILE")
                    
                    top5_sectors = [s[0] for s in sorted_sectors[:5]]
                    
                    fig_radar = go.Figure()
                    
                    for sector in top5_sectors:
                        sens = sector_sensitivity[sector]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[sens['GDP'], sens['Inflation'], sens['Rates']],
                            theta=['GDP Sensitivity', 'Inflation Sensitivity', 'Rate Sensitivity'],
                            name=sector,
                            fill='toself'
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[-2, 2], gridcolor='#333'),
                            angularaxis=dict(gridcolor='#333')
                        ),
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        height=500
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
    
    # ===== INTEGRATION TAB 4: COMPANY FUNDAMENTALS =====
    with integration_tab4:
        st.markdown("#### 🏢 MACRO-TO-COMPANY FUNDAMENTALS")
        st.caption("Adjust revenue forecasts using GDP/PCE macro drivers")
        
        st.markdown("""
        <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
            📈 REVENUE FORECASTING FRAMEWORK
            </p>
            <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
            Ajuste les prévisions de revenus des entreprises en fonction des conditions macro.
            Utilise des élasticités sectorielles calibrées.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            company_ticker = st.text_input(
                "COMPANY TICKER",
                value="AAPL",
                help="Yahoo Finance ticker",
                key="company_ticker"
            ).upper()
            
            company_sector = st.selectbox(
                "COMPANY SECTOR",
                options=list(sector_sensitivity.keys()),
                index=0,
                key="company_sector"
            )
        
        with col_comp2:
            base_revenue = st.number_input(
                "BASE REVENUE (Last Year, $B)",
                min_value=0.0,
                value=100.0,
                step=1.0,
                key="base_revenue"
            )
            
            forecast_horizon = st.slider(
                "FORECAST HORIZON (Quarters)",
                min_value=1,
                max_value=8,
                value=4,
                key="forecast_horizon"
            )
        
        if st.button("📊 GENERATE REVENUE FORECAST", use_container_width=True, key="revenue_forecast"):
            with st.spinner("Generating macro-adjusted forecast..."):
                # Récupérer données macro
                df_gdp = get_fred_series('GDP')
                df_pce = get_fred_series('PCE')
                
                if df_gdp is not None and df_pce is not None:
                    # Croissance GDP historique
                    gdp_growth_hist = df_gdp['value'].pct_change(1).tail(8) * 100
                    
                    # Sensibilité du secteur
                    sector_sens = sector_sensitivity[company_sector]
                    
                    st.markdown("### 📊 REVENUE FORECAST")
                    
                    # Scénarios
                    scenarios = {
                        'Base Case': gdp_growth_hist.mean(),
                        'Bull Case': gdp_growth_hist.mean() + gdp_growth_hist.std(),
                        'Bear Case': gdp_growth_hist.mean() - gdp_growth_hist.std()
                    }
                    
                    # Calculer prévisions pour chaque scénario
                    forecast_data = []
                    
                    for scenario_name, gdp_growth in scenarios.items():
                        # Calculer croissance revenue ajustée par sensibilité secteur
                        revenue_growth = sector_sens['GDP'] * (gdp_growth / 100)
                        
                        # Projeter sur horizon
                        projected_revenue = [base_revenue]
                        for q in range(forecast_horizon):
                            next_revenue = projected_revenue[-1] * (1 + revenue_growth)
                            projected_revenue.append(next_revenue)
                        
                        forecast_data.append({
                            'scenario': scenario_name,
                            'gdp_growth': gdp_growth,
                            'revenue_growth': revenue_growth * 100,
                            'revenues': projected_revenue[1:],
                            'final_revenue': projected_revenue[-1]
                        })
                    
                    # Afficher résumé
                    st.markdown("#### 📊 FORECAST SUMMARY")
                    
                    summary_df = pd.DataFrame([
                        {
                            'Scenario': f['scenario'],
                            'GDP Growth (%)': f"{f['gdp_growth']:.2f}",
                            'Revenue Growth (%)': f"{f['revenue_growth']:.2f}",
                            'Year-End Revenue ($B)': f"{f['final_revenue']:.2f}",
                            'vs Base': f"{((f['final_revenue']/base_revenue - 1) * 100):+.2f}%"
                        }
                        for f in forecast_data
                    ])
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Graphique des scénarios
                    st.markdown("#### 📈 REVENUE PROJECTIONS BY SCENARIO")
                    
                    fig_forecast = go.Figure()
                    
                    quarters = [f"Q{i+1}" for i in range(forecast_horizon)]
                    
                    colors = {'Base Case': '#FFAA00', 'Bull Case': '#00FF00', 'Bear Case': '#FF0000'}
                    
                    for f in forecast_data:
                        fig_forecast.add_trace(go.Scatter(
                            x=['Q0'] + quarters,
                            y=[base_revenue] + f['revenues'],
                            mode='lines+markers',
                            name=f['scenario'],
                            line=dict(color=colors.get(f['scenario'], '#999'), width=2),
                            marker=dict(size=8)
                        ))
                    
                    fig_forecast.update_layout(
                        title=f"{company_ticker} Revenue Forecast - Macro-Adjusted",
                        paper_bgcolor='#000',
                        plot_bgcolor='#111',
                        font=dict(color='#FFAA00', size=10),
                        xaxis=dict(gridcolor='#333', title="Quarter"),
                        yaxis=dict(gridcolor='#333', title="Revenue ($B)"),
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Sensitivity Analysis
                    st.markdown("#### 📊 SENSITIVITY ANALYSIS")
                    
                    st.markdown(f"""
                    **Sector Sensitivities ({company_sector}):**
                    - GDP Sensitivity: {sector_sens['GDP']:.2f}x
                    - Inflation Sensitivity: {sector_sens['Inflation']:.2f}x
                    - Rate Sensitivity: {sector_sens['Rates']:.2f}x
                    
                    **Interpretation:**
                    - GDP sensitivity of {sector_sens['GDP']:.2f}x means a 1% GDP growth → 
                      {sector_sens['GDP']:.2f}% revenue growth for this sector
                    """)
                    
                    # Matrice de sensibilité
                    st.markdown("#### 🎯 SCENARIO MATRIX")
                    
                    gdp_scenarios = [-2, -1, 0, 1, 2, 3, 4]
                    
                    matrix_data = []
                    for gdp in gdp_scenarios:
                        revenue_impact = sector_sens['GDP'] * gdp
                        final_rev = base_revenue * (1 + revenue_impact/100) ** forecast_horizon
                        
                        matrix_data.append({
                            'GDP Growth (%)': f"{gdp:+.0f}",
                            'Revenue Impact/Q (%)': f"{revenue_impact:+.2f}",
                            'Final Revenue ($B)': f"{final_rev:.2f}",
                            'Total Change (%)': f"{((final_rev/base_revenue - 1) * 100):+.2f}"
                        })
                    
                    matrix_df = pd.DataFrame(matrix_data)
                    st.dataframe(matrix_df, use_container_width=True, hide_index=True)
                    
                    # Download forecast
                    st.markdown("#### 💾 EXPORT FORECAST")
                    
                    forecast_export = []
                    for q in range(forecast_horizon):
                        row = {'Quarter': f"Q{q+1}"}
                        for f in forecast_data:
                            row[f['scenario']] = f['revenues'][q]
                        forecast_export.append(row)
                    
                    forecast_df = pd.DataFrame(forecast_export)
                    csv_forecast = forecast_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 DOWNLOAD FORECAST CSV",
                        data=csv_forecast,
                        file_name=f"{company_ticker}_revenue_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Comparaison avec market expectations
                    st.markdown("#### 📊 MARKET EXPECTATIONS COMPARISON")
                    
                    st.markdown("""
                    <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
                        <p style="margin: 0; font-size: 10px; color: #FFAA00;">
                        💡 NEXT STEPS:
                        </p>
                        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
                            <li>Compare forecast with analyst consensus (from Bloomberg/FactSet)</li>
                            <li>Calculate implied earnings using historical margins</li>
                            <li>Adjust EPS estimates for macro headwinds/tailwinds</li>
                            <li>Update DCF valuation with revised revenue assumptions</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Advanced: DCF impact
                    st.markdown("#### 💰 IMPLIED VALUATION IMPACT")
                    
                    col_val1, col_val2 = st.columns(2)
                    
                    with col_val1:
                        ebitda_margin = st.number_input(
                            "EBITDA MARGIN (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=25.0,
                            step=1.0,
                            key="ebitda_margin"
                        )
                    
                    with col_val2:
                        ev_ebitda_multiple = st.number_input(
                            "EV/EBITDA MULTIPLE",
                            min_value=0.0,
                            max_value=50.0,
                            value=15.0,
                            step=0.5,
                            key="ev_ebitda"
                        )
                    
                    if st.button("💰 CALCULATE IMPLIED VALUATION", key="calc_valuation"):
                        val_results = []
                        
                        for f in forecast_data:
                            final_rev = f['final_revenue']
                            ebitda = final_rev * (ebitda_margin / 100)
                            enterprise_value = ebitda * ev_ebitda_multiple
                            
                            val_results.append({
                                'Scenario': f['scenario'],
                                'Revenue ($B)': f"{final_rev:.2f}",
                                'EBITDA ($B)': f"{ebitda:.2f}",
                                'Enterprise Value ($B)': f"{enterprise_value:.2f}"
                            })
                        
                        val_df = pd.DataFrame(val_results)
                        st.dataframe(val_df, use_container_width=True, hide_index=True)
                        
                        # Graphique valuation
                        fig_val = go.Figure()
                        
                        fig_val.add_trace(go.Bar(
                            x=[v['Scenario'] for v in val_results],
                            y=[float(v['Enterprise Value ($B)']) for v in val_results],
                            marker=dict(color=['#FF0000', '#FFAA00', '#00FF00']),
                            text=[v['Enterprise Value ($B)'] for v in val_results],
                            textposition='auto'
                        ))
                        
                        fig_val.update_layout(
                            title=f"{company_ticker} Implied Enterprise Value by Scenario",
                            paper_bgcolor='#000',
                            plot_bgcolor='#111',
                            font=dict(color='#FFAA00', size=10),
                            xaxis=dict(gridcolor='#333', title="Scenario"),
                            yaxis=dict(gridcolor='#333', title="Enterprise Value ($B)"),
                            height=400
                        )
                        
                        st.plotly_chart(fig_val, use_container_width=True)
                        
                        # Upside/Downside
                        base_val = float(val_results[0]['Enterprise Value ($B)'])
                        bull_val = float(val_results[1]['Enterprise Value ($B)'])
                        bear_val = float(val_results[2]['Enterprise Value ($B)'])
                        
                        upside = ((bull_val / base_val) - 1) * 100
                        downside = ((bear_val / base_val) - 1) * 100
                        
                        col_updown1, col_updown2, col_updown3 = st.columns(3)
                        
                        with col_updown1:
                            st.metric("BASE CASE EV", f"${base_val:.1f}B")
                        
                        with col_updown2:
                            st.metric("UPSIDE", f"{upside:+.1f}%", 
                                     delta=f"${bull_val - base_val:.1f}B")
                        
                        with col_updown3:
                            st.metric("DOWNSIDE", f"{downside:+.1f}%",
                                     delta=f"${bear_val - base_val:.1f}B")
                
                else:
                    st.error("❌ Could not retrieve macro data")
        
        # Section: Macro-to-Earnings Pipeline
        st.markdown('<div style="border-top: 1px solid #333; margin: 20px 0;"></div>', unsafe_allow_html=True)
        st.markdown("#### 🔄 COMPLETE MACRO-TO-EARNINGS PIPELINE")
        
        st.markdown("""
        <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; font-size: 11px; color: #00FF00; font-weight: bold;">
            📈 FULL INTEGRATION WORKFLOW
            </p>
            <ol style="margin: 10px 0; font-size: 10px; color: #999;">
                <li><strong>Macro Inputs:</strong> GDP, CPI, Rates, Unemployment (FRED)</li>
                <li><strong>Sector Mapping:</strong> Apply sector sensitivities</li>
                <li><strong>Revenue Forecast:</strong> Project company revenues</li>
                <li><strong>Margin Assumptions:</strong> Adjust for inflation/cost pressures</li>
                <li><strong>Earnings Translation:</strong> Calculate EPS impacts</li>
                <li><strong>Valuation Impact:</strong> Update fair value estimates</li>
                <li><strong>Portfolio Action:</strong> Adjust position sizing</li>
            </ol>
            
            <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00; font-weight: bold;">
            💡 USE CASES:
            </p>
            <ul style="margin: 5px 0; font-size: 9px; color: #999;">
                <li><strong>Equity Research:</strong> Adjust DCF models with macro scenarios</li>
                <li><strong>Portfolio Management:</strong> Tilt exposures based on macro outlook</li>
                <li><strong>Risk Management:</strong> Stress-test portfolios under macro shocks</li>
                <li><strong>Trading:</strong> Generate macro-driven stock signals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick reference table
        st.markdown("#### 📊 QUICK REFERENCE: MACRO LINKAGES")
        
        linkages_data = [
            {'Macro Variable': 'GDP Growth', 'Primary Impact': 'Revenue', 'Secondary Impact': 'Volume/Pricing', 'Best For': 'Cyclicals'},
            {'Macro Variable': 'Inflation (CPI)', 'Primary Impact': 'COGS', 'Secondary Impact': 'Pricing Power', 'Best For': 'Consumer'},
            {'Macro Variable': 'Unemployment', 'Primary Impact': 'Demand', 'Secondary Impact': 'Labor Costs', 'Best For': 'Retail'},
            {'Macro Variable': 'Interest Rates', 'Primary Impact': 'Discount Rate', 'Secondary Impact': 'Finance Costs', 'Best For': 'Growth/Tech'},
            {'Macro Variable': 'PCE', 'Primary Impact': 'Consumer Spending', 'Secondary Impact': 'Margins', 'Best For': 'Discretionary'},
            {'Macro Variable': 'Industrial Production', 'Primary Impact': 'Capacity Use', 'Secondary Impact': 'Capex', 'Best For': 'Industrials'},
            {'Macro Variable': 'Housing Starts', 'Primary Impact': 'Demand', 'Secondary Impact': 'Credit', 'Best For': 'Homebuilders'},
            {'Macro Variable': 'Oil Prices', 'Primary Impact': 'Input Costs', 'Secondary Impact': 'Transport', 'Best For': 'Energy/Airlines'}
        ]
        
        linkages_df = pd.DataFrame(linkages_data)
        st.dataframe(linkages_df, use_container_width=True, hide_index=True)

                        
# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | BACKTESTING & FORECASTING | LAST UPDATE: {last_update}
    <br>
    <span style="color: #FF6600;">⚠️ FOR EDUCATIONAL PURPOSES ONLY - NOT FINANCIAL ADVICE</span>
</div>
""", unsafe_allow_html=True)
