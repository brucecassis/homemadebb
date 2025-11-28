import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# ============================================================================
# CONFIG STREAMLIT - STYLE BLOOMBERG
# ============================================================================
st.set_page_config(page_title="Portfolio Simulator", page_icon="💼", layout="wide")

# CSS personnalisé style Bloomberg
st.markdown("""
<style>
    /* Fond noir Bloomberg */
    .stApp {
        background-color: #000000;
    }
    
    /* Texte en orange Bloomberg */
    .stMarkdown, .stMetric label, p, label {
        color: #FF8C00 !important;
    }
    
    /* Valeurs des métriques en blanc */
    .stMetric .metric-value {
        color: #FFFFFF !important;
        font-weight: bold;
        font-size: 28px !important;
    }
    
    /* Delta des métriques */
    [data-testid="stMetricDelta"] {
        color: #00FF00 !important;
    }
    
    /* Titres */
    h1, h2, h3 {
        color: #FF8C00 !important;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    /* Divider */
    hr {
        border-color: #FF8C00 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #FF8C00 !important;
    }
    
    /* Number input */
    .stNumberInput > label {
        color: #FF8C00 !important;
    }
    
    /* Select box */
    .stSelectbox > label, .stMultiselect > label {
        color: #FF8C00 !important;
    }
    
    /* Date input */
    .stDateInput > label {
        color: #FF8C00 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 PORTFOLIO SIMULATOR")
st.markdown("**Simulate and analyze your stock portfolio performance over time**")
st.divider()

# ============================================================================
# PARAMETRES SUPABASE
# ============================================================================

SUPABASE_URL = "https://gbrefcefeavmqupulzyw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdicmVmY2VmZWF2bXF1cHVsenl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0OTA2NjksImV4cCI6MjA3OTA2NjY2OX0.WsA-3so0J52hAyZTIddVT0qqLuvcxjHYTZ4XkZ5mMio"

# ============================================================================
# RECUPERATION DE LA LISTE DES TABLES
# ============================================================================

@st.cache_data(ttl=3600)
def get_all_tables():
    """Récupère la liste de toutes les tables disponibles"""
    tables = [
        'a_h4_data', 'aal_h4_data', 'aapl_h4_data', 'abbv_h4_data', 'abnb_h4_data', 
        'abt_h4_data', 'acgl_h4_data', 'acn_h4_data', 'adbe_h4_data', 'adi_h4_data',
        'adm_h4_data', 'adp_h4_data', 'adsk_h4_data', 'aee_h4_data', 'aep_h4_data',
        'aes_h4_data', 'afl_h4_data', 'aig_h4_data', 'aiz_h4_data', 'ajg_h4_data',
        'akam_h4_data', 'alb_h4_data', 'algn_h4_data', 'all_h4_data', 'alle_h4_data',
        'amat_h4_data', 'amcr_h4_data', 'amd_h4_data', 'ame_h4_data', 'amgn_h4_data',
        'amp_h4_data', 'amt_h4_data', 'amzn_h4_data', 'anet_h4_data', 'aon_h4_data',
        'aos_h4_data', 'apa_h4_data', 'apd_h4_data', 'aph_h4_data', 'aptv_h4_data',
        'are_h4_data', 'ato_h4_data', 'avb_h4_data', 'avgo_h4_data', 'avy_h4_data',
        'awk_h4_data', 'axon_h4_data', 'axp_h4_data', 'azo_h4_data', 'ba_h4_data',
        'bac_h4_data', 'ball_h4_data', 'bax_h4_data', 'bbwi_h4_data', 'bby_h4_data',
        'bdx_h4_data', 'ben_h4_data', 'bf_b_h4_data', 'bg_h4_data', 'biib_h4_data',
        'bio_h4_data', 'bk_h4_data', 'bkng_h4_data', 'bkr_h4_data', 'blk_h4_data',
        'bmy_h4_data', 'br_h4_data', 'brk_b_h4_data', 'bro_h4_data', 'bsx_h4_data',
        'bwa_h4_data', 'bx_h4_data', 'bxp_h4_data', 'c_h4_data', 'cag_h4_data',
        'cah_h4_data', 'carr_h4_data', 'cat_h4_data', 'cb_h4_data', 'cboe_h4_data',
        'cbre_h4_data', 'cci_h4_data', 'ccl_h4_data', 'cdns_h4_data', 'cdw_h4_data',
        'meta_h4_data', 'msft_h4_data', 'nvda_h4_data', 'ms_h4_data',
        'qqq_h4_data', 'vixm_h4_data'
    ]
    return sorted(tables)

# ============================================================================
# RECUPERATION DES DONNEES (identique au code qui fonctionne)
# ============================================================================

@st.cache_data(ttl=600)
def load_data(table_name):
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        response = supabase.table(table_name).select("*").order("date", desc=False).limit(10000).execute()
        
        if not response.data:
            st.error(f"❌ No data found in table: {table_name}")
            return None
        
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading {table_name}: {str(e)}")
        return None

# ============================================================================
# PARAMETRES DU PORTEFEUILLE - EN HAUT
# ============================================================================

st.subheader("⚙️ PORTFOLIO CONFIGURATION")

# Ligne 1: Période
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date:",
        value=datetime.now().date() - timedelta(days=365),
        max_value=datetime.now().date()
    )
with col2:
    end_date = st.date_input(
        "End Date:",
        value=datetime.now().date(),
        max_value=datetime.now().date()
    )

# Ligne 2: Sélection des actions
available_tables = get_all_tables()
table_names_display = [t.replace('_h4_data', '').upper() for t in available_tables]
table_dict = dict(zip(table_names_display, available_tables))

selected_stocks = st.multiselect(
    "Select Stocks for Portfolio:",
    table_names_display,
    default=['AAPL', 'MSFT', 'NVDA'] if all(s in table_names_display for s in ['AAPL', 'MSFT', 'NVDA']) else table_names_display[:3]
)

if not selected_stocks:
    st.warning("⚠️ Please select at least one stock!")
    st.stop()

# Ligne 3: Configuration des pondérations
st.markdown("**Portfolio Weights (%)**")

weights = {}
total_weight = 0

cols = st.columns(len(selected_stocks))
for i, stock in enumerate(selected_stocks):
    with cols[i]:
        weight = st.number_input(
            f"{stock}",
            min_value=0.0,
            max_value=100.0,
            value=100.0 / len(selected_stocks),
            step=1.0,
            key=f"weight_{stock}"
        )
        weights[stock] = weight
        total_weight += weight

# Vérifier que la somme fait 100%
if abs(total_weight - 100.0) > 0.01:
    st.error(f"⚠️ Total weight must equal 100%! Current: {total_weight:.2f}%")
    st.stop()
else:
    st.success(f"✅ Portfolio weights: {total_weight:.2f}%")

# Capital initial
initial_capital = st.number_input(
    "Initial Capital ($):",
    min_value=1000.0,
    max_value=10000000.0,
    value=10000.0,
    step=1000.0
)

st.divider()

# ============================================================================
# BOUTON DE SIMULATION
# ============================================================================

if st.button("🚀 RUN SIMULATION", type="primary"):
    
    with st.spinner("📊 Loading data and calculating portfolio performance..."):
        
        # Charger les données pour chaque action (en utilisant le code qui fonctionne)
        portfolio_data = {}
        
        for stock in selected_stocks:
            table_name = table_dict[stock]
            df = load_data(table_name)
            
            if df is not None and len(df) > 0:
                # Filtrer par période
                df_filtered = df.loc[start_date:end_date]
                if len(df_filtered) > 0:
                    portfolio_data[stock] = df_filtered['close']
                else:
                    st.warning(f"⚠️ No data for {stock} in selected period")
            else:
                st.warning(f"⚠️ Could not load data for {stock}")
        
        if not portfolio_data:
            st.error("❌ No data available for any stock in the selected period!")
            st.stop()
        
        # Créer un DataFrame avec tous les prix de clôture
        prices_df = pd.DataFrame(portfolio_data)
        
        # Supprimer les NaN (dates manquantes)
        prices_df = prices_df.dropna()
        
        if len(prices_df) == 0:
            st.error("❌ No common dates found for all stocks!")
            st.stop()
        
        # ====================================================================
        # CALCUL EN BASE 100 (Performance Relative)
        # ====================================================================
        
        st.success(f"✅ Loaded {len(prices_df)} candles for {len(selected_stocks)} stocks")
        
        # Normaliser chaque action en base 100
        base_100_df = (prices_df / prices_df.iloc[0]) * 100
        
        # Calculer le portefeuille pondéré en base 100
        portfolio_base_100 = pd.Series(0, index=base_100_df.index)
        for stock in selected_stocks:
            portfolio_base_100 += base_100_df[stock] * (weights[stock] / 100.0)
        
        # Convertir en valeur réelle du portefeuille
        portfolio_values = (portfolio_base_100 / 100.0) * initial_capital
        
        # Calculer les rendements quotidiens
        daily_returns = portfolio_values.pct_change().dropna()
        
        # ====================================================================
        # STATISTIQUES DU PORTEFEUILLE
        # ====================================================================
        
        final_value = portfolio_values.iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        total_return_abs = final_value - initial_capital
        
        # Rendement annualisé
        days = (prices_df.index[-1] - prices_df.index[0]).days
        years = days / 365.25
        annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Volatilité (écart-type annualisé)
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 252 jours de trading
        
        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        excess_return = annualized_return / 100 - risk_free_rate
        sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
        
        # Meilleur et pire jour
        best_day = daily_returns.max() * 100
        worst_day = daily_returns.min() * 100
        
        # ====================================================================
        # AFFICHAGE DES STATISTIQUES PRINCIPALES
        # ====================================================================
        
        st.subheader("📊 PORTFOLIO PERFORMANCE")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Initial Value", f"${initial_capital:,.2f}")
        
        with col2:
            st.metric("Final Value", f"${final_value:,.2f}", 
                     f"{total_return_abs:+,.2f} ({total_return:+.2f}%)")
        
        with col3:
            st.metric("Annualized Return", f"{annualized_return:.2f}%")
        
        with col4:
            st.metric("Volatility", f"{volatility:.2f}%")
        
        with col5:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        col6, col7, col8, col9 = st.columns(4)
        
        with col6:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        with col7:
            st.metric("Best Day", f"{best_day:+.2f}%")
        
        with col8:
            st.metric("Worst Day", f"{worst_day:.2f}%")
        
        with col9:
            st.metric("Trading Days", f"{len(prices_df):,}")
        
        st.divider()
        
        # ====================================================================
        # GRAPHIQUE 1: COMPARAISON BASE 100 (Échelle Log)
        # ====================================================================
        
        st.subheader("📈 PERFORMANCE COMPARISON (BASE 100 - LOG SCALE)")
        
        fig1, ax1 = plt.subplots(figsize=(16, 8), facecolor='#000000')
        ax1.set_facecolor('#0a0a0a')
        
        # Couleurs pour chaque action
        colors = ['#00D9FF', '#FF1493', '#00FF00', '#FFD700', '#FF4500', '#9370DB', '#00CED1']
        
        # Tracer chaque action en base 100
        for i, stock in enumerate(selected_stocks):
            ax1.plot(base_100_df.index, base_100_df[stock], 
                    linewidth=2, label=stock, color=colors[i % len(colors)], alpha=0.7)
        
        # Tracer le portefeuille en gras
        ax1.plot(portfolio_base_100.index, portfolio_base_100, 
                linewidth=3, label='PORTFOLIO', color='#FFFFFF', zorder=10)
        
        # Ligne de référence à 100
        ax1.axhline(y=100, color='#FF8C00', linestyle='--', 
                   linewidth=1.5, label='Base 100', alpha=0.7)
        
        ax1.set_yscale('log')
        ax1.set_title('Portfolio vs Individual Stocks Performance (Log Scale)', 
                     fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
        ax1.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
        ax1.set_ylabel('Performance (Base 100, Log Scale)', fontsize=12, color='#FF8C00', fontweight='bold')
        ax1.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', 
                  fontsize=10, labelcolor='#FF8C00')
        ax1.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Format des dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig1.tight_layout()
        st.pyplot(fig1)
        
        # ====================================================================
        # GRAPHIQUE 2: VALEUR DU PORTEFEUILLE EN $
        # ====================================================================
        
        st.subheader("💰 PORTFOLIO VALUE OVER TIME")
        
        fig2, ax2 = plt.subplots(figsize=(16, 8), facecolor='#000000')
        ax2.set_facecolor('#0a0a0a')
        
        # Tracer la valeur du portefeuille
        ax2.plot(portfolio_values.index, portfolio_values.values, 
                linewidth=2.5, color='#00D9FF', label='Portfolio Value', zorder=5)
        
        # Ligne de référence (capital initial)
        ax2.axhline(y=initial_capital, color='#FF8C00', linestyle='--', 
                   linewidth=1.5, label='Initial Capital', alpha=0.7)
        
        # Remplir les zones de profit/perte
        ax2.fill_between(portfolio_values.index, portfolio_values.values, initial_capital,
                        where=(portfolio_values.values >= initial_capital),
                        color='#00FF00', alpha=0.2, label='Profit Zone')
        ax2.fill_between(portfolio_values.index, portfolio_values.values, initial_capital,
                        where=(portfolio_values.values < initial_capital),
                        color='#FF0000', alpha=0.2, label='Loss Zone')
        
        ax2.set_title('Portfolio Value Evolution', 
                     fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
        ax2.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
        ax2.set_ylabel('Value ($)', fontsize=12, color='#FF8C00', fontweight='bold')
        ax2.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax2.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', 
                  fontsize=10, labelcolor='#FF8C00')
        ax2.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Format des dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig2.tight_layout()
        st.pyplot(fig2)
        
        # ====================================================================
        # GRAPHIQUE 3: DRAWDOWN
        # ====================================================================
        
        st.subheader("📉 DRAWDOWN ANALYSIS")
        
        fig3, ax3 = plt.subplots(figsize=(16, 5), facecolor='#000000')
        ax3.set_facecolor('#0a0a0a')
        
        ax3.fill_between(drawdown.index, drawdown.values, 0,
                         color='#FF0000', alpha=0.5)
        ax3.plot(drawdown.index, drawdown.values, 
                 linewidth=2, color='#FF0000', label='Drawdown')
        
        ax3.set_title('Portfolio Drawdown Over Time', 
                      fontsize=16, fontweight='bold', color='#FF8C00', pad=20)
        ax3.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=12, color='#FF8C00', fontweight='bold')
        ax3.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax3.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Format des dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig3.tight_layout()
        st.pyplot(fig3)
        
        # ====================================================================
        # CONTRIBUTION PAR ACTION
        # ====================================================================
        
        st.subheader("📊 INDIVIDUAL STOCK PERFORMANCE")
        
        # Calculer la performance de chaque action
        stock_performance = {}
        for stock in selected_stocks:
            perf = ((base_100_df[stock].iloc[-1] - 100) / 100) * 100
            stock_performance[stock] = {
                'Weight (%)': weights[stock],
                'Performance (%)': perf,
                'Contribution to Portfolio (%)': perf * (weights[stock] / 100)
            }
        
        perf_df = pd.DataFrame(stock_performance).T
        
        # Graphique en barres
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#000000')
        
        # Performance individuelle
        ax4a.set_facecolor('#0a0a0a')
        colors_perf = ['#00FF00' if x > 0 else '#FF0000' for x in perf_df['Performance (%)']]
        ax4a.barh(perf_df.index, perf_df['Performance (%)'], color=colors_perf, alpha=0.8)
        ax4a.set_title('Individual Stock Performance', 
                      fontsize=14, fontweight='bold', color='#FF8C00', pad=15)
        ax4a.set_xlabel('Return (%)', fontsize=11, color='#FF8C00', fontweight='bold')
        ax4a.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5, axis='x')
        ax4a.tick_params(colors='#FFFFFF', labelsize=10)
        ax4a.axvline(x=0, color='#FFFFFF', linestyle='-', linewidth=0.5)
        
        # Contribution au portefeuille
        ax4b.set_facecolor('#0a0a0a')
        colors_contrib = ['#00FF00' if x > 0 else '#FF0000' for x in perf_df['Contribution to Portfolio (%)']]
        ax4b.barh(perf_df.index, perf_df['Contribution to Portfolio (%)'], color=colors_contrib, alpha=0.8)
        ax4b.set_title('Contribution to Portfolio Return', 
                      fontsize=14, fontweight='bold', color='#FF8C00', pad=15)
        ax4b.set_xlabel('Contribution (%)', fontsize=11, color='#FF8C00', fontweight='bold')
        ax4b.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5, axis='x')
        ax4b.tick_params(colors='#FFFFFF', labelsize=10)
        ax4b.axvline(x=0, color='#FFFFFF', linestyle='-', linewidth=0.5)
        
        fig4.tight_layout()
        st.pyplot(fig4)
        
        # Tableau récapitulatif
        st.dataframe(perf_df.style.format({
            'Weight (%)': '{:.1f}%',
            'Performance (%)': '{:+.2f}%',
            'Contribution to Portfolio (%)': '{:+.2f}%'
        }), width='stretch')
        
        # Footer
        st.markdown("---")
        st.markdown(f"**Simulation completed** | {len(selected_stocks)} stocks | Period: {start_date} to {end_date}")

else:
    st.info("👆 Configure your portfolio above and click 'RUN SIMULATION' to start")
