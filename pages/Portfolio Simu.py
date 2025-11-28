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
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #FF8C00 !important;
        background-color: #1a1a1a !important;
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
        'ba_h4_data', 'bac_h4_data', 'meta_h4_data', 'msft_h4_data', 'nvda_h4_data', 
        'qqq_h4_data', 'vixm_h4_data'
    ]
    return sorted(tables)

# ============================================================================
# RECUPERATION DES DONNEES
# ============================================================================

@st.cache_data(ttl=600)
def load_data(table_name, start_date, end_date):
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        response = supabase.table(table_name).select("*").order("date", desc=False).limit(10000).execute()
        
        if not response.data:
            return None
        
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Filtrer par période
        df = df.loc[start_date:end_date]
        
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
        
        # Charger les données pour chaque action
        portfolio_data = {}
        
        for stock in selected_stocks:
            table_name = stock.lower() + '_h4_data'
            df = load_data(table_name, start_date, end_date)
            
            if df is not None and len(df) > 0:
                portfolio_data[stock] = df['close']
            else:
                st.warning(f"⚠️ No data available for {stock} in this period")
        
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
        # CALCUL DU PORTEFEUILLE
        # ====================================================================
        
        # Calculer le nombre d'actions à acheter pour chaque titre
        shares = {}
        for stock in selected_stocks:
            weight_amount = initial_capital * (weights[stock] / 100.0)
            initial_price = prices_df[stock].iloc[0]
            shares[stock] = weight_amount / initial_price
        
        # Calculer la valeur du portefeuille à chaque date
        portfolio_values = pd.Series(0, index=prices_df.index)
        
        for stock in selected_stocks:
            portfolio_values += prices_df[stock] * shares[stock]
        
        # Calculer les rendements quotidiens
        daily_returns = portfolio_values.pct_change()
        
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
        
        # Sharpe Ratio (en supposant un taux sans risque de 2%)
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
            st.metric(
                "Initial Value",
                f"${initial_capital:,.2f}"
            )
        
        with col2:
            st.metric(
                "Final Value",
                f"${final_value:,.2f}",
                f"{total_return_abs:+,.2f} ({total_return:+.2f}%)"
            )
        
        with col3:
            st.metric(
                "Annualized Return",
                f"{annualized_return:.2f}%"
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{volatility:.2f}%"
            )
        
        with col5:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}"
            )
        
        col6, col7, col8, col9 = st.columns(4)
        
        with col6:
            st.metric(
                "Max Drawdown",
                f"{max_drawdown:.2f}%"
            )
        
        with col7:
            st.metric(
                "Best Day",
                f"{best_day:+.2f}%"
            )
        
        with col8:
            st.metric(
                "Worst Day",
                f"{worst_day:.2f}%"
            )
        
        with col9:
            st.metric(
                "Trading Days",
                f"{len(prices_df):,}"
            )
        
        st.divider()
        
        # ====================================================================
        # GRAPHIQUE DE LA VALEUR DU PORTEFEUILLE
        # ====================================================================
        
        st.subheader("📈 PORTFOLIO VALUE OVER TIME")
        
        fig, ax = plt.subplots(figsize=(16, 8), facecolor='#000000')
        ax.set_facecolor('#0a0a0a')
        
        # Tracer la valeur du portefeuille
        ax.plot(portfolio_values.index, portfolio_values.values, 
                linewidth=2.5, color='#00D9FF', label='Portfolio Value', zorder=5)
        
        # Ligne de référence (capital initial)
        ax.axhline(y=initial_capital, color='#FF8C00', linestyle='--', 
                   linewidth=1.5, label='Initial Capital', alpha=0.7)
        
        # Remplir les zones de profit/perte
        ax.fill_between(portfolio_values.index, portfolio_values.values, initial_capital,
                        where=(portfolio_values.values >= initial_capital),
                        color='#00FF00', alpha=0.2, label='Profit Zone')
        ax.fill_between(portfolio_values.index, portfolio_values.values, initial_capital,
                        where=(portfolio_values.values < initial_capital),
                        color='#FF0000', alpha=0.2, label='Loss Zone')
        
        ax.set_title('Portfolio Value Evolution', 
                     fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
        ax.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
        ax.set_ylabel('Value ($)', fontsize=12, color='#FF8C00', fontweight='bold')
        ax.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', 
                  fontsize=10, labelcolor='#FF8C00')
        ax.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Format des dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # ====================================================================
        # GRAPHIQUE DRAWDOWN
        # ====================================================================
        
        st.subheader("📉 DRAWDOWN ANALYSIS")
        
        fig2, ax2 = plt.subplots(figsize=(16, 5), facecolor='#000000')
        ax2.set_facecolor('#0a0a0a')
        
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color='#FF0000', alpha=0.5)
        ax2.plot(drawdown.index, drawdown.values, 
                 linewidth=2, color='#FF0000', label='Drawdown')
        
        ax2.set_title('Portfolio Drawdown Over Time', 
                      fontsize=16, fontweight='bold', color='#FF8C00', pad=20)
        ax2.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, color='#FF8C00', fontweight='bold')
        ax2.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax2.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Format des dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig2.tight_layout()
        st.pyplot(fig2)
        
        # ====================================================================
        # CONTRIBUTION PAR ACTION
        # ====================================================================
        
        st.subheader("📊 INDIVIDUAL STOCK CONTRIBUTION")
        
        # Calculer la contribution de chaque action
        contributions = {}
        for stock in selected_stocks:
            initial_value = shares[stock] * prices_df[stock].iloc[0]
            final_value_stock = shares[stock] * prices_df[stock].iloc[-1]
            contribution = final_value_stock - initial_value
            contributions[stock] = {
                'Initial Value': initial_value,
                'Final Value': final_value_stock,
                'Absolute Return': contribution,
                'Return %': (contribution / initial_value) * 100,
                'Weight': weights[stock]
            }
        
        contrib_df = pd.DataFrame(contributions).T
        
        # Afficher le tableau
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Graphique en barres
            fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor='#000000')
            ax3.set_facecolor('#0a0a0a')
            
            colors = ['#00FF00' if x > 0 else '#FF0000' for x in contrib_df['Return %']]
            ax3.barh(contrib_df.index, contrib_df['Return %'], color=colors, alpha=0.8)
            
            ax3.set_title('Individual Stock Returns (%)', 
                          fontsize=14, fontweight='bold', color='#FF8C00', pad=15)
            ax3.set_xlabel('Return (%)', fontsize=11, color='#FF8C00', fontweight='bold')
            ax3.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5, axis='x')
            ax3.tick_params(colors='#FFFFFF', labelsize=10)
            ax3.axvline(x=0, color='#FFFFFF', linestyle='-', linewidth=0.5)
            
            fig3.tight_layout()
            st.pyplot(fig3)
        
        with col2:
            # Tableau formaté
            display_contrib = contrib_df.copy()
            display_contrib['Initial Value'] = display_contrib['Initial Value'].apply(lambda x: f"${x:,.2f}")
            display_contrib['Final Value'] = display_contrib['Final Value'].apply(lambda x: f"${x:,.2f}")
            display_contrib['Absolute Return'] = display_contrib['Absolute Return'].apply(lambda x: f"${x:+,.2f}")
            display_contrib['Return %'] = display_contrib['Return %'].apply(lambda x: f"{x:+.2f}%")
            display_contrib['Weight'] = display_contrib['Weight'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_contrib, width='stretch')
        
        # ====================================================================
        # CORRELATION MATRIX
        # ====================================================================
        
        st.subheader("🔗 CORRELATION MATRIX")
        
        # Calculer les rendements quotidiens de chaque action
        returns_df = prices_df.pct_change().dropna()
        correlation_matrix = returns_df.corr()
        
        # Créer une heatmap
        fig4, ax4 = plt.subplots(figsize=(10, 8), facecolor='#000000')
        ax4.set_facecolor('#0a0a0a')
        
        im = ax4.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        
        # Ajouter les valeurs dans les cellules
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        ax4.set_xticks(range(len(correlation_matrix)))
        ax4.set_yticks(range(len(correlation_matrix)))
        ax4.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax4.set_yticklabels(correlation_matrix.columns)
        ax4.tick_params(colors='#FFFFFF', labelsize=10)
        
        ax4.set_title('Stock Returns Correlation Matrix', 
                      fontsize=14, fontweight='bold', color='#FF8C00', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.ax.tick_params(colors='#FFFFFF')
        
        fig4.tight_layout()
        st.pyplot(fig4)
        
        # ====================================================================
        # DONNÉES DÉTAILLÉES
        # ====================================================================
        
        with st.expander("📋 VIEW DETAILED DATA"):
            st.subheader("Portfolio Value History")
            
            detailed_df = pd.DataFrame({
                'Date': portfolio_values.index,
                'Portfolio Value': portfolio_values.values,
                'Daily Return %': daily_returns.values * 100,
                'Cumulative Return %': ((portfolio_values.values / initial_capital) - 1) * 100
            })
            
            detailed_df['Portfolio Value'] = detailed_df['Portfolio Value'].apply(lambda x: f"${x:,.2f}")
            detailed_df['Daily Return %'] = detailed_df['Daily Return %'].apply(lambda x: f"{x:+.2f}%")
            detailed_df['Cumulative Return %'] = detailed_df['Cumulative Return %'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(detailed_df.tail(50), width='stretch')
        
        # Footer
        st.markdown("---")
        st.markdown(f"**Simulation completed** | {len(selected_stocks)} stocks | Period: {start_date} to {end_date}")

else:
    st.info("👆 Configure your portfolio above and click 'RUN SIMULATION' to start")
