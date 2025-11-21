import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

st.set_page_config(
    page_title="Crypto Database Viewer",
    page_icon="📊",
    layout="wide"
)

# Style Bloomberg
st.markdown("""
<style>
    .main {
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stButton > button {
        background-color: #333;
        color: #FFAA00;
        font-weight: bold;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000;
    }
    
    h1, h2, h3 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .table-card {
        background: #1a1a1a;
        border: 1px solid #FFAA00;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f'''
<div style="background: #FFAA00; padding: 10px 20px; color: #000; font-weight: bold; font-size: 16px; font-family: 'Courier New', monospace; letter-spacing: 2px; margin-bottom: 20px;">
    ⬛ CRYPTO DATABASE VIEWER | {datetime.now().strftime("%H:%M:%S")} UTC
</div>
''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase_client():
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    return create_client(supabase_url, supabase_key)

supabase = get_supabase_client()


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_table_list():
    """Récupère la liste des tables depuis le registre"""
    tables_info = []
    
    try:
        response = supabase.table('perp_datasets_registry').select('*').execute()
        
        if response.data:
            for item in response.data:
                tables_info.append({
                    'table_name': f"perp_{item['dataset_id']}", 
                    'row_count': item.get('total_candles', 0),
                    'symbol': item.get('symbol', ''),
                    'timeframe': item.get('timeframe', ''),
                    'period_days': item.get('period_days', 0),
                    'start_date': item.get('start_date', ''),
                    'end_date': item.get('end_date', '')
                })
    except Exception as e:
        st.error(f"Erreur registre: {e}")
    
    return tables_info


def get_table_data(table_name, limit=100000):
    """Récupère les données d'une table - limite augmentée pour ML"""
    try:
        response = supabase.table(table_name).select('*').order('open_time').limit(limit).execute()
        return response.data
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return None


def get_table_schema(table_name):
    """Récupère le schéma d'une table en analysant les données"""
    try:
        response = supabase.table(table_name).select('*').limit(1).execute()
        if response.data and len(response.data) > 0:
            sample = response.data[0]
            schema = []
            for key, value in sample.items():
                dtype = type(value).__name__ if value is not None else 'unknown'
                schema.append({'column': key, 'type': dtype})
            return schema
    except Exception as e:
        st.error(f"Erreur lors de la récupération du schéma: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ML TRADING STRATEGY CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Génère les features techniques pour le ML"""
    
    @staticmethod
    def add_all_features(df):
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Moving Averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        tr = pd.concat([df['high'] - df['low'], 
                        abs(df['high'] - df['close'].shift()), 
                        abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        
        # Volume
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Signals
        df['ema_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    @staticmethod
    def create_target(df, horizon=1, threshold=0.0):
        df = df.copy()
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)
        return df


class MLModels:
    """Collection de modèles ML pour le trading"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.results = {}
    
    def get_models(self):
        return {
            'random_forest': ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            'xgboost': ('XGBoost', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss', verbosity=0)),
            'gradient_boosting': ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            'logistic': ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))
        }
    
    def train(self, df, features, target='target', test_size=0.2):
        df_clean = df.dropna(subset=features + [target])
        X, y = df_clean[features].values, df_clean[target].values
        
        split = int(len(X) * (1 - test_size))
        X_train, X_test = self.scaler.fit_transform(X[:split]), self.scaler.transform(X[split:])
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test, df_clean.iloc[split:]
    
    def evaluate(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else pred
        
        return {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'predictions': pred,
            'probabilities': prob,
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }


class Backtester:
    """Moteur de backtesting pour évaluer les stratégies"""
    
    def __init__(self, capital=10000, commission=0.001):
        self.capital = capital
        self.commission = commission
    
    def run(self, df, predictions, stop_loss=0.02, take_profit=0.04):
        df = df.iloc[-len(predictions):].copy()
        capital, position, entry_price = self.capital, 0, 0
        equity = [capital]
        trades = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            price = float(row['close'])
            pred = predictions[i]
            
            if position == 1:
                pnl = (price - entry_price) / entry_price
                if pnl <= -stop_loss or pnl >= take_profit or pred == 0:
                    capital *= (1 + pnl - self.commission)
                    trades.append({'pnl': pnl})
                    position = 0
            elif pred == 1 and position == 0:
                position, entry_price = 1, price
                capital *= (1 - self.commission)
            
            equity.append(capital)
        
        equity = np.array(equity[1:])
        bh = self.capital * (1 + df['close'].astype(float).pct_change().fillna(0)).cumprod().values
        
        win_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
        return {
            'equity': equity,
            'bh_equity': bh,
            'total_return': (equity[-1] / self.capital - 1) * 100,
            'bh_return': (bh[-1] / self.capital - 1) * 100,
            'max_dd': ((np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)).max() * 100,
            'trades': len(trades),
            'win_rate': len(win_trades) / len(trades) * 100 if trades else 0
        }


class TradingJournal:
    """Journal de trading pour enregistrer et analyser les trades"""
    
    @staticmethod
    def create_journal(df_test, predictions, stop_loss=0.02, take_profit=0.04):
        """Crée un journal détaillé des trades"""
        df = df_test.iloc[-len(predictions):].copy()
        df['prediction'] = predictions
        
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        entry_idx = 0
        
        for i, (idx, row) in enumerate(df.iterrows()):
            price = float(row['close'])
            pred = row['prediction']
            date = row['open_time'] if 'open_time' in row else idx
            
            if position == 0 and pred == 1:
                # Ouvrir une position
                position = 1
                entry_price = price
                entry_date = date
                entry_idx = i
                
            elif position == 1:
                # Vérifier sortie
                pnl_pct = (price - entry_price) / entry_price
                exit_reason = None
                
                if pnl_pct <= -stop_loss:
                    exit_reason = "Stop Loss"
                elif pnl_pct >= take_profit:
                    exit_reason = "Take Profit"
                elif pred == 0:
                    exit_reason = "Signal"
                
                if exit_reason:
                    duration = i - entry_idx
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': round(entry_price, 2),
                        'Exit Price': round(price, 2),
                        'PnL (%)': round(pnl_pct * 100, 2),
                        'Duration': duration,
                        'Exit Reason': exit_reason,
                        'Result': '✅ Win' if pnl_pct > 0 else '❌ Loss'
                    })
                    position = 0
        
        return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

st.title("📊 DATABASE VIEWER")

# Section 1: Vue d'ensemble des tables
st.markdown("### 📂 TABLES DANS SUPABASE")

# Bouton pour rafraîchir
if st.button("🔄 RAFRAÎCHIR", use_container_width=False):
    st.cache_data.clear()
    st.rerun()

# Récupérer la liste des tables
tables_info = get_table_list()

if tables_info:
    # Afficher les tables sous forme de cards
    cols = st.columns(min(len(tables_info), 3))
    
    for idx, table in enumerate(tables_info):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="table-card">
                <h4 style="color: #FFAA00; margin: 0;">📋 {table['table_name']}</h4>
                <p style="color: #888; margin: 5px 0;">Lignes: <span style="color: #FFAA00;">{table['row_count']:,}</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 2: Explorer une table
    st.markdown("### 🔍 EXPLORER UNE TABLE")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_table = st.selectbox(
            "Sélectionner une table",
            options=[t['table_name'] for t in tables_info],
            key="table_select"
        )
    
    with col2:
        limit = st.number_input(
            "Nombre de lignes (aperçu)",
            min_value=10,
            max_value=10000,
            value=100,
            step=100
        )
    
    if selected_table:
        # Afficher le schéma
        st.markdown("#### 📐 SCHÉMA DE LA TABLE")
        schema = get_table_schema(selected_table)
        
        if schema:
            schema_df = pd.DataFrame(schema)
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### 📊 DONNÉES (Aperçu)")
        
        # Récupérer et afficher les données (aperçu limité)
        data = get_table_data(selected_table, limit)
        
        if data:
            df = pd.DataFrame(data)
            
            # Statistiques rapides
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Colonnes", len(df.columns))
            with col_stat2:
                st.metric("Lignes affichées", len(df))
            with col_stat3:
                # Trouver le nombre total dans le registre
                total_rows = next((t['row_count'] for t in tables_info if t['table_name'] == selected_table), len(df))
                st.metric("Total dans la table", f"{total_rows:,}")
            
            # Afficher le dataframe
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Graphique candlestick si données OHLC disponibles
            if all(col in df.columns for col in ['open_time', 'open', 'high', 'low', 'close']):
                st.markdown("#### 📈 GRAPHIQUE")
                
                df['open_time'] = pd.to_datetime(df['open_time'])
                df = df.sort_values('open_time')
                
                fig = go.Figure(data=[go.Candlestick(
                    x=df['open_time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )])
                
                fig.update_layout(
                    title=f"{selected_table}",
                    xaxis_title="Date",
                    yaxis_title="Price (USDT)",
                    height=500,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Export CSV
            st.markdown("#### 💾 EXPORT")
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 TÉLÉCHARGER CSV (aperçu)",
                data=csv,
                file_name=f"{selected_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # ═══════════════════════════════════════════════════════════════
            # SECTION ML TRADING STRATEGY
            # ═══════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 🤖 ML TRADING STRATEGY")
            
            # Info sur les données
            total_rows = next((t['row_count'] for t in tables_info if t['table_name'] == selected_table), 0)
            st.info(f"📊 Cette table contient **{total_rows:,}** lignes au total. Le ML utilisera toutes les données disponibles.")
            
            with st.expander("🧠 Configurer et lancer la stratégie ML", expanded=False):
                
                col_ml1, col_ml2, col_ml3 = st.columns(3)
                with col_ml1:
                    horizon = st.selectbox("Horizon prédiction", [1, 2, 3, 5, 10], index=0)
                with col_ml2:
                    threshold = st.slider("Seuil rendement (%)", 0.0, 1.0, 0.0, 0.1) / 100
                with col_ml3:
                    test_pct = st.slider("Test set (%)", 10, 40, 20)
                
                models_to_train = st.multiselect(
                    "Modèles à entraîner",
                    ['random_forest', 'xgboost', 'gradient_boosting', 'logistic'],
                    default=['random_forest', 'xgboost'],
                    format_func=lambda x: {'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 
                                          'gradient_boosting': 'Gradient Boosting', 'logistic': 'Logistic Regression'}[x]
                )
                
                if st.button("🚀 LANCER L'ENTRAÎNEMENT ML", type="primary", use_container_width=True):
                    
                    # Charger TOUTES les données pour ML
                    with st.spinner("Chargement de TOUTES les données..."):
                        full_data = get_table_data(selected_table, limit=100000)
                        df_ml = pd.DataFrame(full_data)
                    
                    st.success(f"✅ {len(df_ml):,} lignes chargées pour l'entraînement")
                    
                    # Feature Engineering
                    with st.spinner("Génération des features techniques..."):
                        fe = FeatureEngineer()
                        df_ml = fe.add_all_features(df_ml)
                        df_ml = fe.create_target(df_ml, horizon, threshold)
                    
                    features = ['sma_10', 'sma_20', 'ema_9', 'ema_21', 'macd', 'macd_signal',
                               'rsi', 'bb_position', 'atr', 'return_1', 'return_5', 'volume_ratio',
                               'ema_cross', 'rsi_oversold', 'rsi_overbought']
                    
                    st.info(f"📊 {len(features)} features générées")
                    
                    # Entraînement
                    ml = MLModels()
                    X_train, X_test, y_train, y_test, df_test = ml.train(df_ml, features, 'target', test_pct/100)
                    
                    st.success(f"✅ Train: {len(X_train):,} lignes | Test: {len(X_test):,} lignes")
                    
                    # Évaluer les modèles
                    results = {}
                    available = ml.get_models()
                    
                    progress_bar = st.progress(0)
                    for i, key in enumerate(models_to_train):
                        name, model = available[key]
                        with st.spinner(f"Entraînement de {name}..."):
                            results[key] = ml.evaluate(model, X_train, X_test, y_train, y_test)
                            results[key]['name'] = name
                        progress_bar.progress((i + 1) / len(models_to_train))
                    
                    # Afficher résultats
                    st.markdown("#### 📊 Résultats des modèles")
                    res_df = pd.DataFrame([{
                        'Modèle': r['name'],
                        'Accuracy': f"{r['accuracy']:.2%}",
                        'Precision': f"{r['precision']:.2%}",
                        'Recall': f"{r['recall']:.2%}",
                        'F1 Score': f"{r['f1']:.2%}"
                    } for r in results.values()])
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
                    
                    # Graphique comparatif
                    fig_comp = go.Figure()
                    metrics = ['accuracy', 'precision', 'recall', 'f1']
                    for metric in metrics:
                        fig_comp.add_trace(go.Bar(
                            name=metric.title(),
                            x=[results[k]['name'] for k in results],
                            y=[results[k][metric] for k in results]
                        ))
                    fig_comp.update_layout(barmode='group', template='plotly_dark', height=350, 
                                          title='Comparaison des modèles')
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Feature Importance du meilleur modèle
                    best_key = max(results, key=lambda k: results[k]['accuracy'])
                    if results[best_key]['feature_importance'] is not None:
                        st.markdown(f"#### 📈 Feature Importance ({results[best_key]['name']})")
                        
                        importance = results[best_key]['feature_importance']
                        top_n = min(15, len(features))
                        indices = np.argsort(importance)[-top_n:]
                        
                        fig_imp = go.Figure(go.Bar(
                            x=importance[indices],
                            y=[features[i] for i in indices],
                            orientation='h',
                            marker_color='#FFAA00'
                        ))
                        fig_imp.update_layout(template='plotly_dark', height=400,
                                             title=f'Top {top_n} Features')
                        st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Sauvegarder pour backtest
                    st.session_state['ml_results'] = results
                    st.session_state['df_test'] = df_test
                    st.session_state['best_model'] = best_key
                    st.session_state['selected_table'] = selected_table
            
            # ═══════════════════════════════════════════════════════════════
            # SECTION BACKTESTING
            # ═══════════════════════════════════════════════════════════════
            if 'ml_results' in st.session_state and st.session_state.get('selected_table') == selected_table:
                st.markdown("---")
                st.markdown("### 💰 BACKTESTING")
                
                with st.expander("📈 Configurer et lancer le backtest", expanded=False):
                    
                    results = st.session_state['ml_results']
                    
                    col_bt1, col_bt2, col_bt3 = st.columns(3)
                    
                    with col_bt1:
                        backtest_model = st.selectbox(
                            "Modèle pour backtest",
                            options=list(results.keys()),
                            index=list(results.keys()).index(st.session_state['best_model']),
                            format_func=lambda x: results[x]['name']
                        )
                    
                    with col_bt2:
                        stop_loss = st.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.5) / 100
                    
                    with col_bt3:
                        take_profit = st.slider("Take Profit (%)", 1.0, 10.0, 4.0, 0.5) / 100
                    
                    if st.button("📈 LANCER LE BACKTEST", type="primary", use_container_width=True):
                        
                        with st.spinner("Exécution du backtest..."):
                            bt = Backtester()
                            bt_results = bt.run(
                                st.session_state['df_test'],
                                results[backtest_model]['predictions'],
                                stop_loss,
                                take_profit
                            )
                        
                        # Métriques
                        st.markdown("#### 📊 Performance")
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric(
                                "Rendement Stratégie", 
                                f"{bt_results['total_return']:.2f}%",
                                delta=f"vs B&H: {bt_results['total_return'] - bt_results['bh_return']:.2f}%"
                            )
                        
                        with col_m2:
                            st.metric("Max Drawdown", f"-{bt_results['max_dd']:.2f}%")
                        
                        with col_m3:
                            st.metric("Nombre de Trades", bt_results['trades'])
                        
                        with col_m4:
                            st.metric("Win Rate", f"{bt_results['win_rate']:.1f}%")
                        
                        # Equity Curve
                        st.markdown("#### 📈 Equity Curve")
                        
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            y=bt_results['equity'],
                            mode='lines',
                            name='Strategy',
                            line=dict(color='#00ff88', width=2)
                        ))
                        fig_eq.add_trace(go.Scatter(
                            y=bt_results['bh_equity'],
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='#ff8800', width=2, dash='dash')
                        ))
                        fig_eq.update_layout(
                            template='plotly_dark',
                            height=400,
                            title='Strategy vs Buy & Hold',
                            xaxis_title='Période',
                            yaxis_title='Capital ($)',
                            legend=dict(x=0.02, y=0.98)
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)
                        
                        # Drawdown
                        equity = bt_results['equity']
                        peak = np.maximum.accumulate(equity)
                        dd = (peak - equity) / peak * 100
                        
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(
                            y=-dd,
                            fill='tozeroy',
                            mode='lines',
                            name='Drawdown',
                            line=dict(color='#ff4444'),
                            fillcolor='rgba(255, 68, 68, 0.3)'
                        ))
                        fig_dd.update_layout(
                            template='plotly_dark',
                            height=250,
                            title='Drawdown',
                            xaxis_title='Période',
                            yaxis_title='Drawdown (%)'
                        )
                        st.plotly_chart(fig_dd, use_container_width=True)
                        
                        # ═══════════════════════════════════════════════════
                        # JOURNAL DE TRADING
                        # ═══════════════════════════════════════════════════
                        st.markdown("#### 📓 Journal de Trading")
                        
                        journal = TradingJournal.create_journal(
                            st.session_state['df_test'],
                            results[backtest_model]['predictions'],
                            stop_loss,
                            take_profit
                        )
                        
                        if not journal.empty:
                            # Stats du journal
                            col_j1, col_j2, col_j3, col_j4 = st.columns(4)
                            
                            with col_j1:
                                avg_win = journal[journal['PnL (%)'] > 0]['PnL (%)'].mean()
                                st.metric("Gain moyen", f"{avg_win:.2f}%" if not pd.isna(avg_win) else "N/A")
                            
                            with col_j2:
                                avg_loss = journal[journal['PnL (%)'] <= 0]['PnL (%)'].mean()
                                st.metric("Perte moyenne", f"{avg_loss:.2f}%" if not pd.isna(avg_loss) else "N/A")
                            
                            with col_j3:
                                avg_duration = journal['Duration'].mean()
                                st.metric("Durée moyenne", f"{avg_duration:.1f} candles")
                            
                            with col_j4:
                                best_trade = journal['PnL (%)'].max()
                                st.metric("Meilleur trade", f"{best_trade:.2f}%")
                            
                            # Répartition des sorties
                            st.markdown("##### 📊 Répartition des sorties")
                            exit_counts = journal['Exit Reason'].value_counts()
                            
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=exit_counts.index,
                                values=exit_counts.values,
                                hole=0.4,
                                marker_colors=['#00ff88', '#ff4444', '#ffaa00']
                            )])
                            fig_pie.update_layout(template='plotly_dark', height=300)
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Tableau des trades
                            st.markdown("##### 📋 Détail des trades")
                            st.dataframe(journal, use_container_width=True, hide_index=True)
                            
                            # Export du journal
                            csv_journal = journal.to_csv(index=False)
                            st.download_button(
                                label="📥 TÉLÉCHARGER LE JOURNAL",
                                data=csv_journal,
                                file_name=f"trading_journal_{selected_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.info("📭 Aucun trade enregistré pendant la période de test")
        
        else:
            st.info("📭 Aucune donnée dans cette table")
    
    st.markdown("---")
    
    # Section 3: Gestion des données
    st.markdown("### 🗑️ GESTION DES DONNÉES")
    
    with st.expander("⚠️ Zone Dangereuse - Suppression de données"):
        st.warning("Attention: Ces actions sont irréversibles!")
        
        delete_table = st.selectbox(
            "Table à nettoyer",
            options=[t['table_name'] for t in tables_info],
            key="delete_table_select"
        )
        
        if st.button("🗑️ VIDER LA TABLE", type="secondary"):
            confirm = st.checkbox(f"Je confirme vouloir supprimer TOUTES les données de {delete_table}")
            if confirm:
                try:
                    supabase.table(delete_table).delete().neq('id', -99999).execute()
                    st.success(f"✅ Table {delete_table} vidée!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {e}")

else:
    st.warning("⚠️ Impossible de récupérer la liste des tables. Vérifiez votre connexion Supabase.")
    
    # Mode manuel
    st.markdown("### 🔧 MODE MANUEL")
    manual_table = st.text_input("Nom de la table à explorer")
    
    if manual_table:
        data = get_table_data(manual_table)
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace;'>
    © 2025 CRYPTO DATABASE VIEWER + ML TRADING | Connected to Supabase
</div>
""", unsafe_allow_html=True)
