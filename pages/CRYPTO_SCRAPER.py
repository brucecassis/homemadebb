"""
CRYPTO ML TRADING PLATFORM v2.0
================================
Plateforme avancée de trading ML avec:
- Optimisation des hyperparamètres (RandomizedSearchCV)
- Walk-Forward Analysis
- Gestion du risque avancée (Kelly Criterion, Trailing Stop, R/R Ratio)
- Features techniques avancées
- Ensemble de modèles
- Filtres de marché
- Métriques avancées (Sharpe, Sortino, Profit Factor, etc.)
- Visualisations avancées

Nice to Have (à venir):
- Sauvegarde/Import des modèles
- Alertes temps réel (Telegram/Discord)
- Multi-timeframe analysis
- Comparaison multi-crypto
"""

import streamlit as st
from supabase import create_client, Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crypto ML Trading Platform",
    page_icon="📊",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLE BLOOMBERG
# ═══════════════════════════════════════════════════════════════════════════════

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
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #FFAA00;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .roadmap-item {
        background: #1a1a1a;
        border: 1px solid #444;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f'''
<div style="background: #FFAA00; padding: 10px 20px; color: #000; font-weight: bold; font-size: 16px; font-family: 'Courier New', monospace; letter-spacing: 2px; margin-bottom: 20px;">
    ⬛ CRYPTO ML TRADING PLATFORM v2.0 | {datetime.now().strftime("%H:%M:%S")} UTC
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


def get_table_data(table_name, limit=1000):
    """Récupère les données d'une table (pour aperçu)"""
    try:
        response = supabase.table(table_name).select('*').order('open_time').limit(limit).execute()
        return response.data
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return None


def get_all_table_data(table_name, progress_callback=None):
    """Récupère TOUTES les données d'une table avec pagination"""
    all_data = []
    batch_size = 1000
    offset = 0
    try:
        while True:
            response = supabase.table(table_name).select('*').order('open_time').range(offset, offset + batch_size - 1).execute()
            if not response.data:
                break
            all_data.extend(response.data)
            offset += batch_size
            if progress_callback:
                progress_callback(len(all_data))
            if len(response.data) < batch_size:
                break
        return all_data
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return all_data if all_data else None


def get_table_schema(table_name):
    """Récupère le schéma d'une table"""
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
        st.error(f"Erreur: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING AVANCÉ
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedFeatureEngineer:
    """Génère les features techniques avancées pour le ML"""
    
    @staticmethod
    def add_all_features(df, include_advanced=True):
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        for period in [9, 12, 21, 26, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Stochastic RSI
        rsi_min = df['rsi'].rolling(14).min()
        rsi_max = df['rsi'].rolling(14).max()
        df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
        df['stoch_rsi_k'] = df['stoch_rsi'].rolling(3).mean()
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        tr = pd.concat([df['high'] - df['low'], 
                        abs(df['high'] - df['close'].shift()), 
                        abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Returns
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_change'] = df['volume'].pct_change()
        
        # Signals binaires
        df['ema_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        if include_advanced:
            df = AdvancedFeatureEngineer.add_advanced_features(df)
        
        return df
    
    @staticmethod
    def add_advanced_features(df):
        """Features avancées supplémentaires"""
        df = df.copy()
        
        # Volatilité
        df['volatility_20'] = df['return_1'].rolling(20).std() * np.sqrt(252)
        df['volatility_50'] = df['return_1'].rolling(50).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
        
        # Parkinson Volatility
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(df['high'] / df['low'])) ** 2).rolling(20).mean()
        )
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Williams %R
        highest_high = df['high'].rolling(14).max()
        lowest_low = df['low'].rolling(14).min()
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        # CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_deviation = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * mean_deviation)
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([df['high'] - df['low'], 
                        abs(df['high'] - df['close'].shift()), 
                        abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Trend Strength
        df['trend_strength'] = abs(df['ema_9'] - df['ema_21']) / df['atr']
        
        # Higher Highs / Lower Lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['hh_count'] = df['higher_high'].rolling(10).sum()
        df['ll_count'] = df['lower_low'].rolling(10).sum()
        
        # OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        df['obv_signal'] = (df['obv'] > df['obv_ema']).astype(int)
        
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
        
        # Candle patterns
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open'] * 100
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['is_doji'] = (abs(df['body']) < df['candle_range'] * 0.1).astype(int)
        
        # Market Regime
        vol_percentile = df['volatility_20'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        df['high_volatility'] = (vol_percentile > 0.8).astype(int)
        df['low_volatility'] = (vol_percentile < 0.2).astype(int)
        
        # Trend regime
        df['uptrend'] = ((df['close'] > df['sma_50']) & (df['sma_20'] > df['sma_50'])).astype(int)
        df['downtrend'] = ((df['close'] < df['sma_50']) & (df['sma_20'] < df['sma_50'])).astype(int)
        df['ranging'] = ((~df['uptrend'].astype(bool)) & (~df['downtrend'].astype(bool))).astype(int)
        
        return df
    
    @staticmethod
    def create_target(df, horizon=1, threshold=0.0):
        """Crée la variable cible multi-classe"""
        df = df.copy()
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = 1  # Neutre
        df.loc[df['future_return'] > threshold, 'target'] = 2   # Long
        df.loc[df['future_return'] < -threshold, 'target'] = 0  # Short
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION DES HYPERPARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════════

class HyperparameterOptimizer:
    """Optimisation des hyperparamètres avec RandomizedSearchCV"""
    
    @staticmethod
    def get_param_distributions():
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'newton-cg']
            }
        }
    
    @staticmethod
    def optimize(model_type, X_train, y_train, n_iter=20, cv=3):
        """Optimise les hyperparamètres d'un modèle"""
        param_dist = HyperparameterOptimizer.get_param_distributions()[model_type]
        
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
        elif model_type == 'logistic':
            base_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        
        tscv = TimeSeriesSplit(n_splits=cv)
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring='f1_weighted',
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, search.best_score_


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardAnalyzer:
    """Walk-Forward Analysis pour validation robuste"""
    
    def __init__(self, n_splits=5, train_ratio=0.8):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
    
    def run(self, df, features, target, model, scaler):
        """Exécute la walk-forward analysis"""
        df_clean = df.dropna(subset=features + [target])
        X = df_clean[features].values
        y = df_clean[target].values
        
        n = len(X)
        fold_size = n // self.n_splits
        
        all_predictions = []
        all_actuals = []
        fold_metrics = []
        
        for i in range(self.n_splits):
            train_end = int(fold_size * (i + 1) * self.train_ratio)
            test_start = train_end
            test_end = min(fold_size * (i + 2), n)
            
            if test_start >= n or test_end <= test_start:
                continue
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            
            if len(X_train) < 100 or len(X_test) < 10:
                continue
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            
            all_predictions.extend(pred)
            all_actuals.extend(y_test)
            
            fold_metrics.append({
                'fold': i + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy_score(y_test, pred),
                'f1': f1_score(y_test, pred, average='weighted', zero_division=0)
            })
        
        overall_metrics = {
            'accuracy': accuracy_score(all_actuals, all_predictions),
            'precision': precision_score(all_actuals, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_actuals, all_predictions, average='weighted', zero_division=0),
            'f1': f1_score(all_actuals, all_predictions, average='weighted', zero_division=0),
            'predictions': np.array(all_predictions),
            'actuals': np.array(all_actuals)
        }
        
        return overall_metrics, fold_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE DE MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleBuilder:
    """Construction d'ensembles de modèles"""
    
    @staticmethod
    def weighted_prediction(predictions_dict, weights_dict):
        """Prédiction pondérée basée sur les poids"""
        n_samples = len(list(predictions_dict.values())[0])
        weighted_preds = np.zeros((n_samples, 3))
        
        for name, preds in predictions_dict.items():
            weight = weights_dict.get(name, 1.0)
            for i, pred in enumerate(preds):
                weighted_preds[i, int(pred)] += weight
        
        return np.argmax(weighted_preds, axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# FILTRES DE MARCHÉ
# ═══════════════════════════════════════════════════════════════════════════════

class MarketFilter:
    """Filtres de conditions de marché"""
    
    @staticmethod
    def apply_filters(df, predictions, filters_config):
        """Applique les filtres de marché aux prédictions"""
        filtered_preds = predictions.copy()
        
        if filters_config.get('trend_filter', False):
            if 'uptrend' in df.columns:
                uptrend_mask = df['uptrend'].values[-len(predictions):]
                filtered_preds = np.where(uptrend_mask & (filtered_preds == 0), 1, filtered_preds)
            if 'downtrend' in df.columns:
                downtrend_mask = df['downtrend'].values[-len(predictions):]
                filtered_preds = np.where(downtrend_mask & (filtered_preds == 2), 1, filtered_preds)
        
        if filters_config.get('volatility_filter', False):
            if 'high_volatility' in df.columns:
                high_vol_mask = df['high_volatility'].values[-len(predictions):]
                filtered_preds = np.where(high_vol_mask, 1, filtered_preds)
        
        if filters_config.get('volume_filter', False):
            if 'volume_ratio' in df.columns:
                low_vol_mask = df['volume_ratio'].values[-len(predictions):] < 0.5
                filtered_preds = np.where(low_vol_mask, 1, filtered_preds)
        
        if filters_config.get('adx_filter', False):
            if 'adx' in df.columns:
                adx = df['adx'].values[-len(predictions):]
                adx_threshold = filters_config.get('adx_threshold', 25)
                weak_trend = adx < adx_threshold
                filtered_preds = np.where(weak_trend, 1, filtered_preds)
        
        return filtered_preds


# ═══════════════════════════════════════════════════════════════════════════════
# GESTION DU RISQUE AVANCÉE
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedRiskManager:
    """Gestion du risque avancée"""
    
    @staticmethod
    def kelly_criterion(win_rate, avg_win, avg_loss):
        """Calcule la fraction de Kelly"""
        if avg_loss == 0:
            return 0
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        return max(0, min(kelly, 1))
    
    @staticmethod
    def calculate_rr_ratio(entry_price, stop_loss, take_profit, is_long=True):
        """Calcule le ratio Risk/Reward"""
        if is_long:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        if risk <= 0:
            return 0
        return reward / risk


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTER AVANCÉ
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedBacktester:
    """Moteur de backtesting avancé avec gestion du risque"""
    
    def __init__(self, capital=10000, commission=0.001):
        self.initial_capital = capital
        self.commission = commission
    
    def run(self, df, predictions, config):
        """Exécute le backtest avec configuration avancée"""
        df = df.iloc[-len(predictions):].copy()
        
        # Extraire les dates si disponibles
        if 'open_time' in df.columns:
            dates = pd.to_datetime(df['open_time']).values
        else:
            dates = np.arange(len(df))
        
        # Extraire les prix pour le Buy & Hold
        prices = df['close'].astype(float).values
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        highest_since_entry = 0
        lowest_since_entry = float('inf')
        entry_idx = 0
        
        equity = [capital]
        trades = []
        daily_returns = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            price = float(row['close'])
            high = float(row['high'])
            low = float(row['low'])
            pred = predictions[i]
            
            # Mise à jour trailing
            if position == 1:
                highest_since_entry = max(highest_since_entry, high)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, low)
            
            # Gestion position LONG
            if position == 1:
                pnl = (price - entry_price) / entry_price
                
                # Trailing Stop
                if config.get('trailing_stop', False):
                    trailing_stop_price = highest_since_entry * (1 - config.get('trailing_pct', 0.01))
                    if price <= trailing_stop_price:
                        capital *= (1 + pnl - self.commission)
                        trades.append({'pnl': pnl, 'type': 'LONG', 'exit': 'Trailing Stop', 'duration': i - entry_idx})
                        position = 0
                        continue
                
                if pnl <= -config['stop_loss']:
                    capital *= (1 + pnl - self.commission)
                    trades.append({'pnl': pnl, 'type': 'LONG', 'exit': 'Stop Loss', 'duration': i - entry_idx})
                    position = 0
                elif pnl >= config['take_profit']:
                    capital *= (1 + pnl - self.commission)
                    trades.append({'pnl': pnl, 'type': 'LONG', 'exit': 'Take Profit', 'duration': i - entry_idx})
                    position = 0
                elif pred != 2:
                    capital *= (1 + pnl - self.commission)
                    trades.append({'pnl': pnl, 'type': 'LONG', 'exit': 'Signal', 'duration': i - entry_idx})
                    position = 0
            
            # Gestion position SHORT
            elif position == -1:
                pnl = (entry_price - price) / entry_price
                
                if config.get('trailing_stop', False):
                    trailing_stop_price = lowest_since_entry * (1 + config.get('trailing_pct', 0.01))
                    if price >= trailing_stop_price:
                        capital *= (1 + pnl - self.commission)
                        trades.append({'pnl': pnl, 'type': 'SHORT', 'exit': 'Trailing Stop', 'duration': i - entry_idx})
                        position = 0
                        continue
                
                if pnl <= -config['stop_loss']:
                    capital *= (1 + pnl - self.commission)
                    trades.append({'pnl': pnl, 'type': 'SHORT', 'exit': 'Stop Loss', 'duration': i - entry_idx})
                    position = 0
                elif pnl >= config['take_profit']:
                    capital *= (1 + pnl - self.commission)
                    trades.append({'pnl': pnl, 'type': 'SHORT', 'exit': 'Take Profit', 'duration': i - entry_idx})
                    position = 0
                elif pred != 0:
                    capital *= (1 + pnl - self.commission)
                    trades.append({'pnl': pnl, 'type': 'SHORT', 'exit': 'Signal', 'duration': i - entry_idx})
                    position = 0
            
            # Ouverture de position
            elif position == 0:
                min_rr = config.get('min_rr_ratio', 0)
                
                if pred == 2:  # LONG
                    sl_price = price * (1 - config['stop_loss'])
                    tp_price = price * (1 + config['take_profit'])
                    rr = AdvancedRiskManager.calculate_rr_ratio(price, sl_price, tp_price, True)
                    
                    if rr >= min_rr:
                        position = 1
                        entry_price = price
                        highest_since_entry = high
                        entry_idx = i
                        capital *= (1 - self.commission)
                
                elif pred == 0:  # SHORT
                    sl_price = price * (1 + config['stop_loss'])
                    tp_price = price * (1 - config['take_profit'])
                    rr = AdvancedRiskManager.calculate_rr_ratio(price, sl_price, tp_price, False)
                    
                    if rr >= min_rr:
                        position = -1
                        entry_price = price
                        lowest_since_entry = low
                        entry_idx = i
                        capital *= (1 - self.commission)
            
            equity.append(capital)
            if i > 0:
                daily_returns.append((equity[-1] / equity[-2]) - 1)
        
        equity = np.array(equity[1:])
        # Buy & Hold: si on avait acheté au prix initial et gardé
        bh_equity = self.initial_capital * (prices / prices[0])
        
        return self._calculate_metrics(equity, bh_equity, trades, daily_returns, dates, prices)
    
    def _calculate_metrics(self, equity, bh_equity, trades, daily_returns, dates, prices):
        """Calcule toutes les métriques de performance"""
        daily_returns = np.array(daily_returns) if daily_returns else np.array([0])
        
        total_return = (equity[-1] / self.initial_capital - 1) * 100
        bh_return = (bh_equity[-1] / self.initial_capital - 1) * 100
        
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max() * 100
        
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]
        long_trades = [t for t in trades if t['type'] == 'LONG']
        short_trades = [t for t in trades if t['type'] == 'SHORT']
        
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl'] for t in win_trades]) * 100 if win_trades else 0
        avg_loss = np.mean([t['pnl'] for t in loss_trades]) * 100 if loss_trades else 0
        
        gross_profit = sum([t['pnl'] for t in win_trades]) if win_trades else 0
        gross_loss = abs(sum([t['pnl'] for t in loss_trades])) if loss_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) != 0 else 0
        
        downside_returns = daily_returns[daily_returns < 0]
        sortino = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) != 0 else 0
        
        recovery_factor = total_return / max_dd if max_dd != 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        kelly = AdvancedRiskManager.kelly_criterion(
            win_rate / 100,
            avg_win / 100 if avg_win > 0 else 0,
            avg_loss / 100 if avg_loss < 0 else 0.01
        )
        
        return {
            'equity': equity,
            'bh_equity': bh_equity,
            'dates': dates,
            'prices': prices,
            'total_return': total_return,
            'bh_return': bh_return,
            'max_dd': max_dd,
            'trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'recovery_factor': recovery_factor,
            'expectancy': expectancy,
            'kelly_fraction': kelly * 100,
            'trades_detail': trades
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS AVANCÉES
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedVisualizer:
    """Visualisations avancées pour l'analyse"""
    
    @staticmethod
    def create_monthly_heatmap(df, equity):
        """Crée une heatmap des rendements mensuels"""
        if 'open_time' not in df.columns:
            return None
        
        df_viz = df.copy()
        df_viz['open_time'] = pd.to_datetime(df_viz['open_time'])
        df_viz = df_viz.iloc[-len(equity):]
        df_viz['equity'] = equity
        df_viz['return'] = df_viz['equity'].pct_change()
        df_viz['month'] = df_viz['open_time'].dt.month
        df_viz['year'] = df_viz['open_time'].dt.year
        
        monthly = df_viz.groupby(['year', 'month'])['return'].sum() * 100
        monthly_df = monthly.unstack()
        
        if monthly_df.empty:
            return None
        
        fig = px.imshow(
            monthly_df.values,
            x=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'][:monthly_df.shape[1]],
            y=monthly_df.index.astype(str),
            color_continuous_scale='RdYlGn',
            aspect='auto',
            labels={'color': 'Return (%)'}
        )
        fig.update_layout(title='Rendements Mensuels (%)', template='plotly_dark', height=300)
        return fig
    
    @staticmethod
    def create_pnl_distribution(trades):
        """Crée la distribution des PnL"""
        if not trades:
            return None
        
        pnls = [t['pnl'] * 100 for t in trades]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pnls, nbinsx=30, marker_color='#FFAA00', opacity=0.7))
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=np.mean(pnls), line_dash="solid", line_color="#00ff88",
                     annotation_text=f"Moy: {np.mean(pnls):.2f}%")
        fig.update_layout(title='Distribution des PnL', xaxis_title='PnL (%)', yaxis_title='Fréquence',
                         template='plotly_dark', height=300)
        return fig
    
    @staticmethod
    def create_feature_correlation(df, features, target='target'):
        """Crée une matrice de corrélation features vs target"""
        df_clean = df.dropna(subset=features + [target])
        
        correlations = []
        for f in features:
            corr = df_clean[f].corr(df_clean[target])
            correlations.append({'feature': f, 'correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
        
        fig = go.Figure(go.Bar(
            x=corr_df['correlation'],
            y=corr_df['feature'],
            orientation='h',
            marker_color=['#00ff88' if c > 0 else '#ff4444' for c in corr_df['correlation']]
        ))
        fig.update_layout(title='Corrélation Features / Target', xaxis_title='Corrélation',
                         template='plotly_dark', height=400)
        return fig
    
    @staticmethod
    def create_confusion_matrix_plot(y_true, y_pred):
        """Crée une matrice de confusion visuelle"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            x=['Short', 'Neutre', 'Long'],
            y=['Short', 'Neutre', 'Long'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(title='Matrice de Confusion', xaxis_title='Prédit', yaxis_title='Réel',
                         template='plotly_dark', height=350)
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedMLModels:
    """Collection de modèles ML avancés"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def get_models(self):
        return {
            'random_forest': ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            'xgboost': ('XGBoost', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss', verbosity=0)),
            'gradient_boosting': ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            'logistic': ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'))
        }
    
    def train(self, df, features, target='target', test_size=0.2):
        df_clean = df.dropna(subset=features + [target])
        X = df_clean[features].values
        y = df_clean[target].values
        
        split = int(len(X) * (1 - test_size))
        X_train = self.scaler.fit_transform(X[:split])
        X_test = self.scaler.transform(X[split:])
        y_train = y[:split]
        y_test = y[split:]
        
        return X_train, X_test, y_train, y_test, df_clean.iloc[split:]
    
    def evaluate(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, pred, average='weighted', zero_division=0),
            'predictions': pred,
            'actuals': y_test,
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING JOURNAL
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedTradingJournal:
    """Journal de trading avancé"""
    
    @staticmethod
    def create_journal(df_test, predictions, config):
        df = df_test.iloc[-len(predictions):].copy()
        df['prediction'] = predictions
        
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        entry_idx = 0
        trade_type = None
        highest_price = 0
        lowest_price = float('inf')
        
        for i, (idx, row) in enumerate(df.iterrows()):
            price = float(row['close'])
            high = float(row['high'])
            low = float(row['low'])
            pred = row['prediction']
            date = row['open_time'] if 'open_time' in row else idx
            
            if position == 0:
                if pred == 2:
                    position, trade_type = 1, '🟢 LONG'
                    entry_price, entry_date, entry_idx = price, date, i
                    highest_price = high
                elif pred == 0:
                    position, trade_type = -1, '🔴 SHORT'
                    entry_price, entry_date, entry_idx = price, date, i
                    lowest_price = low
            
            elif position != 0:
                if position == 1:
                    highest_price = max(highest_price, high)
                    pnl_pct = (price - entry_price) / entry_price
                    max_favorable = (highest_price - entry_price) / entry_price
                else:
                    lowest_price = min(lowest_price, low)
                    pnl_pct = (entry_price - price) / entry_price
                    max_favorable = (entry_price - lowest_price) / entry_price
                
                exit_reason = None
                should_close = False
                
                if pnl_pct <= -config['stop_loss']:
                    exit_reason = "🛑 Stop Loss"
                    should_close = True
                elif pnl_pct >= config['take_profit']:
                    exit_reason = "🎯 Take Profit"
                    should_close = True
                elif (position == 1 and pred != 2) or (position == -1 and pred != 0):
                    exit_reason = "📊 Signal"
                    should_close = True
                
                if should_close:
                    risk = config['stop_loss']
                    r_multiple = pnl_pct / risk if risk > 0 else 0
                    
                    trades.append({
                        'Type': trade_type,
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': round(entry_price, 4),
                        'Exit Price': round(price, 4),
                        'PnL (%)': round(pnl_pct * 100, 2),
                        'Max Favorable (%)': round(max_favorable * 100, 2),
                        'R Multiple': round(r_multiple, 2),
                        'Duration': i - entry_idx,
                        'Exit Reason': exit_reason,
                        'Result': '✅ Win' if pnl_pct > 0 else '❌ Loss'
                    })
                    position = 0
                    highest_price = 0
                    lowest_price = float('inf')
        
        return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

st.title("📊 CRYPTO ML TRADING PLATFORM")

# Sidebar navigation
st.sidebar.markdown("### 🧭 NAVIGATION")
page = st.sidebar.radio(
    "",
    ["📂 Database", "🤖 ML Training", "📈 Backtesting", "📊 Analytics", "🗺️ Roadmap"],
    index=0
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📂 Database":
    st.markdown("### 📂 TABLES DANS SUPABASE")
    
    if st.button("🔄 RAFRAÎCHIR"):
        st.cache_data.clear()
        st.rerun()
    
    tables_info = get_table_list()
    
    if tables_info:
        cols = st.columns(min(len(tables_info), 3))
        for idx, table in enumerate(tables_info):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="table-card">
                    <h4 style="color: #FFAA00; margin: 0;">📋 {table['table_name']}</h4>
                    <p style="color: #888; margin: 5px 0;">Lignes: <span style="color: #FFAA00;">{table['row_count']:,}</span></p>
                    <p style="color: #666; margin: 0; font-size: 12px;">{table['symbol']} | {table['timeframe']} | {table['period_days']}j</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔍 EXPLORER UNE TABLE")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_table = st.selectbox("Sélectionner une table", [t['table_name'] for t in tables_info])
        with col2:
            limit = st.number_input("Lignes (aperçu)", min_value=10, max_value=10000, value=100, step=100)
        
        if selected_table:
            st.session_state['selected_table'] = selected_table
            st.session_state['table_info'] = next((t for t in tables_info if t['table_name'] == selected_table), None)
            
            schema = get_table_schema(selected_table)
            if schema:
                with st.expander("📐 Schéma"):
                    st.dataframe(pd.DataFrame(schema), use_container_width=True, hide_index=True)
            
            data = get_table_data(selected_table, limit)
            if data:
                df = pd.DataFrame(data)
                
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Colonnes", len(df.columns))
                col_s2.metric("Lignes affichées", len(df))
                table_info = st.session_state.get('table_info', {})
                col_s3.metric("Total", f"{table_info.get('row_count', len(df)):,}")
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                if all(col in df.columns for col in ['open_time', 'open', 'high', 'low', 'close']):
                    with st.expander("📈 Graphique"):
                        df['open_time'] = pd.to_datetime(df['open_time'])
                        df = df.sort_values('open_time')
                        fig = go.Figure(data=[go.Candlestick(
                            x=df['open_time'], open=df['open'], high=df['high'], low=df['low'], close=df['close']
                        )])
                        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button("📥 TÉLÉCHARGER CSV", data=csv,
                                  file_name=f"{selected_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                  mime="text/csv")
    else:
        st.warning("⚠️ Aucune table disponible.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Training":
    st.markdown("### 🤖 ML TRAINING CENTER")
    
    tables_info = get_table_list()
    
    if not tables_info:
        st.warning("⚠️ Aucune table disponible.")
    else:
        selected_table = st.selectbox("📋 Table", [t['table_name'] for t in tables_info])
        table_info = next((t for t in tables_info if t['table_name'] == selected_table), {})
        st.info(f"📊 **{table_info.get('symbol', '')}** | {table_info.get('timeframe', '')} | {table_info.get('row_count', 0):,} lignes")
        
        st.markdown("---")
        st.markdown("#### ⚙️ Configuration ML")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            horizon = st.selectbox("Horizon", [1, 2, 3, 5, 10, 20], index=0)
        with col2:
            threshold = st.slider("Seuil (%)", 0.0, 2.0, 0.1, 0.05) / 100
        with col3:
            test_pct = st.slider("Test (%)", 10, 40, 20)
        with col4:
            include_advanced = st.checkbox("Features avancées", value=True)
        
        with st.expander("🔧 Options avancées"):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                optimize_params = st.checkbox("🎯 Optimiser hyperparamètres", value=False)
                n_iter = st.slider("Itérations", 10, 50, 20) if optimize_params else 20
                use_walk_forward = st.checkbox("📊 Walk-Forward Analysis", value=False)
                wf_splits = st.slider("Splits WF", 3, 10, 5) if use_walk_forward else 5
            with col_a2:
                use_ensemble = st.checkbox("🤝 Ensemble", value=False)
                st.markdown("**Filtres:**")
                trend_filter = st.checkbox("Tendance", value=False)
                volatility_filter = st.checkbox("Volatilité", value=False)
                adx_filter = st.checkbox("ADX", value=False)
                adx_threshold = st.slider("Seuil ADX", 15, 40, 25) if adx_filter else 25
        
        models_to_train = st.multiselect(
            "Modèles",
            ['random_forest', 'xgboost', 'gradient_boosting', 'logistic'],
            default=['random_forest', 'xgboost'],
            format_func=lambda x: {'random_forest': '🌲 Random Forest', 'xgboost': '⚡ XGBoost',
                                  'gradient_boosting': '📈 Gradient Boosting', 'logistic': '📐 Logistic'}[x]
        )
        
        if st.button("🚀 LANCER L'ENTRAÎNEMENT", type="primary", use_container_width=True):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            total_rows = table_info.get('row_count', 0)
            
            def update_progress(count):
                progress_text.text(f"📥 Chargement... {count:,} / {total_rows:,}")
                if total_rows > 0:
                    progress_bar.progress(min(count / total_rows, 1.0))
            
            with st.spinner("Chargement..."):
                full_data = get_all_table_data(selected_table, update_progress)
                df_ml = pd.DataFrame(full_data)
            
            progress_bar.empty()
            progress_text.empty()
            st.success(f"✅ **{len(df_ml):,}** lignes chargées!")
            
            with st.spinner("🔧 Features..."):
                fe = AdvancedFeatureEngineer()
                df_ml = fe.add_all_features(df_ml, include_advanced)
                df_ml = fe.create_target(df_ml, horizon, threshold)
            
            if include_advanced:
                features = ['sma_10', 'sma_20', 'sma_50', 'ema_9', 'ema_21', 'macd', 'macd_signal',
                           'macd_histogram', 'rsi', 'stoch_rsi_k', 'bb_position', 'bb_width', 'atr_percent',
                           'return_1', 'return_5', 'return_10', 'volume_ratio', 'volatility_20',
                           'roc_5', 'momentum_10', 'williams_r', 'cci', 'adx', 'mfi', 'obv_signal',
                           'trend_strength', 'ema_cross', 'macd_bullish', 'uptrend', 'downtrend']
            else:
                features = ['sma_10', 'sma_20', 'ema_9', 'ema_21', 'macd', 'macd_signal', 'rsi',
                           'bb_position', 'atr', 'return_1', 'return_5', 'volume_ratio', 'ema_cross']
            
            features = [f for f in features if f in df_ml.columns]
            
            target_counts = df_ml['target'].value_counts()
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("🔴 SHORT", f"{target_counts.get(0, 0):,}")
            col_t2.metric("⚪ NEUTRE", f"{target_counts.get(1, 0):,}")
            col_t3.metric("🟢 LONG", f"{target_counts.get(2, 0):,}")
            
            ml = AdvancedMLModels()
            X_train, X_test, y_train, y_test, df_test = ml.train(df_ml, features, 'target', test_pct/100)
            st.success(f"📊 Train: **{len(X_train):,}** | Test: **{len(X_test):,}**")
            
            results = {}
            optimized_params = {}
            available_models = ml.get_models()
            model_progress = st.progress(0)
            status_text = st.empty()
            
            for i, key in enumerate(models_to_train):
                name, model = available_models[key]
                status_text.text(f"🔄 {name}...")
                
                if optimize_params:
                    status_text.text(f"🎯 Optimisation {name}...")
                    model, best_params, _ = HyperparameterOptimizer.optimize(key, X_train, y_train, n_iter)
                    optimized_params[key] = best_params
                
                results[key] = ml.evaluate(model, X_train, X_test, y_train, y_test)
                results[key]['name'] = name
                results[key]['model'] = model
                model_progress.progress((i + 1) / len(models_to_train))
            
            model_progress.empty()
            status_text.empty()
            
            if use_walk_forward:
                st.markdown("#### 📊 Walk-Forward Analysis")
                wf_data = []
                for key in models_to_train:
                    model = results[key]['model']
                    wf = WalkForwardAnalyzer(n_splits=wf_splits)
                    wf_metrics, _ = wf.run(df_ml, features, 'target', model, StandardScaler())
                    wf_data.append({
                        'Modèle': results[key]['name'],
                        'WF Accuracy': f"{wf_metrics['accuracy']:.2%}",
                        'WF F1': f"{wf_metrics['f1']:.2%}",
                        'Std Accuracy': f"{results[key]['accuracy']:.2%}"
                    })
                st.dataframe(pd.DataFrame(wf_data), use_container_width=True, hide_index=True)
            
            if use_ensemble and len(models_to_train) > 1:
                st.markdown("#### 🤝 Ensemble")
                pred_dict = {k: results[k]['predictions'] for k in models_to_train}
                weight_dict = {k: results[k]['f1'] for k in models_to_train}
                ensemble_pred = EnsembleBuilder.weighted_prediction(pred_dict, weight_dict)
                ens_acc = accuracy_score(y_test, ensemble_pred)
                ens_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
                st.success(f"🤝 Ensemble - Accuracy: **{ens_acc:.2%}** | F1: **{ens_f1:.2%}**")
                results['ensemble'] = {'name': 'Ensemble', 'accuracy': ens_acc, 'f1': ens_f1,
                                       'predictions': ensemble_pred, 'actuals': y_test}
            
            st.markdown("#### 📊 Résultats")
            res_df = pd.DataFrame([{
                'Modèle': results[k]['name'],
                'Accuracy': f"{results[k]['accuracy']:.2%}",
                'Precision': f"{results[k].get('precision', 0):.2%}",
                'Recall': f"{results[k].get('recall', 0):.2%}",
                'F1': f"{results[k]['f1']:.2%}"
            } for k in results.keys()])
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig_comp = go.Figure()
                for metric in ['accuracy', 'f1']:
                    fig_comp.add_trace(go.Bar(name=metric.title(),
                                             x=[results[k]['name'] for k in results],
                                             y=[results[k].get(metric, 0) for k in results]))
                fig_comp.update_layout(barmode='group', template='plotly_dark', height=350, title='Comparaison')
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with col_v2:
                best_key = max(results, key=lambda k: results[k]['f1'])
                if 'actuals' in results[best_key]:
                    fig_cm = AdvancedVisualizer.create_confusion_matrix_plot(
                        results[best_key]['actuals'], results[best_key]['predictions'])
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            best_key = max(results, key=lambda k: results[k]['f1'])
            if results[best_key].get('feature_importance') is not None:
                with st.expander("📈 Feature Importance"):
                    importance = results[best_key]['feature_importance']
                    top_n = min(20, len(features))
                    indices = np.argsort(importance)[-top_n:]
                    fig_imp = go.Figure(go.Bar(x=importance[indices], y=[features[i] for i in indices],
                                              orientation='h', marker_color='#FFAA00'))
                    fig_imp.update_layout(template='plotly_dark', height=500, title=f'Top {top_n} Features')
                    st.plotly_chart(fig_imp, use_container_width=True)
            
            if optimized_params:
                with st.expander("🎯 Hyperparamètres optimisés"):
                    for key, params in optimized_params.items():
                        st.markdown(f"**{results[key]['name']}:** `{params}`")
            
            st.session_state['ml_results'] = results
            st.session_state['df_test'] = df_test
            st.session_state['df_ml'] = df_ml
            st.session_state['features'] = features
            st.session_state['best_model'] = best_key
            st.session_state['selected_table_ml'] = selected_table
            st.session_state['filters_config'] = {
                'trend_filter': trend_filter, 'volatility_filter': volatility_filter,
                'adx_filter': adx_filter, 'adx_threshold': adx_threshold, 'volume_filter': False
            }
            st.success("✅ Allez sur **📈 Backtesting**!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Backtesting":
    st.markdown("### 📈 BACKTESTING AVANCÉ")
    
    if 'ml_results' not in st.session_state:
        st.warning("⚠️ Entraînez d'abord un modèle sur **🤖 ML Training**.")
    else:
        results = st.session_state['ml_results']
        df_test = st.session_state['df_test']
        df_ml = st.session_state.get('df_ml')
        
        st.info(f"📊 Backtest sur **{len(df_test):,}** candles")
        
        st.markdown("#### ⚙️ Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            backtest_model = st.selectbox("Modèle", list(results.keys()),
                                         format_func=lambda x: results[x]['name'])
        with col2:
            stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0, 0.5) / 100
        with col3:
            take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 4.0, 0.5) / 100
        
        with st.expander("🔧 Gestion du risque avancée"):
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                trailing_stop = st.checkbox("Trailing Stop", value=False)
                trailing_pct = st.slider("Trailing (%)", 0.5, 5.0, 1.0, 0.5) / 100 if trailing_stop else 0.01
            with col_r2:
                min_rr = st.slider("Min R/R", 0.0, 3.0, 1.5, 0.5)
            with col_r3:
                apply_filters = st.checkbox("Filtres marché", value=False)
        
        if st.button("📈 LANCER LE BACKTEST", type="primary", use_container_width=True):
            predictions = results[backtest_model]['predictions']
            
            if apply_filters and df_ml is not None:
                filters_config = st.session_state.get('filters_config', {})
                predictions = MarketFilter.apply_filters(df_ml, predictions, filters_config)
            
            bt_config = {
                'stop_loss': stop_loss, 'take_profit': take_profit,
                'trailing_stop': trailing_stop, 'trailing_pct': trailing_pct, 'min_rr_ratio': min_rr
            }
            
            with st.spinner("Backtest..."):
                bt = AdvancedBacktester()
                bt_results = bt.run(df_test, predictions, bt_config)
            
            st.session_state['bt_results'] = bt_results
            st.session_state['bt_config'] = bt_config
            st.session_state['bt_predictions'] = predictions
        
        if 'bt_results' in st.session_state:
            bt_results = st.session_state['bt_results']
            
            st.markdown("#### 📊 Performance")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            delta = bt_results['total_return'] - bt_results['bh_return']
            col_m1.metric("💰 Return", f"{bt_results['total_return']:.2f}%", delta=f"vs B&H: {delta:+.2f}%")
            col_m2.metric("📉 Max DD", f"-{bt_results['max_dd']:.2f}%")
            col_m3.metric("📊 Sharpe", f"{bt_results['sharpe_ratio']:.2f}")
            col_m4.metric("📈 Profit Factor", f"{bt_results['profit_factor']:.2f}")
            col_m5.metric("🎯 Win Rate", f"{bt_results['win_rate']:.1f}%")
            
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            col_s1.metric("Sortino", f"{bt_results['sortino_ratio']:.2f}")
            col_s2.metric("Recovery", f"{bt_results['recovery_factor']:.2f}")
            col_s3.metric("Expectancy", f"{bt_results['expectancy']:.2f}%")
            col_s4.metric("Kelly", f"{bt_results['kelly_fraction']:.1f}%")
            col_s5.metric("Trades", bt_results['trades'])
            
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            col_t1.metric("🟢 Long", bt_results['long_trades'])
            col_t2.metric("🔴 Short", bt_results['short_trades'])
            col_t3.metric("✅ Avg Win", f"{bt_results['avg_win']:.2f}%")
            col_t4.metric("❌ Avg Loss", f"{bt_results['avg_loss']:.2f}%")
            
            st.markdown("---")
            st.markdown("#### 📈 Equity Curve")
            
            equity = bt_results['equity']
            bh_equity = bt_results['bh_equity']
            dates = bt_results.get('dates')
            prices = bt_results.get('prices')
            
            # Utiliser les dates si disponibles
            if dates is not None and len(dates) == len(equity):
                x_axis = dates
            else:
                x_axis = np.arange(len(equity))
            
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=x_axis, y=equity, mode='lines', name='Strategy', line=dict(color='#00ff88', width=2)))
            fig_eq.add_trace(go.Scatter(x=x_axis, y=bh_equity, mode='lines', name='Buy & Hold', line=dict(color='#ff8800', width=2, dash='dash')))
            fig_eq.update_layout(template='plotly_dark', height=400, title='Strategy vs Buy & Hold ($)',
                               xaxis_title='Date', yaxis_title='Capital ($)', legend=dict(x=0.02, y=0.98))
            st.plotly_chart(fig_eq, use_container_width=True)
            
            # Graphique des prix BTC pour comparaison
            if prices is not None:
                with st.expander("📊 Prix de l'actif (référence)"):
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=x_axis, y=prices, mode='lines', name='Prix', line=dict(color='#FFAA00', width=1)))
                    fig_price.update_layout(template='plotly_dark', height=300, title='Prix de l\'actif',
                                          xaxis_title='Date', yaxis_title='Prix ($)')
                    st.plotly_chart(fig_price, use_container_width=True)
            
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=x_axis, y=-dd, fill='tozeroy', mode='lines', name='Drawdown',
                                       line=dict(color='#ff4444'), fillcolor='rgba(255, 68, 68, 0.3)'))
            fig_dd.update_layout(template='plotly_dark', height=250, title='Drawdown', xaxis_title='Date', yaxis_title='Drawdown (%)')
            st.plotly_chart(fig_dd, use_container_width=True)
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig_pnl = AdvancedVisualizer.create_pnl_distribution(bt_results['trades_detail'])
                if fig_pnl:
                    st.plotly_chart(fig_pnl, use_container_width=True)
            with col_v2:
                fig_heat = AdvancedVisualizer.create_monthly_heatmap(df_test, equity)
                if fig_heat:
                    st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown("#### 📓 Journal de Trading")
            journal = AdvancedTradingJournal.create_journal(
                df_test, st.session_state.get('bt_predictions', results[backtest_model]['predictions']),
                st.session_state.get('bt_config', {'stop_loss': 0.02, 'take_profit': 0.04})
            )
            
            if not journal.empty:
                col_j1, col_j2, col_j3, col_j4 = st.columns(4)
                col_j1.metric("Avg R", f"{journal['R Multiple'].mean():.2f}")
                col_j2.metric("Avg Duration", f"{journal['Duration'].mean():.1f}")
                col_j3.metric("Best", f"{journal['PnL (%)'].max():.2f}%")
                col_j4.metric("Worst", f"{journal['PnL (%)'].min():.2f}%")
                
                st.dataframe(journal, use_container_width=True, hide_index=True)
                csv_j = journal.to_csv(index=False)
                st.download_button("📥 TÉLÉCHARGER JOURNAL", data=csv_j,
                                  file_name=f"journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics":
    st.markdown("### 📊 ANALYTICS DASHBOARD")
    
    if 'bt_results' not in st.session_state:
        st.warning("⚠️ Lancez d'abord un backtest.")
    else:
        bt_results = st.session_state['bt_results']
        trades = bt_results['trades_detail']
        
        if not trades:
            st.warning("⚠️ Aucun trade.")
        else:
            st.markdown("#### 📈 Analyse des Trades")
            
            col1, col2 = st.columns(2)
            with col1:
                long_trades = [t for t in trades if t['type'] == 'LONG']
                short_trades = [t for t in trades if t['type'] == 'SHORT']
                
                fig_type = go.Figure(data=[go.Pie(
                    labels=['LONG', 'SHORT'],
                    values=[len(long_trades), len(short_trades)],
                    hole=0.4,
                    marker_colors=['#00ff88', '#ff4444']
                )])
                fig_type.update_layout(title='Répartition Long/Short', template='plotly_dark', height=300)
                st.plotly_chart(fig_type, use_container_width=True)
            
            with col2:
                exit_reasons = {}
                for t in trades:
                    reason = t['exit']
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
                fig_exit = go.Figure(data=[go.Pie(
                    labels=list(exit_reasons.keys()),
                    values=list(exit_reasons.values()),
                    hole=0.4
                )])
                fig_exit.update_layout(title='Raisons de Sortie', template='plotly_dark', height=300)
                st.plotly_chart(fig_exit, use_container_width=True)
            
            # Cumulative PnL
            pnls = [t['pnl'] * 100 for t in trades]
            cumulative_pnl = np.cumsum(pnls)
            
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(y=cumulative_pnl, mode='lines', fill='tozeroy',
                                        line=dict(color='#FFAA00'), fillcolor='rgba(255, 170, 0, 0.3)'))
            fig_cum.update_layout(title='PnL Cumulatif par Trade', template='plotly_dark', height=300,
                                 xaxis_title='Trade #', yaxis_title='PnL Cumulatif (%)')
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Stats par type
            st.markdown("#### 📋 Stats par Type de Trade")
            
            stats_data = []
            for trade_type, trades_list in [('LONG', long_trades), ('SHORT', short_trades)]:
                if trades_list:
                    wins = [t for t in trades_list if t['pnl'] > 0]
                    stats_data.append({
                        'Type': trade_type,
                        'Count': len(trades_list),
                        'Win Rate': f"{len(wins)/len(trades_list)*100:.1f}%",
                        'Avg PnL': f"{np.mean([t['pnl']*100 for t in trades_list]):.2f}%",
                        'Avg Duration': f"{np.mean([t['duration'] for t in trades_list]):.1f}"
                    })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            # Corrélation features
            if 'features' in st.session_state and 'df_ml' in st.session_state:
                with st.expander("📊 Corrélation Features/Target"):
                    features = st.session_state['features'][:15]
                    fig_corr = AdvancedVisualizer.create_feature_correlation(
                        st.session_state['df_ml'], features)
                    if fig_corr:
                        st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ROADMAP
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Roadmap":
    st.markdown("### 🗺️ ROADMAP & FONCTIONNALITÉS")
    
    st.markdown("#### ✅ Fonctionnalités Implémentées (v2.0)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔥 Priorité Haute:**
        - ✅ Optimisation hyperparamètres (RandomizedSearchCV)
        - ✅ Walk-Forward Analysis (validation robuste)
        - ✅ Gestion risque avancée:
          - Kelly Criterion
          - Trailing Stop Loss
          - Min Risk/Reward Ratio
        - ✅ Features techniques avancées:
          - ADX, Williams %R, CCI
          - Volatilité Parkinson
          - OBV, MFI
          - Patterns de bougies
          - Régime de marché
        """)
    
    with col2:
        st.markdown("""
        **⚡ Priorité Moyenne:**
        - ✅ Ensemble de modèles (voting pondéré)
        - ✅ Filtres de marché:
          - Trend filter
          - Volatility filter
          - ADX filter
        - ✅ Métriques supplémentaires:
          - Sharpe / Sortino Ratio
          - Profit Factor
          - Recovery Factor
          - Expectancy
        - ✅ Visualisations avancées:
          - Heatmap mensuelle
          - Distribution PnL
          - Corrélation features
        """)
    
    st.markdown("---")
    st.markdown("#### 💡 Nice to Have (À Venir)")
    
    st.markdown("""
    <div class="roadmap-item">
        <b>🔜 1. Sauvegarde des Modèles</b><br>
        Exporter/importer les modèles entraînés pour éviter de re-entraîner à chaque fois.
    </div>
    <div class="roadmap-item">
        <b>🔜 2. Alertes en Temps Réel</b><br>
        Notification Telegram/Discord quand le modèle génère un signal.
    </div>
    <div class="roadmap-item">
        <b>🔜 3. Multi-Timeframe Analysis</b><br>
        Combiner signaux de plusieurs timeframes (ex: trend 4h, entrée 15m).
    </div>
    <div class="roadmap-item">
        <b>🔜 4. Comparaison Multi-Crypto</b><br>
        Tester la même stratégie sur plusieurs assets et identifier où elle marche le mieux.
    </div>
    <div class="roadmap-item">
        <b>🔜 5. Données Externes</b><br>
        Fear & Greed Index, données on-chain, corrélation BTC/ETH.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💬 **Feedback?** Ces fonctionnalités seront ajoutées dans les prochaines versions!")


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace;'>
    © 2025 CRYPTO ML TRADING PLATFORM v2.0 | Supabase Connected
</div>
""", unsafe_allow_html=True)
