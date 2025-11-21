"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ML TRADING STRATEGY MODULE                               ║
║                    Pour intégration Streamlit                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Ce module contient:
1. Feature Engineering (indicateurs techniques)
2. Plusieurs modèles ML (Random Forest, XGBoost, LSTM, etc.)
3. Backtesting engine
4. Métriques de performance (Sharpe, Sortino, Max Drawdown, etc.)
5. Visualisations

À intégrer dans ton app Streamlit existante.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb

# Pour LSTM (optionnel - nécessite tensorflow)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING - Indicateurs Techniques
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Génère les features techniques pour le ML"""
    
    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute tous les indicateurs techniques"""
        df = df.copy()
        
        # S'assurer que les colonnes sont numériques
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ═══ TREND INDICATORS ═══
        # Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_55'] = df['close'].ewm(span=55, adjust=False).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ═══ MOMENTUM INDICATORS ═══
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        # ROC (Rate of Change)
        df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # ═══ VOLATILITY INDICATORS ═══
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Volatility (Standard Deviation)
        df['volatility_10'] = df['close'].rolling(window=10).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # ═══ VOLUME INDICATORS ═══
        # Volume Moving Average
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VWAP approximation
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # ═══ PRICE ACTION FEATURES ═══
        # Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Log Returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low Range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Close Position in Range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # ═══ TREND STRENGTH ═══
        # ADX (Average Directional Index) - Simplified
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        atr_14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.abs().rolling(window=14).mean() / atr_14)
        
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=14).mean()
        
        # ═══ PATTERN RECOGNITION (Simplified) ═══
        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # Consecutive candles
        df['consecutive_bullish'] = df['is_bullish'].rolling(window=3).sum()
        df['consecutive_bearish'] = 3 - df['consecutive_bullish']
        
        # ═══ CROSS SIGNALS ═══
        # MA Crosses
        df['ema_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['sma_cross_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # Price vs MA
        df['price_above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
        df['price_above_sma_200'] = (df['close'] > df['sma_200']).astype(int)
        
        # RSI Zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    @staticmethod
    def create_target(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
        """
        Crée la variable cible (target) pour la classification
        
        Args:
            horizon: Nombre de périodes dans le futur pour la prédiction
            threshold: Seuil de rendement pour considérer un mouvement significatif
        
        Returns:
            1 = Long (prix monte), 0 = Short/Neutre (prix baisse/stagne)
        """
        df = df.copy()
        
        # Rendement futur
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Target binaire
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Target multi-classe (optionnel)
        df['target_3class'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, -threshold, threshold, np.inf],
            labels=[0, 1, 2]  # 0=Short, 1=Neutre, 2=Long
        )
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODÈLES ML
# ═══════════════════════════════════════════════════════════════════════════════

class MLModels:
    """Collection de modèles ML pour le trading"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
    
    def get_available_models(self) -> dict:
        """Retourne les modèles disponibles"""
        models = {
            'random_forest': {
                'name': 'Random Forest',
                'description': 'Ensemble de arbres de décision. Bon pour capturer des patterns non-linéaires.',
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            },
            'xgboost': {
                'name': 'XGBoost',
                'description': 'Gradient Boosting optimisé. Excellent pour les données tabulaires.',
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'description': 'Boosting classique. Robuste et interprétable.',
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'description': 'Modèle linéaire simple. Baseline rapide.',
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            },
            'svm': {
                'name': 'Support Vector Machine',
                'description': 'SVM avec kernel RBF. Bon pour les frontières complexes.',
                'model': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
            }
        }
        
        if LSTM_AVAILABLE:
            models['lstm'] = {
                'name': 'LSTM Neural Network',
                'description': 'Réseau de neurones récurrent. Capture les dépendances temporelles.',
                'model': 'lstm'  # Géré séparément
            }
        
        return models
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'target',
                     test_size: float = 0.2, scale: bool = True):
        """Prépare les données pour l'entraînement"""
        
        # Supprimer les NaN
        df_clean = df.dropna(subset=feature_cols + [target_col])
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        # Split temporel (pas de shuffle pour les séries temporelles!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scaling
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scalers['standard'] = scaler
        
        return X_train, X_test, y_train, y_test, df_clean.iloc[split_idx:].index
    
    def train_model(self, model_key: str, X_train, y_train, X_test, y_test):
        """Entraîne un modèle et retourne les métriques"""
        
        models = self.get_available_models()
        
        if model_key not in models:
            raise ValueError(f"Modèle inconnu: {model_key}")
        
        if model_key == 'lstm':
            return self._train_lstm(X_train, y_train, X_test, y_test)
        
        model = models[model_key]['model']
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
        
        # Métriques
        results = {
            'model_name': models[model_key]['name'],
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'predictions': y_pred_test,
            'probabilities': y_prob_test,
            'model': model
        }
        
        # Feature importance si disponible
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_[0])
        
        self.models[model_key] = model
        self.results[model_key] = results
        
        return results
    
    def _train_lstm(self, X_train, y_train, X_test, y_test, lookback: int = 10):
        """Entraîne un modèle LSTM"""
        
        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow non installé")
        
        # Reshape pour LSTM [samples, timesteps, features]
        def create_sequences(X, y, lookback):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i-lookback:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)
        
        # Modèle LSTM
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Entraînement
        model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, 
                  validation_split=0.1, verbose=0)
        
        # Prédictions
        y_prob_test = model.predict(X_test_seq).flatten()
        y_pred_test = (y_prob_test > 0.5).astype(int)
        
        results = {
            'model_name': 'LSTM Neural Network',
            'train_accuracy': model.evaluate(X_train_seq, y_train_seq, verbose=0)[1],
            'test_accuracy': accuracy_score(y_test_seq, y_pred_test),
            'precision': precision_score(y_test_seq, y_pred_test, zero_division=0),
            'recall': recall_score(y_test_seq, y_pred_test, zero_division=0),
            'f1': f1_score(y_test_seq, y_pred_test, zero_division=0),
            'predictions': y_pred_test,
            'probabilities': y_prob_test,
            'model': model
        }
        
        self.models['lstm'] = model
        self.results['lstm'] = results
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Moteur de backtesting pour évaluer les stratégies"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% par trade
        self.slippage = slippage  # 0.05% de slippage
    
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray, 
                     probabilities: np.ndarray = None, threshold: float = 0.5,
                     position_size: float = 1.0, use_stop_loss: bool = True,
                     stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04) -> dict:
        """
        Exécute le backtest
        
        Args:
            df: DataFrame avec les prix
            predictions: Prédictions du modèle (0 ou 1)
            probabilities: Probabilités des prédictions
            threshold: Seuil de probabilité pour entrer en position
            position_size: Taille de position (1.0 = 100% du capital)
            use_stop_loss: Utiliser stop loss / take profit
            stop_loss_pct: Pourcentage de stop loss
            take_profit_pct: Pourcentage de take profit
        """
        
        df = df.copy()
        n = len(predictions)
        
        # Aligner les données
        df = df.iloc[-n:].copy()
        df['prediction'] = predictions
        df['probability'] = probabilities if probabilities is not None else predictions
        
        # Initialisation
        capital = self.initial_capital
        position = 0  # 0 = pas de position, 1 = long
        entry_price = 0
        
        # Tracking
        trades = []
        equity_curve = [capital]
        positions = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            pred = row['prediction']
            prob = row['probability']
            
            # Gestion des positions existantes
            if position == 1:
                # Check stop loss / take profit
                pnl_pct = (current_price - entry_price) / entry_price
                
                if use_stop_loss:
                    if pnl_pct <= -stop_loss_pct:
                        # Stop Loss touché
                        exit_price = entry_price * (1 - stop_loss_pct)
                        pnl = (exit_price - entry_price) / entry_price
                        pnl_after_costs = pnl - self.commission - self.slippage
                        capital *= (1 + pnl_after_costs * position_size)
                        trades.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pct': pnl_after_costs,
                            'exit_reason': 'stop_loss'
                        })
                        position = 0
                        
                    elif pnl_pct >= take_profit_pct:
                        # Take Profit touché
                        exit_price = entry_price * (1 + take_profit_pct)
                        pnl = (exit_price - entry_price) / entry_price
                        pnl_after_costs = pnl - self.commission - self.slippage
                        capital *= (1 + pnl_after_costs * position_size)
                        trades.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pct': pnl_after_costs,
                            'exit_reason': 'take_profit'
                        })
                        position = 0
                
                # Signal de sortie (prédiction = 0)
                elif pred == 0 and position == 1:
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    pnl_after_costs = pnl - self.commission
                    capital *= (1 + pnl_after_costs * position_size)
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_after_costs,
                        'exit_reason': 'signal'
                    })
                    position = 0
            
            # Signal d'entrée
            elif position == 0 and pred == 1 and prob >= threshold:
                position = 1
                entry_price = current_price * (1 + self.slippage)
                entry_idx = i
                capital *= (1 - self.commission)  # Commission d'entrée
            
            positions.append(position)
            equity_curve.append(capital)
        
        # Fermer position finale si ouverte
        if position == 1:
            exit_price = df.iloc[-1]['close'] * (1 - self.slippage)
            pnl = (exit_price - entry_price) / entry_price
            pnl_after_costs = pnl - self.commission
            capital *= (1 + pnl_after_costs * position_size)
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': len(df) - 1,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_after_costs,
                'exit_reason': 'end'
            })
        
        # Calculer les métriques
        equity_curve = np.array(equity_curve[1:])  # Enlever le premier (initial)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Buy & Hold pour comparaison
        bh_returns = df['close'].pct_change().fillna(0).values
        bh_equity = self.initial_capital * (1 + bh_returns).cumprod()
        
        metrics = self._calculate_metrics(equity_curve, returns, trades, bh_equity)
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'bh_equity': bh_equity,
            'positions': positions,
            'df': df
        }
    
    def _calculate_metrics(self, equity_curve: np.ndarray, returns: np.ndarray,
                           trades: list, bh_equity: np.ndarray) -> dict:
        """Calcule les métriques de performance"""
        
        # Rendement total
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        bh_return = (bh_equity[-1] - self.initial_capital) / self.initial_capital
        
        # Rendements annualisés (supposant 252 jours de trading)
        n_periods = len(equity_curve)
        annual_factor = 252 / n_periods if n_periods > 0 else 1
        annual_return = (1 + total_return) ** annual_factor - 1
        
        # Volatilité
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Sharpe Ratio (risk-free rate = 0 pour simplifier)
        sharpe = (annual_return / volatility) if volatility > 0 else 0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino = (annual_return / downside_vol) if downside_vol > 0 else 0
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Calmar Ratio
        calmar = (annual_return / max_drawdown) if max_drawdown > 0 else 0
        
        # Statistiques des trades
        if trades:
            trade_pnls = [t['pnl_pct'] for t in trades]
            winning_trades = [p for p in trade_pnls if p > 0]
            losing_trades = [p for p in trade_pnls if p <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        else:
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'bh_return_pct': bh_return * 100,
            'alpha': (total_return - bh_return) * 100,  # Surperformance vs B&H
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'expectancy_pct': expectancy * 100,
            'final_capital': equity_curve[-1] if len(equity_curve) > 0 else self.initial_capital
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TradingVisualizer:
    """Visualisations pour l'analyse de stratégie"""
    
    @staticmethod
    def plot_equity_curve(backtest_results: dict, title: str = "Equity Curve") -> go.Figure:
        """Graphique de la courbe d'équité vs Buy & Hold"""
        
        equity = backtest_results['equity_curve']
        bh_equity = backtest_results['bh_equity']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=equity,
            mode='lines',
            name='Strategy',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=bh_equity,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#ff8800', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Période",
            yaxis_title="Capital ($)",
            template="plotly_dark",
            height=400,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    @staticmethod
    def plot_drawdown(backtest_results: dict) -> go.Figure:
        """Graphique du drawdown"""
        
        equity = backtest_results['equity_curve']
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=-drawdown,
            fill='tozeroy',
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4444', width=1),
            fillcolor='rgba(255, 68, 68, 0.3)'
        ))
        
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Période",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    @staticmethod
    def plot_trades_on_price(backtest_results: dict) -> go.Figure:
        """Graphique des prix avec les trades"""
        
        df = backtest_results['df']
        trades = backtest_results['trades']
        
        fig = go.Figure()
        
        # Prix
        fig.add_trace(go.Candlestick(
            x=df.index if 'open_time' not in df.columns else df['open_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Entrées et sorties
        for trade in trades:
            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            
            # Point d'entrée (vert)
            fig.add_trace(go.Scatter(
                x=[entry_idx],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Entry',
                showlegend=False
            ))
            
            # Point de sortie (rouge ou vert selon PnL)
            color = 'green' if trade['pnl_pct'] > 0 else 'red'
            fig.add_trace(go.Scatter(
                x=[exit_idx],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(color=color, size=10, symbol='triangle-down'),
                name='Exit',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Trades sur le graphique",
            xaxis_title="Période",
            yaxis_title="Prix",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_names: list, importances: np.ndarray, top_n: int = 20) -> go.Figure:
        """Graphique de l'importance des features"""
        
        # Trier par importance
        indices = np.argsort(importances)[-top_n:]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker_color='#FFAA00'
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Features les plus importantes",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_dark",
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_model_comparison(results: dict) -> go.Figure:
        """Compare les métriques de différents modèles"""
        
        models = list(results.keys())
        metrics = ['test_accuracy', 'precision', 'recall', 'f1']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[m].get(metric, 0) for m in models]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=[results[m]['model_name'] for m in models],
                y=values
            ))
        
        fig.update_layout(
            title="Comparaison des modèles ML",
            xaxis_title="Modèle",
            yaxis_title="Score",
            barmode='group',
            template="plotly_dark",
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns(backtest_results: dict) -> go.Figure:
        """Heatmap des rendements mensuels"""
        
        equity = backtest_results['equity_curve']
        returns = np.diff(equity) / equity[:-1]
        
        # Simuler des mois (à adapter avec les vraies dates)
        n_months = min(24, len(returns) // 30)
        monthly_returns = []
        
        for i in range(n_months):
            start = i * 30
            end = (i + 1) * 30
            monthly_ret = np.sum(returns[start:end]) * 100
            monthly_returns.append(monthly_ret)
        
        # Reshape en grille (ex: 2 ans x 12 mois)
        n_years = max(1, n_months // 12)
        
        fig = go.Figure(data=go.Heatmap(
            z=[monthly_returns[i*12:(i+1)*12] for i in range(n_years)],
            x=[f'M{i+1}' for i in range(12)],
            y=[f'Année {i+1}' for i in range(n_years)],
            colorscale='RdYlGn',
            zmid=0
        ))
        
        fig.update_layout(
            title="Rendements par période",
            template="plotly_dark",
            height=300
        )
        
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

def render_ml_strategy_page(df: pd.DataFrame, table_name: str):
    """
    Render la page de stratégie ML dans Streamlit
    
    Args:
        df: DataFrame avec les données OHLCV
        table_name: Nom de la table sélectionnée
    """
    
    st.markdown("---")
    st.markdown("## 🤖 ML TRADING STRATEGY LAB")
    st.markdown(f"**Dataset:** `{table_name}` | **{len(df):,}** candles")
    
    # ═══ ÉTAPE 1: Configuration ═══
    st.markdown("### ⚙️ Configuration de la stratégie")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_horizon = st.selectbox(
            "Horizon de prédiction",
            options=[1, 2, 3, 5, 10, 20],
            index=0,
            help="Nombre de périodes dans le futur à prédire"
        )
    
    with col2:
        threshold = st.slider(
            "Seuil de rendement (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Rendement minimum pour considérer un signal positif"
        ) / 100
    
    with col3:
        test_size = st.slider(
            "Taille du test set (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Pourcentage des données pour le test"
        ) / 100
    
    # ═══ ÉTAPE 2: Feature Engineering ═══
    st.markdown("### 🔧 Feature Engineering")
    
    with st.spinner("Génération des indicateurs techniques..."):
        fe = FeatureEngineer()
        df_features = fe.add_all_features(df)
        df_features = fe.create_target(df_features, horizon=prediction_horizon, threshold=threshold)
    
    # Sélection des features
    all_features = [col for col in df_features.columns if col not in 
                    ['open_time', 'close_time', 'target', 'target_3class', 'future_return', 
                     'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
                     'taker_buy_base', 'taker_buy_quote', 'id', 'symbol', 'timeframe', 'created_at']]
    
    st.info(f"📊 **{len(all_features)}** features techniques générées")
    
    with st.expander("Voir les features disponibles"):
        feature_cols = st.columns(4)
        for i, feat in enumerate(all_features):
            with feature_cols[i % 4]:
                st.write(f"• {feat}")
    
    # ═══ ÉTAPE 3: Sélection et entraînement des modèles ═══
    st.markdown("### 🧠 Modèles ML")
    
    ml = MLModels()
    available_models = ml.get_available_models()
    
    selected_models = st.multiselect(
        "Sélectionner les modèles à entraîner",
        options=list(available_models.keys()),
        default=['random_forest', 'xgboost'],
        format_func=lambda x: f"{available_models[x]['name']} - {available_models[x]['description'][:50]}..."
    )
    
    if st.button("🚀 LANCER L'ENTRAÎNEMENT", type="primary", use_container_width=True):
        
        # Préparer les données
        with st.spinner("Préparation des données..."):
            X_train, X_test, y_train, y_test, test_indices = ml.prepare_data(
                df_features, 
                all_features, 
                'target',
                test_size=test_size
            )
        
        st.success(f"✅ Données préparées: {len(X_train):,} train / {len(X_test):,} test")
        
        # Entraîner les modèles
        results = {}
        progress_bar = st.progress(0)
        
        for i, model_key in enumerate(selected_models):
            with st.spinner(f"Entraînement de {available_models[model_key]['name']}..."):
                try:
                    results[model_key] = ml.train_model(model_key, X_train, y_train, X_test, y_test)
                    st.success(f"✅ {available_models[model_key]['name']}: Accuracy = {results[model_key]['test_accuracy']:.2%}")
                except Exception as e:
                    st.error(f"❌ {model_key}: {e}")
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        if results:
            # Stocker dans session_state
            st.session_state['ml_results'] = results
            st.session_state['ml_features'] = all_features
            st.session_state['df_features'] = df_features
            st.session_state['test_indices'] = test_indices
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
    
    # ═══ ÉTAPE 4: Résultats et Comparaison ═══
    if 'ml_results' in st.session_state and st.session_state['ml_results']:
        results = st.session_state['ml_results']
        
        st.markdown("### 📊 Résultats des modèles")
        
        # Tableau comparatif
        comparison_data = []
        for model_key, res in results.items():
            comparison_data.append({
                'Modèle': res['model_name'],
                'Accuracy Train': f"{res['train_accuracy']:.2%}",
                'Accuracy Test': f"{res['test_accuracy']:.2%}",
                'Precision': f"{res['precision']:.2%}",
                'Recall': f"{res['recall']:.2%}",
                'F1 Score': f"{res['f1']:.2%}"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        # Graphique de comparaison
        viz = TradingVisualizer()
        st.plotly_chart(viz.plot_model_comparison(results), use_container_width=True)
        
        # Feature importance du meilleur modèle
        best_model_key = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        
        if 'feature_importance' in results[best_model_key]:
            st.markdown(f"### 📈 Feature Importance ({results[best_model_key]['model_name']})")
            fig_importance = viz.plot_feature_importance(
                st.session_state['ml_features'],
                results[best_model_key]['feature_importance']
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # ═══ ÉTAPE 5: Backtesting ═══
        st.markdown("### 💰 Backtesting")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            backtest_model = st.selectbox(
                "Modèle pour backtest",
                options=list(results.keys()),
                format_func=lambda x: results[x]['model_name']
            )
        
        with col2:
            initial_capital = st.number_input(
                "Capital initial ($)",
                min_value=1000,
                max_value=1000000,
                value=10000
            )
        
        with col3:
            stop_loss = st.slider("Stop Loss (%)", 1.0, 10.0, 2.0) / 100
        
        with col4:
            take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 4.0) / 100
        
        if st.button("📈 LANCER LE BACKTEST", type="secondary", use_container_width=True):
            
            with st.spinner("Exécution du backtest..."):
                engine = BacktestEngine(initial_capital=initial_capital)
                
                # Récupérer les prédictions du modèle sélectionné
                predictions = results[backtest_model]['predictions']
                probabilities = results[backtest_model]['probabilities']
                
                # Aligner avec le DataFrame
                df_test = st.session_state['df_features'].iloc[-len(predictions):].copy()
                
                backtest_results = engine.run_backtest(
                    df_test,
                    predictions,
                    probabilities,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit
                )
            
            # Afficher les métriques
            metrics = backtest_results['metrics']
            
            st.markdown("#### 📊 Métriques de Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rendement Total", f"{metrics['total_return_pct']:.2f}%",
                         delta=f"vs B&H: {metrics['alpha']:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
            with col2:
                st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            
            with col3:
                st.metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")
                st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            
            with col4:
                st.metric("Total Trades", metrics['total_trades'])
                st.metric("Capital Final", f"${metrics['final_capital']:,.2f}")
            
            # Graphiques
            st.markdown("#### 📈 Equity Curve")
            st.plotly_chart(viz.plot_equity_curve(backtest_results), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📉 Drawdown")
                st.plotly_chart(viz.plot_drawdown(backtest_results), use_container_width=True)
            
            with col2:
                st.markdown("#### 📅 Rendements par période")
                st.plotly_chart(viz.plot_monthly_returns(backtest_results), use_container_width=True)
            
            # Détail des trades
            st.markdown("#### 📋 Détail des trades")
            if backtest_results['trades']:
                trades_df = pd.DataFrame(backtest_results['trades'])
                trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun trade exécuté")
            
            # Sauvegarder les résultats
            st.session_state['backtest_results'] = backtest_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN - Pour test standalone
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    st.set_page_config(page_title="ML Trading Lab", layout="wide")
    st.title("🤖 ML Trading Strategy Lab")
    st.info("Ce module doit être intégré dans ton app Streamlit principale.")
