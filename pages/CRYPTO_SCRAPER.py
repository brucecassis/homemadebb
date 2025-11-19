import streamlit as st
from supabase import create_client
import pandas as pd
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
import time

# =============================================
# CONFIGURATION DE LA PAGE
# =============================================
st.set_page_config(
    page_title="Crypto Perpetual Scraper",
    page_icon="📊",
    layout="wide"
)

# =============================================
# STYLE BLOOMBERG (noir + orange)
# =============================================
st.markdown("""
<style>
    .main {background-color: #000000; color: #FFAA00; padding: 10px;}
    .stButton>button {
        background-color: #333; color: #FFAA00; font-weight: bold;
        border: 1px solid #FFAA00; border-radius: 0px; font-family: 'Courier New';
    }
    .stButton>button:hover {background-color: #FFAA00; color: #000;}
    h1, h2, h3 {color: #FFAA00 !important; font-family: 'Courier New', monospace !important;}
</style>
""", unsafe_allow_html=True)

# Header animé
st.markdown(f'''
<div style="background: #FFAA00; padding: 15px 25px; color: #000; font-weight: bold; font-size: 18px; font-family: 'Courier New', monospace; letter-spacing: 3px;">
    PERPETUAL FUTURES SCRAPER • {datetime.now().strftime("%H:%M:%S")} UTC
</div>
''', unsafe_allow_html=True)

# =============================================
# SUPABASE
# =============================================
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
supabase = get_supabase()

# =============================================
# FONCTION : BINANCE FUTURES PERPETUAL (public, sans clé, anti-403)
# =============================================
def fetch_binance_perpetual(symbol: str, interval: str, days: int):
    try:
        st.info(f"Fetching {symbol}USDT Perpetual Futures from Binance (public API)...")

        # OPTION PROXY (décommente si bloqué + ajoute tes creds IPRoyal ~5€/mois)
        # proxy_url = "http://username:password@brd.superproxy.io:22225"  # Ex: IPRoyal EU
        # session.proxies = {"http": proxy_url, "https": proxy_url}

        session = requests.Session()
        
        # Retry strategy pour 403/429
        retry_strategy = Retry(
            total=3, backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers anti-bot renforcés (simule navigateur réel)
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.binance.com/en/futures/BTCUSDT",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site"
        })

        # Calcul des timestamps
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = end_ts - days * 24 * 60 * 60 * 1000

        # Mapping intervalles Binance Futures
        interval_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
            "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
        }

        symbol_perp = f"{symbol}USDT"
        url = "https://fapi.binance.com/fapi/v1/klines"

        all_klines = []
        current_start = start_ts

        while current_start < end_ts:
            params = {
                "symbol": symbol_perp,
                "interval": interval_map[interval],
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000
            }
            response = session.get(url, params=params, timeout=30)

            if response.status_code != 200:
                st.error(f"HTTP {response.status_code}: {response.text[:200]}")
                if response.status_code == 403:
                    st.warning("🔒 403 = IP bloquée (datacenter ?). Teste localement ou ajoute un proxy IPRoyal.")
                    return fallback_coingecko(symbol, days)  # Fallback daily/hourly
                return None

            data = response.json()
            if not data or isinstance(data, dict):  # Erreur Binance
                st.warning("API error, fallback to CoinGecko...")
                return fallback_coingecko(symbol, days)

            all_klines.extend(data)
            current_start = data[-1][0] + 1
            st.write(f"Fetched {len(data)} candles → total: {len(all_klines)}")
            time.sleep(0.2)  # Rate limit safe (20 req/s)

        if not all_klines:
            st.error("Aucune donnée retournée")
            return None

        # DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        df['trades'] = df['trades'].astype(int)

        st.success(f"✅ Fetched {len(df):,} candles • {symbol}USDT Perpetual")
        return df

    except Exception as e:
        st.error(f"Erreur fetch : {e}")
        return fallback_coingecko(symbol, days)

# Fallback CoinGecko (pour daily/hourly si Binance bloque)
def fallback_coingecko(symbol, days):
    try:
        st.info("Fallback to CoinGecko (daily/hourly only)...")
        cg_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "polygon"}
        cg_id = cg_map.get(symbol, symbol.lower())
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(url, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = df["timestamp"]
        df["volume"] = df["quote_volume"] = df["trades"] = 0.0
        st.success(f"Fallback: {len(df)} candles from CoinGecko")
        return df
    except Exception as e:
        st.error(f"Fallback failed: {e}")
        return None

# =============================================
# SAUVEGARDE SUPABASE
# =============================================
def save_to_supabase(df, symbol, interval, days):
    try:
        pair = f"{symbol}USDT.P"  # Marque perp
        st.info("Saving to Supabase...")

        records = []
        for _, row in df.iterrows():
            records.append({
                "symbol": symbol,
                "pair": pair,
                "timeframe": interval,
                "timestamp": int(row['timestamp'].timestamp()),
                "open_time": row['timestamp'].isoformat(),
                "close_time": row['close_time'].isoformat(),
                "open_price": float(row['open']),
                "high_price": float(row['high']),
                "low_price": float(row['low']),
                "close_price": float(row['close']),
                "volume": float(row['volume']),
                "quote_volume": float(row['quote_volume']),
                "trades": int(row['trades'])
            })

        # Upsert par batchs de 1000
        batch_size = 1000
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table("crypto_data").upsert(batch).execute()
            total_inserted += len(batch)
            st.write(f"Inserted {total_inserted}/{len(records)} rows...")

        # Métadonnées dataset
        meta = {
            "symbol": symbol,
            "pair": pair,
            "timeframe": interval,
            "period_days": days,
            "total_candles": len(df),
            "start_date": df['timestamp'].min().isoformat(),
            "end_date": df['timestamp'].max().isoformat(),
            "size_mb": round(len(df) * 100 / (1024 * 1024), 2)
        }
        supabase.table("crypto_datasets").upsert(meta).execute()

        st.success(f"✅ Saved {len(df):,} rows • {pair} {interval}")
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde : {e}")
        return False

# =============================================
# INTERFACE UTILISATEUR
# =============================================
st.title("📊 PERPETUAL FUTURES DATA SCRAPER")

st.markdown("### ⚙️ PARAMÈTRES")
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Cryptocurrency", 
        ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"], 
        index=0)
with col2:
    interval = st.selectbox("Timeframe", 
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"], index=4)
with col3:
    days = st.selectbox("Période (jours)", 
        [1, 7, 30, 90, 180, 365], index=2)  # Limité pour éviter timeouts

st.markdown("---")

if st.button("📥 FETCH & STORE PERPETUAL DATA", use_container_width=True):
    with st.spinner("Récupération des données perpetual..."):
        df = fetch_binance_perpetual(symbol, interval, days)
        
        if df is not None and len(df) > 0:
            # Aperçu
            st.markdown("#### 📊 DATA PREVIEW")
            st.dataframe(df.head(10), use_container_width=True)

            # Graphique
            fig = go.Figure(data=[go.Candlestick(
                x=df['timestamp'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close']
            )])
            fig.update_layout(
                title=f"{symbol}USDT.P (Perpetual) • {interval}",
                template="plotly_dark",
                height=500,
                xaxis_title="Date",
                yaxis_title="Price (USDT)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sauvegarde
            if save_to_supabase(df, symbol, interval, days):
                st.balloons()
        else:
            st.error("❌ Échec fetch. Teste localement ou ajoute un proxy.")

st.markdown("---")

# =============================================
# LISTE DES DATASETS ENREGISTRÉS
# =============================================
st.markdown("### 📂 STORED DATASETS")
try:
    resp = supabase.table("crypto_datasets").select("*").order("created_at", desc=True).execute()
    if resp.data:
        df_sets = pd.DataFrame(resp.data)
        display = df_sets[["symbol", "pair", "timeframe", "period_days", "total_candles", "created_at"]].copy()
        display["created_at"] = pd.to_datetime(display["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display, use_container_width=True, hide_index=True)
        
        # Suppression
        st.markdown("#### 🔧 ACTIONS")
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            dataset_to_delete = st.selectbox(
                "Select to delete",
                options=[f"{d['symbol']}-{d['timeframe']}-{d['period_days']}d" for d in resp.data],
                key="delete_select"
            )
        with col_del2:
            if st.button("🗑️ DELETE", use_container_width=True):
                parts = dataset_to_delete.split('-')
                sym, tf, pd_str = parts[0], parts[1], int(parts[2].replace('d', ''))
                supabase.table("crypto_data").delete().eq('symbol', sym).eq('timeframe', tf).eq('period_days', pd_str).execute()
                supabase.table("crypto_datasets").delete().eq('symbol', sym).eq('timeframe', tf).eq('period_days', pd_str).execute()
                st.success(f"Deleted {dataset_to_delete}")
                st.rerun()
    else:
        st.info("📭 No datasets yet. Fetch one above!")
except Exception as e:
    st.error(f"Error loading datasets: {e}")

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:10px; font-family:Courier New;'>
    © 2025 • Binance Futures Public API • Anti-403 Headers • Fallback CoinGecko
</div>
""", unsafe_allow_html=True)
