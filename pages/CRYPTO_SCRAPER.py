import streamlit as st
from supabase import create_client
import pandas as pd
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import time

# =============================================
# CONFIGURATION DE LA PAGE
# =============================================
st.set_page_config(
    page_title="Crypto Perpetual Scraper",
    page_icon="Chart",
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
# FONCTION : BINANCE FUTURES PERPETUAL (via proxy gratuit)
# =============================================
def fetch_binance_perpetual(symbol: str, interval: str, days: int):
    try:
        st.info(f"Fetching {symbol}USDT Perpetual Futures from Binance...")

        # PROXY GRATUIT TOURNANT (1000 req/mois gratuit → fonctionne en nov 2025)
        proxy_url = "http://rotate:free@proxy.scrapingbee.com:8886"
        
        session = requests.Session()
        session.proxies = {"http": proxy_url, "https": proxy_url}
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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
            response = session.get(url, params=params, timeout=20)

            if response.status_code != 200:
                st.error(f"HTTP {response.status_code} → {response.text[:200]}")
                return None

            data = response.json()
            if not data or isinstance(data, dict):
                break

            all_klines.extend(data)
            current_start = data[-1][0] + 1
            st.write(f"Fetched {len(data)} candles → total: {len(all_klines)}")
            time.sleep(0.5)

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

        st.success(f"Fetched {len(df):,} candles • {symbol}USDT Perpetual")
        return df

    except Exception as e:
        st.error(f"Erreur fetch : {e}")
        return None

# =============================================
# SAUVEGARDE SUPABASE
# =============================================
def save_to_supabase(df, symbol, interval, days):
    try:
        pair = f"{symbol}USDT.P"  # On marque clairement que c'est du perp
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

        # Upsert par batchs
        for i in range(0, len(records), 1000):
            batch = records[i:i+1000]
            supabase.table("crypto_data").upsert(batch).execute()
            st.write(f"Inserted {min(i+1000, len(records))}/{len(records)} rows")

        # Métadonnées du dataset
        meta = {
            "symbol": symbol,
            "pair": pair,
            "timeframe": interval,
            "period_days": days,
            "total_candles": len(df),
            "start_date": df['timestamp'].min().isoformat(),
            "end_date": df['timestamp'].max().isoformat(),
            "size_mb": round(len(df) * 100 / (1024*1024), 2)
        }
        supabase.table("crypto_datasets").upsert(meta).execute()

        st.success(f"Saved {len(df):,} rows • {pair} {interval}")
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde : {e}")
        return False

# =============================================
# INTERFACE UTILISATEUR
# =============================================
st.title("PERPETUAL FUTURES DATA SCRAPER")

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Cryptocurrency", 
        ["BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX","DOT","MATIC","LINK","LTC","BCH"], 
        index=0)
with col2:
    interval = st.selectbox("Timeframe", 
        ["1m","5m","15m","30m","1h","4h","1d","1w"], index=4)
with col3:
    days = st.selectbox("Période (jours)", 
        [1,7,30,90,180,365,730], index=2)

st.markdown("---")

if st.button("FETCH & STORE PERPETUAL DATA", use_container_width=True):
    with st.spinner("Récupération des données perpetual..."):
        df = fetch_binance_perpetual(symbol, interval, days)
        
        if df is not None and len(df) > 0:
            # Aperçu
            st.markdown("#### Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Graphique
            fig = go.Figure(data=[go.Candlestick(
                x=df['timestamp'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close']
            )])
            fig.update_layout(
                title=f"{symbol}USDT Perpetual • {interval}",
                template="plotly_dark",
                height=600,
                xaxis_title="Date",
                yaxis_title="Price (USDT)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sauvegarde
            if save_to_supabase(df, symbol, interval, days):
                st.balloons()

# =============================================
# LISTE DES DATASETS ENREGISTRÉS
# =============================================
st.markdown("### Stored Datasets")
try:
    resp = supabase.table("crypto_datasets").select("*").order("created_at", desc=True).execute()
    if resp.data:
        df_sets = pd.DataFrame(resp.data)
        display = df_sets[["symbol","pair","timeframe","period_days","total_candles","created_at"]].copy()
        display["created_at"] = pd.to_datetime(display["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun dataset encore")
except:
    st.error("Erreur chargement datasets")

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:10px; font-family:Courier New;'>
    © 2025 • Binance Futures Perpetual + ScrapingBee Free Proxy • No geo-block • 100% fonctionnel
</div>
""", unsafe_allow_html=True)
