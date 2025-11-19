import streamlit as st
from supabase import create_client
import pandas as pd
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import time
import os

# =============================================
# CONFIGURATION (à remplir une seule fois)
# =============================================
st.set_page_config(page_title="Crypto Perp Local Scraper", page_icon="Chart", layout="wide")

# === TES SECRETS SUPABASE (à remplir une fois) ===
# Va sur https://app.supabase.com → Project Settings → API
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://tonprojet.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "ta_cle_anonyme_ou_service_role")

# Si tu n'as pas de secrets (fichier .streamlit/secrets.toml), décommente et remplis :
# SUPABASE_URL = "https://xyz.supabase.co"
# SUPABASE_KEY = "eyJhbGciOi..."

# =============================================
# STYLE BLOOMBERG
# =============================================
st.markdown("""
<style>
    .main {background:#000;color:#FFAA00;padding:20px;}
    .stButton>button {background:#333;color:#FFAA00;border:2px solid #FFAA00;font-weight:bold;}
    .stButton>button:hover {background:#FFAA00;color:#000;}
    h1,h2,h3 {color:#FFAA00 !important;font-family:'Courier New',monospace !important;}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
<div style="background:#FFAA00;padding:15px;color:#000;font-weight:bold;font-size:20px;font-family:Courier New;">
    LOCAL PERPETUAL FUTURES SCRAPER • {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
</div>
''', unsafe_allow_html=True)

# =============================================
# CONNEXION SUPABASE
# =============================================
@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase()

# =============================================
# FONCTION : BINANCE FUTURES PERPETUAL (local = marche direct)
# =============================================
def fetch_perpetual(symbol: str, interval: str, days: int):
    try:
        st.info(f"Récupération {symbol}USDT Perpetual depuis Binance Futures...")

        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = end_ts - days * 24 * 60 * 60 * 1000

        interval_map = {
            "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
            "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
            "1d":"1d","3d":"3d","1w":"1w","1M":"1M"
        }

        symbol_perp = f"{symbol}USDT"
        url = "https://fapi.binance.com/fapi/v1/klines"
        all_klines = []
        current = start_ts

        with st.spinner(f"Téléchargement {days} jours en {interval}..."):
            progress_bar = st.progress(0)
            total_expected = int((end_ts - start_ts) / (1000 * 60 * int(interval_map[interval].replace("m","").replace("h","60").replace("d","1440").replace("w","10080").replace("M","43200"))))

            while current < end_ts:
                params = {
                    "symbol": symbol_perp,
                    "interval": interval_map[interval],
                    "startTime": current,
                    "endTime": end_ts,
                    "limit": 1000
                }
                r = requests.get(url, params=params, timeout=30)
                if r.status_code != 200:
                    st.error(f"Erreur HTTP {r.status_code}")
                    return None

                data = r.json()
                if not data:
                    break

                all_klines.extend(data)
                current = data[-1][0] + 1

                # Mise à jour barre
                fetched = len(all_klines)
                progress_bar.progress(min(fetched / max(total_expected, 1), 1.0))

                time.sleep(0.05)  # Très rapide localement

        if not all_klines:
            st.error("Aucune donnée")
            return None

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        for c in ['open','high','low','close','volume','quote_volume']:
            df[c] = df[c].astype(float)
        df['trades'] = df['trades'].astype(int)

        st.success(f"✅ {len(df):,} bougies récupérées • {symbol}USDT.P")
        return df

    except Exception as e:
        st.error(f"Erreur : {e}")
        return None

# =============================================
# SAUVEGARDE SUPABASE
# =============================================
def save_to_supabase(df, symbol, interval, days):
    try:
        pair = f"{symbol}USDT.P"
        st.info("Sauvegarde dans Supabase...")

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
            st.write(f"✓ {min(i+1000, len(records))}/{len(records)} lignes insérées")

        # Métadonnées
        meta = {
            "symbol": symbol,
            "pair": pair,
            "timeframe": interval,
            "period_days": days,
            "total_candles": len(df),
            "start_date": df['timestamp'].min().isoformat(),
            "end_date": df['timestamp'].max().isoformat(),
        }
        supabase.table("crypto_datasets").upsert(meta).execute()

        st.success(f"✅ Tout sauvegardé dans Supabase !")
        st.balloons()
    except Exception as e:
        st.error(f"Erreur sauvegarde : {e}")

# =============================================
# INTERFACE
# =============================================
st.title("LOCAL PERPETUAL FUTURES SCRAPER")

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Crypto", ["BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX","DOT","MATIC","LINK","PEPE","WIF"], index=0)
with col2:
    interval = st.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h","1d","1w"], index=4)
with col3:
    days = st.slider("Jours", 1, 1095, 90)  # Jusqu'à 3 ans

if st.button("LANCER LE SCRAPING & SAUVEGARDER DANS SUPABASE", use_container_width=True, type="primary"):
    df = fetch_perpetual(symbol, interval, days)
    if df is not None:
        st.dataframe(df.tail(10))
        fig = go.Figure(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.update_layout(title=f"{symbol}USDT.P • {interval} • {days} jours", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)
        save_to_supabase(df, symbol, interval, days)

# Liste des datasets déjà scrapés
st.markdown("### Datasets dans Supabase")
try:
    data = supabase.table("crypto_datasets").select("*").order("created_at", desc=True).execute()
    if data.data:
        df_disp = pd.DataFrame(data.data)[["symbol","pair","timeframe","period_days","total_candles","created_at"]]
        df_disp["created_at"] = pd.to_datetime(df_disp["created_at"]).dt.strftime("%d/%m/%Y %H:%M")
        st.dataframe(df_disp, use_container_width=True)
except:
    st.info("Connexion Supabase OK, aucun dataset encore")

st.markdown("---")
st.caption("Local → IP résidentielle → Aucun blocage • Binance Futures Perpetual • Sauvegarde directe Supabase")
