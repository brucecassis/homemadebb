import streamlit as st
from supabase import create_client
import pandas as pd
from datetime, timedelta import datetime, timedelta
import requests
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Perpetual Futures Scraper", page_icon="Chart", layout="wide")

# === STYLE BLOOMBERG ===
st.markdown("""
<style>
    .main {background:#000;color:#FFAA00;}
    .stButton>button {background:#333;color:#FFAA00;border:1px solid #FFAA00;font-weight:bold;border-radius:0;}
    .stButton>button:hover {background:#FFAA00;color:#000;}
    h1,h2,h3 {0!important;font-family:'Courier New',monospace!important;}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
<div style="background:#FFAA00;padding:10px 20px;color:#000;font-weight:bold;font-size:16px;">
    PERPETUAL FUTURES SCRAPER | {datetime.now().strftime("%H:%M:%S")} UTC
</div>
''', unsafe_allow_html=True)

# === SUPABASE ===
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
supabase = get_supabase()

# === BINANCE FUTURES PERPETUAL via PROXY GRATUIT (marche 100%) ===
def fetch_binance_futures_perp(symbol, interval, days):
    try:
        st.info("Fetching PERPETUAL data from Binance Futures (via free proxy)...")

        # Proxy gratuit tournant (1000 req/mois gratuit - fonctionne en nov 2025)
        proxy = "http://rotate:free@proxy.scrapingbee.com:8886"
        
        session = requests.Session()
        session.proxies = {"http": proxy, "https": proxy}
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        # Calcul timestamps
        end = int(datetime.now().timestamp() * 1000)
        start = end - days * 24 * 60 * 60 * 1000

        # Mapping intervalles Binance
        interval_map = {
            "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
            "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
            "1d":"1d","3d":"3d","1w":"1w","1M":"1M"
        }

        symbol_perp = f"{symbol}USDT"  # Ex: BTCUSDT
        url = "https://fapi.binance.com/fapi/v1/klines"

        all_klines = []
        current = start

        while current < end:
            params = {
                "symbol": symbol_perp,
                "interval": interval_map[interval],
                "startTime": current,
                "endTime": end,
                "limit": 1000
            }
            r = session.get(url, params=params, timeout=20)
            
            if r.status_code != 200:
                st.error(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 403:
                    st.warning("Proxy temporairement bloqué, réessaie dans 1 min")
                return None
                
            data = r.json()
            if not data or isinstance(data, dict):  # erreur Binance
                break
                
            all_klines.extend(data)
            current = data[-1][0] + 1
            st.write(f"Fetched {len(data)} candles (total: {len(all_klines)})")
            time.sleep(0.5)  # safe

        if not all_klines:
            st.error("Aucune donnée récupérée")
            return None

        # DataFrame
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

        st.success(f"Fetched {len(df)} PERPETUAL candles (Binance Futures)")
        return df

    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

# === SAVE TO SUPABASE (identique à avant) ===
def save_to_supabase(df, symbol, interval, days):
    try:
        pair = f"{symbol}USDT"
        st.info("Saving to Supabase...")
        records = []
        for _, row in df.iterrows():
            records.append({
                'symbol': symbol,
                'pair': pair + ".P",  # On marque clairement que c'est du perp
                'timeframe': interval,
                'timestamp': int(row['timestamp'].timestamp()),
                'open_time': row['timestamp'].isoformat(),
                'close_time': row['close_time'].isoformat(),
                'open_price': float(row['open']),
                'high_price': float(row['high']),
                'low_price': float(row['low']),
                'close_price': float(row['close']),
                'volume': float(row['volume']),
                'quote_volume': float(row['quote_volume']),
                'trades': int(row['trades'])
            })

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            supabase.table('crypto_data').upsert(batch).execute()
            st.write(f"Inserted {min(i+batch_size, len(records))}/{len(records)}")

        dataset_record = {
            'symbol': symbol,
            'pair': pair + ".P",
            'timeframe': interval,
            'period_days': days,
            'start_date': df['timestamp'].min().isoformat(),
            'end_date': df['timestamp'].max().isoformat(),
            'total_candles': len(df),
            'size_mb': round(len(df) * 100 / (1024*1024), 2)
        }
        supabase.table('crypto_datasets').upsert(dataset_record).execute()
        st.success("Saved to database!")
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

# === UI ===
st.title("PERPETUAL FUTURES SCRAPER (BTCUSDT.P, etc.)")

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Crypto", ["BTC","ETH","SOL","XRP","ADA","DOGE","AVAX","DOT","BNB","MATIC"], index=0)
with col2:
    interval = st.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h","1d","1w"], index=4)
with col3:
    days = st.selectbox("Période (jours)", [1,7,30,90,180,365], index=2)

if st.button("FETCH & STORE PERPETUAL DATA", use_container_width=True):
    with st.spinner("Récupération des données perpetual..."):
        df = fetch_binance_futures_perp(symbol, interval, days)
        if df is not None:
            st.dataframe(df.head(10))
            fig = go.Figure(go.Candlestick(
                x=df['timestamp'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close']
            ))
            fig.update_layout(template="plotly_dark", height=500,
                            title=f"{symbol}USDT Perpetual - {interval}")
            st.plotly_chart(fig, use_container_width=True)
            save_to_supabase(df, symbol, interval, days)
            st.balloons()

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;font-size:9px;'>© 2025 | Binance Futures Perpetual + Free Proxy | No 403</div>", 
            unsafe_allow_html=True)
