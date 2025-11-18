import streamlit as st
from supabase import create_client, Client
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go

st.set_page_config(
    page_title="Crypto Perpetual Scraper",
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f'''
<div style="background: #FFAA00; padding: 10px 20px; color: #000; font-weight: bold; font-size: 16px; font-family: 'Courier New', monospace; letter-spacing: 2px; margin-bottom: 20px;">
    ⬛ CRYPTO PERPETUAL FUTURES SCRAPER | {datetime.now().strftime("%H:%M:%S")} UTC
</div>
''', unsafe_allow_html=True)

# Connexion Supabase
@st.cache_resource
def get_supabase_client():
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    return create_client(supabase_url, supabase_key)

supabase = get_supabase_client()

# Fonction pour récupérer les données Bybit PERPETUALS (linear = USDT perps)
def fetch_bybit_perp_data(symbol, interval, days):
    """Récupère les données OHLCV PERPETUAL depuis Bybit (category=linear)"""
    try:
        # Calculer les timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
       
        # Convertir en millisecondes
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
       
        # Mapping des intervalles Bybit (minutes ou D/W/M)
        interval_map = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "8h": "480", "12h": "720",
            "1d": "D", "3d": "D", "1w": "W", "1M": "M"
        }
       
        pair = f"{symbol}USDT"  # Format pour perps : BTCUSDT (sans .P)
       
        st.info(f"📥 Fetching PERPETUAL {pair} data from Bybit (linear)...")
       
        # Endpoint Bybit V5
        base_url = "https://api.bybit.com"
        all_klines = []
        current_start = start_ts
        limit = 1000  # Max par appel
        
        # Session avec retry et headers anti-bot (contre 403)
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers pour simuler un browser (évite détection bot + 403)
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.bybit.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        })
        
        while current_start < end_ts:
            params = {
                "category": "linear",  # ← CHANGEMENT CLÉ : pour perpetual USDT
                "symbol": pair,
                "interval": interval_map[interval],
                "start": current_start,
                "end": end_ts,
                "limit": limit
            }
           
            response = session.get(f"{base_url}/v5/market/kline", params=params, timeout=30)
            
            # Log pour debug (affiche le JSON d'erreur si 403)
            if response.status_code != 200:
                st.error(f"HTTP {response.status_code}: {response.text[:500]}")  # Premier 500 chars
                if response.status_code == 403:
                    st.warning("🔒 403 = IP bloquée par Bybit (datacenter ?). Essaie un proxy EU ou migre vers PythonAnywhere EU.")
                return None
            
            data = response.json()
           
            if data["retCode"] != 0:
                st.error(f"Bybit API error (retCode {data['retCode']}): {data.get('retMsg', 'Unknown')}")
                if "Request blocked" in data.get('retMsg', ''):
                    st.warning("🔒 IP bloquée par Bybit. Essaie un proxy ou migre vers un serveur EU.")
                return None
           
            klines = data["result"]["list"]
            if not klines:
                break
           
            all_klines.extend(klines)
            # Mise à jour du start pour la prochaine page
            oldest_ts = int(klines[-1][0])
            current_start = oldest_ts + 1  # +1 pour éviter doublons
           
            st.write(f"📄 Fetched {len(klines)} candles (total: {len(all_klines)})")
            time.sleep(1)  # Rate limit safe (120 req/min)
        
        if not all_klines:
            st.error("No data returned from Bybit")
            return None
       
        # Inverser pour ordre chronologique croissant (Bybit renvoie descendant)
        all_klines.reverse()
       
        # Convertir en DataFrame (même structure que spot)
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'turnover'  # Équivalent quote_volume pour perps
        ])
       
        # Ajouter close_time (dupliqué pour compatibilité)
        df['close_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
       
        # Convertir les types (tous strings chez Bybit)
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)
       
        # Ajouter trades (non disponible en klines perps, set to 0)
        df['trades'] = 0
       
        st.success(f"✅ Fetched {len(df)} PERPETUAL candles")
       
        return df
       
    except Exception as e:
        st.error(f"Error fetching PERPETUAL data: {e}")
        return None

# Fonction pour sauvegarder dans Supabase (inchangée, compatible perps)
def save_to_supabase(df, symbol, interval, days):
    """Sauvegarde les données dans Supabase"""
    try:
        pair = f"{symbol}USDT"  # Format perp
       
        st.info("💾 Saving PERPETUAL data to database...")
       
        # Préparer les données
        records = []
        for _, row in df.iterrows():
            records.append({
                'symbol': symbol,
                'pair': pair,
                'timeframe': interval,
                'timestamp': int(row['timestamp'].timestamp()),
                'open_time': row['timestamp'].isoformat(),
                'close_time': row['close_time'].isoformat(),
                'open_price': float(row['open']),
                'high_price': float(row['high']),
                'low_price': float(row['low']),
                'close_price': float(row['close']),
                'volume': float(row['volume']),
                'quote_volume': float(row['turnover']),  # turnover = quote_volume pour perps
                'trades': int(row['trades'])
            })
       
        # Insérer par batch de 1000
        batch_size = 1000
        total_inserted = 0
       
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            response = supabase.table('crypto_data').upsert(batch).execute()
            total_inserted += len(batch)
            st.write(f" Inserted {total_inserted}/{len(records)} records...")
       
        # Calculer la taille approximative en MB
        size_mb = len(df) * 100 / (1024 * 1024) # Approximation
       
        # Enregistrer le dataset (ajout 'perp' pour distinction)
        dataset_record = {
            'symbol': symbol,
            'pair': pair,
            'timeframe': interval,
            'period_days': days,
            'start_date': df['timestamp'].min().isoformat(),
            'end_date': df['timestamp'].max().isoformat(),
            'total_candles': len(df),
            'size_mb': round(size_mb, 2)
        }
       
        supabase.table('crypto_datasets').upsert(dataset_record).execute()
       
        st.success(f"✅ Saved {len(df)} PERPETUAL records to database!")
        return True
       
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False

# ===== INTERFACE =====
st.title("📊 CRYPTO PERPETUAL FUTURES SCRAPER")

# Section 1: Paramètres de scraping
st.markdown("### ⚙️ PERPETUAL SCRAPING PARAMETERS")
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox(
        "Cryptocurrency",
        options=["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "MATIC", "DOT", "AVAX"],
        index=0
    )
with col2:
    interval = st.selectbox(
        "Timeframe",
        options=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        index=4 # Default: 1h
    )
with col3:
    days = st.selectbox(
        "Period (days)",
        options=[1, 7, 30, 90, 180, 365, 730, 1095, 1825, 3650],
        index=3 # Default: 90 days
    )
st.markdown("---")

# Section 2: Actions
col_action1, col_action2, col_action3 = st.columns([2, 1, 1])
with col_action1:
    if st.button("📥 FETCH & STORE PERPETUAL DATA", use_container_width=True):
        with st.spinner("Fetching PERPETUAL data from Bybit..."):
            df = fetch_bybit_perp_data(symbol, interval, days)
           
            if df is not None:
                # Afficher un aperçu
                st.markdown("#### 📊 PERPETUAL DATA PREVIEW")
                st.dataframe(df.head(10))
               
                # Graphique rapide
                fig = go.Figure(data=[go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )])
               
                fig.update_layout(
                    title=f"{symbol}USDT.P (Perpetual) - {interval}",
                    xaxis_title="Date",
                    yaxis_title="Price (USDT)",
                    height=400,
                    template="plotly_dark"
                )
               
                st.plotly_chart(fig, use_container_width=True)
               
                # Sauvegarder
                if save_to_supabase(df, symbol, interval, days):
                    st.balloons()

st.markdown("---")

# Section 3: Datasets existants (inchangée)
st.markdown("### 📂 STORED PERPETUAL DATASETS")
try:
    response = supabase.table('crypto_datasets').select("*").order('created_at', desc=True).execute()
   
    if response.data:
        datasets_df = pd.DataFrame(response.data)
       
        # Formater l'affichage
        display_df = datasets_df[[
            'symbol', 'timeframe', 'period_days', 'total_candles',
            'size_mb', 'start_date', 'end_date', 'created_at'
        ]].copy()
       
        display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.strftime('%Y-%m-%d')
        display_df['end_date'] = pd.to_datetime(display_df['end_date']).dt.strftime('%Y-%m-%d')
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
       
        st.dataframe(display_df, use_container_width=True, hide_index=True)
       
        # Actions sur les datasets (suppression)
        st.markdown("#### 🔧 DATASET ACTIONS")
       
        col_del1, col_del2 = st.columns([3, 1])
       
        with col_del1:
            dataset_to_delete = st.selectbox(
                "Select dataset to delete",
                options=[f"{d['symbol']}-{d['timeframe']}-{d['period_days']}d" for d in response.data],
                key="dataset_select"
            )
       
        with col_del2:
            if st.button("🗑️ DELETE", use_container_width=True):
                # Parser la sélection
                parts = dataset_to_delete.split('-')
                sym = parts[0]
                tf = parts[1]
                pd_str = parts[2].replace('d', '')
               
                try:
                    # Supprimer de crypto_data
                    supabase.table('crypto_data').delete().eq('symbol', sym).eq('timeframe', tf).execute()
                   
                    # Supprimer de crypto_datasets
                    supabase.table('crypto_datasets').delete().eq('symbol', sym).eq('timeframe', tf).eq('period_days', int(pd_str)).execute()
                   
                    st.success(f"✅ Deleted {dataset_to_delete}")
                    st.rerun()
                   
                except Exception as e:
                    st.error(f"Error deleting: {e}")
   
    else:
        st.info("📭 No PERPETUAL datasets stored yet. Create your first one above!")
       
except Exception as e:
    st.error(f"Error loading datasets: {e}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace;'>
    © 2025 CRYPTO PERPETUAL SCRAPER | Data from Bybit Linear API | Stored in Supabase
</div>
""", unsafe_allow_html=True)
