import streamlit as st
from supabase import create_client, Client
import pandas as pd
from datetime import datetime, timedelta
import time
from binance.client import Client as BinanceClient
import plotly.graph_objects as go

st.set_page_config(
    page_title="Crypto Scraper",
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
    ⬛ CRYPTO DATA SCRAPER | {datetime.now().strftime("%H:%M:%S")} UTC
</div>
''', unsafe_allow_html=True)

# Connexion Supabase
@st.cache_resource
def get_supabase_client():
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    return create_client(supabase_url, supabase_key)

supabase = get_supabase_client()

# Connexion Binance (API publique, pas besoin de clés)
@st.cache_resource
def get_binance_client():
    return BinanceClient("", "")  # API publique

binance_client = get_binance_client()

# Fonction pour récupérer les données Binance
def fetch_binance_data(symbol, interval, days):
    """Récupère les données OHLCV depuis Binance"""
    try:
        # Calculer les timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Convertir en millisecondes
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Mapping des intervalles
        interval_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
            "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
        }
        
        pair = f"{symbol}USDT"
        
        st.info(f"📥 Fetching {symbol} data from Binance...")
        
        # Récupérer les klines (chandelier data)
        klines = binance_client.get_historical_klines(
            pair, 
            interval_map[interval],
            start_ts,
            end_ts
        )
        
        if not klines:
            st.error("No data returned from Binance")
            return None
        
        # Convertir en DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convertir les types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        df['trades'] = df['trades'].astype(int)
        
        st.success(f"✅ Fetched {len(df)} candles")
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fonction pour sauvegarder dans Supabase
def save_to_supabase(df, symbol, interval, days):
    """Sauvegarde les données dans Supabase"""
    try:
        pair = f"{symbol}USDT"
        
        st.info("💾 Saving to database...")
        
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
                'quote_volume': float(row['quote_volume']),
                'trades': int(row['trades'])
            })
        
        # Insérer par batch de 1000 (limite Supabase)
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            response = supabase.table('crypto_data').upsert(batch).execute()
            total_inserted += len(batch)
            st.write(f"  Inserted {total_inserted}/{len(records)} records...")
        
        # Calculer la taille approximative en MB
        size_mb = len(df) * 100 / (1024 * 1024)  # Approximation
        
        # Enregistrer le dataset
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
        
        st.success(f"✅ Saved {len(df)} records to database!")
        return True
        
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False

# ===== INTERFACE =====

st.title("📊 CRYPTO DATA SCRAPER")

# Section 1: Paramètres de scraping
st.markdown("### ⚙️ SCRAPING PARAMETERS")

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
        index=4  # Default: 1h
    )

with col3:
    days = st.selectbox(
        "Period (days)",
        options=[1, 7, 30, 90, 180, 365, 730, 1095, 1825, 3650],
        index=3  # Default: 90 days
    )

st.markdown("---")

# Section 2: Actions
col_action1, col_action2, col_action3 = st.columns([2, 1, 1])

with col_action1:
    if st.button("📥 FETCH & STORE DATA", use_container_width=True):
        with st.spinner("Fetching data from Binance..."):
            df = fetch_binance_data(symbol, interval, days)
            
            if df is not None:
                # Afficher un aperçu
                st.markdown("#### 📊 DATA PREVIEW")
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
                    title=f"{symbol}/USDT - {interval}",
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

# Section 3: Datasets existants
st.markdown("### 📂 STORED DATASETS")

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
        
        # Actions sur les datasets
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
        st.info("📭 No datasets stored yet. Create your first one above!")
        
except Exception as e:
    st.error(f"Error loading datasets: {e}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace;'>
    © 2025 CRYPTO SCRAPER | Data from Binance API | Stored in Supabase
</div>
""", unsafe_allow_html=True)
