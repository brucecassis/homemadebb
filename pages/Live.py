# pages/Indices_Europe.py

# Indices Européens Live

import streamlit as st
import yfinance as yf
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

# Auto-refresh toutes les 5 secondes

count = st_autorefresh(interval=5000, limit=None, key=“indices_refresh”)

st.set_page_config(page_title=“Indices Europe Live”, page_icon=“🇪🇺”, layout=“wide”)

# Style Bloomberg

st.markdown(”””

<style>
    .main {background: #000; color: #FFAA00;}
    .stMetric {background: #111; padding: 20px; border-left: 4px solid #FFAA00;}
    .stMetric label {color: #00FFFF !important; font-size: 14px !important;}
    .stMetric [data-testid="stMetricValue"] {color: #FFAA00 !important; font-size: 32px !important;}
    .stMetric [data-testid="stMetricDelta"] {font-size: 16px !important;}
</style>

“””, unsafe_allow_html=True)

# Header

st.markdown(f”””

<div style="background:#FFAA00;padding:10px 20px;color:#000;font-weight:bold;margin-bottom:20px;">
    🇪🇺 INDICES EUROPÉENS LIVE • {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)

# Indices européens

indices = {
‘^FCHI’: ‘🇫🇷 CAC 40’,
‘^GDAXI’: ‘🇩🇪 DAX’,
‘^FTSE’: ‘🇬🇧 FTSE 100’,
‘^STOXX50E’: ‘🇪🇺 EURO STOXX 50’,
‘^IBEX’: ‘🇪🇸 IBEX 35’,
‘FTSEMIB.MI’: ‘🇮🇹 FTSE MIB’
}

cols = st.columns(3)

for idx, (symbol, name) in enumerate(indices.items()):
try:
ticker = yf.Ticker(symbol)
data = ticker.history(period=‘1d’, interval=‘1m’)

```
    if not data.empty:
        price = data['Close'].iloc[-1]
        prev_close = ticker.info.get('previousClose', price)
        change = price - prev_close
        change_pct = (change / prev_close) * 100
        
        with cols[idx % 3]:
            st.metric(
                label=name,
                value=f"{price:,.2f}",
                delta=f"{change:+,.2f} ({change_pct:+.2f}%)"
            )
    else:
        with cols[idx % 3]:
            st.metric(label=name, value="N/A", delta="N/A")
except:
    with cols[idx % 3]:
        st.metric(label=name, value="N/A", delta="N/A")
```

st.markdown(f”””

<div style="text-align:center;color:#666;font-size:10px;margin-top:30px;">
    MAJ automatique toutes les 5s • Yahoo Finance
</div>
""", unsafe_allow_html=True)
