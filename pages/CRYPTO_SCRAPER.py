import streamlit as st
from supabase import create_client, Client
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

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

# Connexion Supabase
@st.cache_resource
def get_supabase_client():
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    return create_client(supabase_url, supabase_key)

supabase = get_supabase_client()


def get_table_list():
    """Récupère la liste des tables via la vue information_schema"""
    try:
        # Requête pour obtenir les tables publiques
        response = supabase.rpc('get_tables_info').execute()
        if response.data:
            return response.data
    except Exception:
        pass
    
    # Fallback: liste des tables connues à vérifier
    known_tables = ['crypto_data', 'crypto_datasets']
    tables_info = []
    
    for table_name in known_tables:
        try:
            # Essayer de compter les lignes
            count_response = supabase.table(table_name).select('*', count='exact', head=True).execute()
            row_count = count_response.count if count_response.count else 0
            tables_info.append({
                'table_name': table_name,
                'row_count': row_count
            })
        except Exception:
            pass
    
    return tables_info


def get_table_data(table_name, limit=1000):
    """Récupère les données d'une table"""
    try:
        response = supabase.table(table_name).select('*').limit(limit).execute()
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


# ===== INTERFACE =====

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
            "Nombre de lignes",
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
        
        st.markdown("#### 📊 DONNÉES")
        
        # Récupérer et afficher les données
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
                # Taille approximative
                size_kb = df.memory_usage(deep=True).sum() / 1024
                st.metric("Taille (KB)", f"{size_kb:.1f}")
            
            # Filtres pour les tables crypto
            if selected_table == 'crypto_data' and 'symbol' in df.columns:
                st.markdown("#### 🔧 FILTRES")
                
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    symbols = ['Tous'] + sorted(df['symbol'].unique().tolist())
                    selected_symbol = st.selectbox("Symbole", symbols)
                
                with filter_col2:
                    if 'timeframe' in df.columns:
                        timeframes = ['Tous'] + sorted(df['timeframe'].unique().tolist())
                        selected_tf = st.selectbox("Timeframe", timeframes)
                    else:
                        selected_tf = 'Tous'
                
                # Appliquer les filtres
                if selected_symbol != 'Tous':
                    df = df[df['symbol'] == selected_symbol]
                if selected_tf != 'Tous' and 'timeframe' in df.columns:
                    df = df[df['timeframe'] == selected_tf]
            
            # Afficher le dataframe
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Graphique si c'est des données crypto avec prix
            if selected_table == 'crypto_data' and all(col in df.columns for col in ['open_time', 'open_price', 'high_price', 'low_price', 'close_price']):
                st.markdown("#### 📈 GRAPHIQUE")
                
                # Convertir les dates
                df['open_time'] = pd.to_datetime(df['open_time'])
                df = df.sort_values('open_time')
                
                fig = go.Figure(data=[go.Candlestick(
                    x=df['open_time'],
                    open=df['open_price'],
                    high=df['high_price'],
                    low=df['low_price'],
                    close=df['close_price']
                )])
                
                title = f"{selected_symbol}/USDT" if selected_symbol != 'Tous' else "Prix"
                
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    yaxis_title="Price (USDT)",
                    height=500,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Export CSV
            st.markdown("#### 💾 EXPORT")
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 TÉLÉCHARGER CSV",
                data=csv,
                file_name=f"{selected_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("📭 Aucune donnée dans cette table")
    
    st.markdown("---")
    
    # Section 3: Actions sur les données
    st.markdown("### 🗑️ GESTION DES DONNÉES")
    
    with st.expander("⚠️ Zone Dangereuse - Suppression de données"):
        st.warning("Attention: Ces actions sont irréversibles!")
        
        delete_table = st.selectbox(
            "Table à nettoyer",
            options=[t['table_name'] for t in tables_info],
            key="delete_table_select"
        )
        
        col_del1, col_del2 = st.columns(2)
        
        with col_del1:
            if delete_table == 'crypto_data':
                # Options de suppression spécifiques
                try:
                    symbols_response = supabase.table('crypto_data').select('symbol').execute()
                    if symbols_response.data:
                        available_symbols = list(set([d['symbol'] for d in symbols_response.data]))
                        symbol_to_delete = st.selectbox("Symbole à supprimer", [''] + available_symbols)
                        
                        if symbol_to_delete and st.button("🗑️ SUPPRIMER CE SYMBOLE", type="secondary"):
                            try:
                                supabase.table('crypto_data').delete().eq('symbol', symbol_to_delete).execute()
                                st.success(f"✅ Données de {symbol_to_delete} supprimées!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erreur: {e}")
                except Exception:
                    pass
        
        with col_del2:
            st.write("")
            st.write("")
            if st.button("🗑️ VIDER TOUTE LA TABLE", type="secondary"):
                confirm = st.checkbox(f"Je confirme vouloir supprimer TOUTES les données de {delete_table}")
                if confirm:
                    try:
                        # Supprimer toutes les lignes
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
    © 2025 CRYPTO DATABASE VIEWER | Connected to Supabase
</div>
""", unsafe_allow_html=True)
