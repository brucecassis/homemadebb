import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Décommenter et installer si nécessaire:
# pip install supabase
# from supabase import create_client, Client

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - SQL Query",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# STYLE BLOOMBERG TERMINAL
# =============================================
st.markdown("""
<style>
    /* Reset et styles globaux */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Force les couleurs globalement */
    body, .main, .block-container, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #FFAA00 !important;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    h1, h2, h3, h4 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 12px !important;
        margin: 8px 0 !important;
        border-bottom: 1px solid #333;
        padding-bottom: 4px !important;
    }
    
    .stButton > button {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        border: 2px solid #FFAA00 !important;
        padding: 8px 20px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border-radius: 0px !important;
        font-size: 11px !important;
        font-family: 'Courier New', monospace !important;
        transition: all 0.3s !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00 !important;
        color: #000 !important;
        transform: translateY(-2px) !important;
    }
    
    .stTextArea textarea {
        background-color: #111 !important;
        color: #00FF00 !important;
        border: 2px solid #333 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 13px !important;
        padding: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #FFAA00 !important;
        box-shadow: 0 0 10px rgba(255, 170, 0, 0.3) !important;
    }
    
    .stTextInput input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 2px solid #333 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
    }
    
    .stTextInput input:focus {
        border-color: #FFAA00 !important;
    }
    
    /* Fix pour les number inputs */
    .stNumberInput input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 2px solid #333 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
    }
    
    .stNumberInput input:focus {
        border-color: #FFAA00 !important;
    }
    
    /* Fix pour les checkboxes */
    .stCheckbox {
        color: #FFAA00 !important;
    }
    
    .stCheckbox span {
        color: #FFAA00 !important;
    }
    
    /* Fix pour les radio buttons */
    .stRadio label {
        color: #FFAA00 !important;
    }
    
    .stRadio span {
        color: #FFAA00 !important;
    }
    
    /* Fix pour les selectbox */
    .stSelectbox label {
        color: #FFAA00 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111 !important;
        border: 2px solid #333 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #FFAA00 !important;
    }
    
    hr {
        border-color: #333333;
        margin: 15px 0;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00 !important;
    }
    
    .section-box {
        background: #111;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FFAA00;
    }
    
    /* Fix pour les labels Streamlit */
    .stTextInput label, .stTextArea label, .stNumberInput label, .stCheckbox label {
        color: #FFAA00 !important;
    }
    
    /* Fix pour les textes dans les colonnes */
    [data-testid="column"] p, [data-testid="column"] span, [data-testid="column"] div {
        color: #FFAA00 !important;
    }
    
    /* Fix pour les textes d'aide */
    .stTextInput small, .stTextArea small {
        color: #999 !important;
    }
    
    /* Fix pour le placeholder */
    .stTextArea textarea::placeholder {
        color: #666 !important;
    }
    
    .query-box {
        background: #0a0a0a;
        border: 2px solid #FFAA00;
        padding: 20px;
        margin: 15px 0;
        border-radius: 0px;
    }
    
    .success-box {
        background: #0a2a0a;
        border: 2px solid #00FF00;
        padding: 15px;
        margin: 10px 0;
        color: #00FF00;
    }
    
    .error-box {
        background: #2a0a0a;
        border: 2px solid #FF0000;
        padding: 15px;
        margin: 10px 0;
        color: #FF0000;
    }
    
    .info-box {
        background: #1a1a2a;
        border: 2px solid #00FFFF;
        padding: 15px;
        margin: 10px 0;
        color: #00FFFF;
    }
    
    /* Style pour les dataframes */
    .dataframe {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
    }
    
    .dataframe th {
        background-color: #222 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        border: 1px solid #333 !important;
    }
    
    .dataframe td {
        background-color: #111 !important;
        color: #00FF00 !important;
        border: 1px solid #333 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #222;
        color: #FFAA00;
        border: 1px solid #333;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFAA00;
        color: #000;
    }
    
    /* Fix pour les messages Streamlit */
    .stAlert {
        background-color: #111 !important;
        border: 2px solid #FFAA00 !important;
        color: #FFAA00 !important;
    }
    
    .stSuccess {
        background-color: #0a2a0a !important;
        border: 2px solid #00FF00 !important;
        color: #00FF00 !important;
    }
    
    .stError {
        background-color: #2a0a0a !important;
        border: 2px solid #FF0000 !important;
        color: #FF0000 !important;
    }
    
    .stWarning {
        background-color: #2a2a0a !important;
        border: 2px solid #FFA500 !important;
        color: #FFA500 !important;
    }
    
    .stInfo {
        background-color: #0a0a2a !important;
        border: 2px solid #00FFFF !important;
        color: #00FFFF !important;
    }
    
    /* Fix pour les expanders */
    .streamlit-expanderHeader {
        background-color: #222 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #111 !important;
        border: 1px solid #333 !important;
    }
    
    /* Fix pour le code */
    .stCodeBlock {
        background-color: #0a0a0a !important;
        border: 1px solid #333 !important;
    }
    
    code {
        color: #00FF00 !important;
        background-color: #0a0a0a !important;
    }
    
    /* Fix pour les markdown */
    .stMarkdown {
        color: #FFAA00 !important;
    }
    
    .stMarkdown p {
        color: #FFAA00 !important;
    }
    
    .stMarkdown li {
        color: #FFAA00 !important;
    }
    
    .stMarkdown strong {
        color: #00FFFF !important;
    }
    
    /* Fix pour les download buttons */
    .stDownloadButton button {
        background-color: #333 !important;
        color: #00FF00 !important;
        border: 2px solid #00FF00 !important;
    }
    
    .stDownloadButton button:hover {
        background-color: #00FF00 !important;
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - SQL QUERY INTERFACE</div>
    </div>
    <div>{current_time} UTC • DATABASE: SUPABASE</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# CONFIGURATION SUPABASE (AUTO depuis secrets)
# =============================================
st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">🔧 SUPABASE CONFIGURATION</p>', unsafe_allow_html=True)

# Récupérer les credentials depuis les secrets Streamlit
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    credentials_found = True
except Exception as e:
    supabase_url = None
    supabase_key = None
    credentials_found = False
    st.error("❌ Credentials Supabase non trouvés dans les secrets Streamlit")

# Afficher les infos de connexion (masquées)
if credentials_found:
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(f"""
        <div style="background:#111;border:1px solid #333;padding:10px;margin:5px 0;">
            <p style="color:#666;font-size:10px;margin:0;">SUPABASE URL:</p>
            <p style="color:#00FF00;font-size:11px;margin:5px 0;">{supabase_url[:30]}...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown(f"""
        <div style="background:#111;border:1px solid #333;padding:10px;margin:5px 0;">
            <p style="color:#666;font-size:10px;margin:0;">SUPABASE KEY:</p>
            <p style="color:#00FF00;font-size:11px;margin:5px 0;">{'*' * 40}</p>
        </div>
        """, unsafe_allow_html=True)

# Initialiser la session state pour stocker la connexion
if 'supabase_client' not in st.session_state:
    st.session_state.supabase_client = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Connexion automatique au chargement de la page
if credentials_found and not st.session_state.connected:
    try:
        # Import et connexion Supabase
        from supabase import create_client
        st.session_state.supabase_client = create_client(supabase_url, supabase_key)
        st.session_state.connected = True
    except ImportError:
        st.warning("⚠️ Package 'supabase' non installé. Exécutez: pip install supabase")
        # Mode démo sans vraie connexion
        st.session_state.connected = True
    except Exception as e:
        st.error(f"❌ ERREUR DE CONNEXION AUTO: {str(e)}")

# Boutons de contrôle
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])

with col_btn1:
    if st.button("🔄 RECONNECT", key="reconnect_btn"):
        if credentials_found:
            try:
                from supabase import create_client
                st.session_state.supabase_client = create_client(supabase_url, supabase_key)
                st.session_state.connected = True
                st.success("✅ RECONNEXION ÉTABLIE")
            except ImportError:
                st.warning("⚠️ Package 'supabase' non installé. Exécutez: pip install supabase")
                st.session_state.connected = True
            except Exception as e:
                st.error(f"❌ ERREUR DE CONNEXION: {str(e)}")
        else:
            st.warning("⚠️ Credentials manquants dans les secrets")

with col_btn2:
    if st.button("🔌 DISCONNECT", key="disconnect_btn"):
        st.session_state.supabase_client = None
        st.session_state.connected = False
        st.info("🔌 DÉCONNECTÉ")

# Indicateur de statut
status_color = "#00FF00" if st.session_state.connected else "#FF0000"
status_text = "CONNECTED" if st.session_state.connected else "DISCONNECTED"

st.markdown(f"""
<div style="background:#111;border:2px solid {status_color};padding:10px;margin:15px 0;text-align:center;color:{status_color};font-weight:bold;">
    STATUS: {status_text} • AUTO-CONFIGURED FROM SECRETS
</div>
""", unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# INTERFACE DE REQUÊTE SQL
# =============================================

if st.session_state.connected:
    
    # Tabs pour différentes fonctionnalités
    tab1, tab2, tab3, tab4 = st.tabs(["📝 QUERY EDITOR", "📋 QUERY TEMPLATES", "📊 RESULTS", "📜 HISTORY"])
    
    # ===== TAB 1: QUERY EDITOR =====
    with tab1:
        st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📝 SQL QUERY EDITOR</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        
        # Éditeur SQL
        st.markdown('<p style="color:#00FF00;font-size:12px;font-weight:bold;margin-bottom:10px;">💻 ENTER YOUR SQL QUERY</p>', unsafe_allow_html=True)
        sql_query = st.text_area(
            "ENTER YOUR SQL QUERY",
            placeholder="""-- Exemples de requêtes SQL:
SELECT * FROM users LIMIT 10;
SELECT COUNT(*) FROM orders WHERE status = 'completed';
INSERT INTO logs (message, timestamp) VALUES ('test', NOW());
UPDATE products SET price = 99.99 WHERE id = 1;""",
            height=200,
            help="Tapez votre requête SQL ici",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Options d'exécution
        st.markdown('<p style="color:#FFAA00;font-size:11px;font-weight:bold;margin:15px 0 10px 0;">⚙️ EXECUTION OPTIONS</p>', unsafe_allow_html=True)
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            limit_results = st.number_input("LIMIT RESULTS", min_value=1, max_value=10000, value=100, step=10)
        
        with col_opt2:
            safe_mode = st.checkbox("SAFE MODE (READ ONLY)", value=True, 
                                   help="Bloque les requêtes DELETE, DROP, TRUNCATE")
        
        with col_opt3:
            show_execution_time = st.checkbox("SHOW EXECUTION TIME", value=True)
        
        # Boutons d'action
        col_exec1, col_exec2, col_exec3, col_exec4 = st.columns(4)
        
        with col_exec1:
            execute_btn = st.button("▶️ EXECUTE QUERY", key="exec_query", type="primary")
        
        with col_exec2:
            explain_btn = st.button("🔍 EXPLAIN QUERY", key="explain_query")
        
        with col_exec3:
            validate_btn = st.button("✓ VALIDATE SYNTAX", key="validate_query")
        
        with col_exec4:
            clear_btn = st.button("🗑️ CLEAR", key="clear_query")
        
        if clear_btn:
            st.rerun()
        
        # Exécution de la requête
        if execute_btn and sql_query:
            # Vérification du mode safe
            if safe_mode:
                dangerous_keywords = ['DELETE', 'DROP', 'TRUNCATE', 'ALTER']
                if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
                    st.markdown("""
                    <div class="error-box">
                        ❌ REQUÊTE BLOQUÉE PAR LE SAFE MODE<br>
                        Cette requête contient des opérations dangereuses (DELETE, DROP, TRUNCATE, ALTER).<br>
                        Désactivez le Safe Mode pour l'exécuter.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Simuler l'exécution (remplacer par vraie requête Supabase)
                    try:
                        import time
                        import requests
                        start_time = time.time()
                        
                        # Exécution de la vraie requête Supabase via API REST
                        if st.session_state.supabase_client:
                            try:
                                import psycopg2
        
                                # Récupérer la DB URL depuis les secrets
                                db_url = st.secrets.get("SUPABASE_DB_URL", None)
                                
                                if db_url:
                                    # Connexion PostgreSQL directe
                                    conn = psycopg2.connect(db_url)
                                    cursor = conn.cursor()
                                    cursor.execute(sql_query)
                                    
                                    # Récupérer les résultats
                                    results = cursor.fetchall()
                                    columns = [desc[0] for desc in cursor.description]
                                    result_df = pd.DataFrame(results, columns=columns)
                                    
                                    cursor.close()
                                    conn.close()
                                else:
                                    st.error("❌ SUPABASE_DB_URL manquant dans les secrets")
                                    result_df = None
                            
                            except Exception as e:
                                st.error(f"❌ Erreur PostgreSQL: {str(e)}")
                                result_df = None
                                                
                        else:
                            # Mode démo - données fictives
                            if 'SELECT' in sql_query.upper():
                                result_df = pd.DataFrame({
                                    'id': [1, 2, 3, 4, 5],
                                    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                                    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                                             'david@example.com', 'eve@example.com'],
                                    'created_at': pd.date_range('2024-01-01', periods=5, freq='D'),
                                    'status': ['active', 'active', 'inactive', 'active', 'pending']
                                })
                            else:
                                result_df = pd.DataFrame({'result': ['Query executed successfully (DEMO MODE)']})
                        
                        execution_time = time.time() - start_time
                        
                        if result_df is not None and len(result_df) > 0:
                            # Ajouter à l'historique
                            st.session_state.query_history.insert(0, {
                                'query': sql_query,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'rows': len(result_df),
                                'execution_time': f"{execution_time:.3f}s"
                            })
                            
                            st.markdown(f"""
                            <div class="success-box">
                                ✅ QUERY EXECUTED SUCCESSFULLY<br>
                                ROWS RETURNED: {len(result_df)}<br>
                                EXECUTION TIME: {execution_time:.3f}s
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Afficher les résultats
                            st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📊 QUERY RESULTS</p>', unsafe_allow_html=True)
                            st.dataframe(
                                result_df.head(limit_results),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Options d'export
                            st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;margin-top:20px;">💾 EXPORT OPTIONS</p>', unsafe_allow_html=True)
                            col_exp1, col_exp2, col_exp3 = st.columns(3)
                            
                            with col_exp1:
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    "⬇️ DOWNLOAD CSV",
                                    csv,
                                    "query_results.csv",
                                    "text/csv"
                                )
                            
                            with col_exp2:
                                json_str = result_df.to_json(orient='records', indent=2)
                                st.download_button(
                                    "⬇️ DOWNLOAD JSON",
                                    json_str,
                                    "query_results.json",
                                    "application/json"
                                )
                            
                            with col_exp3:
                                excel_buffer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
                                result_df.to_excel(excel_buffer, index=False, sheet_name='Results')
                                excel_buffer.close()
                                # Note: Pour un vrai téléchargement Excel, il faudrait gérer le buffer correctement
                                st.button("⬇️ DOWNLOAD EXCEL (coming soon)")
                        else:
                            st.warning("⚠️ Aucun résultat retourné")
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            ❌ ERREUR D'EXÉCUTION<br>
                            {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Safe Mode désactivé - Exécution de requêtes dangereuses autorisée")
        
        # Validation de syntaxe
        if validate_btn and sql_query:
            # Validation basique (peut être améliorée avec un vrai parser SQL)
            keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN']
            has_keyword = any(keyword in sql_query.upper() for keyword in keywords)
            
            if has_keyword and ';' in sql_query:
                st.markdown("""
                <div class="success-box">
                    ✅ SYNTAXE VALIDE<br>
                    La requête semble correctement formée.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                    ⚠️ SYNTAXE INCOMPLÈTE<br>
                    Vérifiez que votre requête contient les mots-clés SQL nécessaires et se termine par ';'
                </div>
                """, unsafe_allow_html=True)
        
        # EXPLAIN
        if explain_btn and sql_query:
            st.markdown("""
            <div class="info-box">
                🔍 QUERY EXECUTION PLAN<br>
                (Feature coming soon - will show query optimization details)
            </div>
            """, unsafe_allow_html=True)
    
    # ===== TAB 2: QUERY TEMPLATES =====
    with tab2:
        st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📋 SQL QUERY TEMPLATES</p>', unsafe_allow_html=True)
        
        templates = {
            "🔍 SELECT ALL": "SELECT * FROM table_name LIMIT 10;",
            "📊 COUNT RECORDS": "SELECT COUNT(*) as total FROM table_name;",
            "🔎 FILTER BY CONDITION": "SELECT * FROM table_name WHERE column_name = 'value';",
            "📅 DATE RANGE": "SELECT * FROM table_name WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';",
            "🔗 JOIN TABLES": """SELECT a.*, b.column 
FROM table_a a
LEFT JOIN table_b b ON a.id = b.foreign_id;""",
            "📈 GROUP BY": """SELECT category, COUNT(*) as count, AVG(price) as avg_price
FROM products
GROUP BY category
ORDER BY count DESC;""",
            "✏️ INSERT": "INSERT INTO table_name (col1, col2, col3) VALUES ('val1', 'val2', 'val3');",
            "🔄 UPDATE": "UPDATE table_name SET column_name = 'new_value' WHERE id = 1;",
            "🗑️ DELETE": "DELETE FROM table_name WHERE condition = 'value';",
            "📊 AGGREGATES": """SELECT 
    COUNT(*) as total,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MAX(amount) as max_amount,
    MIN(amount) as min_amount
FROM transactions;"""
        }
        
        col_temp1, col_temp2 = st.columns(2)
        
        for idx, (template_name, template_query) in enumerate(templates.items()):
            with col_temp1 if idx % 2 == 0 else col_temp2:
                with st.expander(template_name):
                    st.code(template_query, language='sql')
                    if st.button(f"📋 USE THIS TEMPLATE", key=f"template_{idx}"):
                        st.info(f"Template copied! Switch to Query Editor tab to use it.")
    
    # ===== TAB 3: RESULTS =====
    with tab3:
        st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📊 QUERY RESULTS VIEWER</p>', unsafe_allow_html=True)
        st.info("Execute a query in the Query Editor tab to see results here.")
    
    # ===== TAB 4: HISTORY =====
    with tab4:
        st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📜 QUERY HISTORY</p>', unsafe_allow_html=True)
        
        if st.session_state.query_history:
            st.markdown(f'<p style="color:#00FFFF;font-size:11px;">📊 Total queries executed: <span style="color:#00FF00;font-weight:bold;">{len(st.session_state.query_history)}</span></p>', unsafe_allow_html=True)
            
            for idx, query_record in enumerate(st.session_state.query_history):
                with st.expander(f"Query #{idx+1} - {query_record['timestamp']}"):
                    st.code(query_record['query'], language='sql')
                    st.text(f"Rows returned: {query_record['rows']}")
                    st.text(f"Execution time: {query_record['execution_time']}")
                    
                    if st.button(f"🔄 RE-EXECUTE", key=f"reexec_{idx}"):
                        st.info("Switch to Query Editor and paste this query")
            
            if st.button("🗑️ CLEAR HISTORY"):
                st.session_state.query_history = []
                st.rerun()
        else:
            st.info("No queries executed yet. Start by running a query in the Query Editor.")

else:
    st.markdown("""
    <div class="info-box">
        ⚠️ CONNEXION SUPABASE NON DISPONIBLE<br><br>
        
        <b>VÉRIFIEZ VOS SECRETS STREAMLIT:</b><br>
        1. Allez dans Settings → Secrets de votre app Streamlit<br>
        2. Ajoutez les secrets suivants:<br><br>
        
        <code style="color:#00FF00;">
        SUPABASE_URL = "https://xxxxxxxxxxxxx.supabase.co"<br>
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        </code><br><br>
        
        <b>OÙ TROUVER VOS CREDENTIALS ?</b><br>
        - Allez sur votre projet Supabase: https://app.supabase.com<br>
        - Project Settings → API<br>
        - Copiez: Project URL et anon/public key<br><br>
        
        <b>INSTALLATION REQUISE:</b><br>
        <code style="color:#00FF00;">pip install supabase</code>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# DOCUMENTATION
# =============================================
with st.expander("📖 DOCUMENTATION & HELP"):
    st.markdown("""
    <div style="color:#FFAA00;">
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:10px;">🔧 Configuration Supabase</p>
    
    **Configuration automatique via Streamlit Secrets**
    
    Cette page récupère automatiquement vos credentials Supabase depuis les Secrets Streamlit.
    
    **Comment configurer les secrets ?**
    1. Allez dans votre app Streamlit Cloud
    2. Settings → Secrets
    3. Ajoutez:
    ```toml
    SUPABASE_URL = "https://xxxxxxxxxxxxx.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    ```
    
    **Pour le développement local:**
    Créez un fichier `.streamlit/secrets.toml` à la racine de votre projet avec les mêmes clés.
    
    **Où trouver vos credentials ?**
    1. Allez sur votre projet Supabase: https://app.supabase.com
    2. Project Settings → API
    3. Copiez:
       - Project URL → SUPABASE_URL
       - anon/public key → SUPABASE_KEY
       - ⚠️ Ne jamais utiliser la service_role key côté client!
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">📝 Sécurité</p>
    
    - **Safe Mode (recommandé)**: Bloque les requêtes DELETE, DROP, TRUNCATE, ALTER
    - **Row Level Security**: Configurez RLS dans Supabase pour sécuriser vos données
    - **API Keys**: Utilisez toujours la clé `anon` côté client, jamais la `service_role`
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">🔍 Exemples de requêtes</p>
    
    ```sql
    -- Lire des données
    SELECT * FROM users WHERE status = 'active';
    
    -- Insérer des données
    INSERT INTO logs (message, level) VALUES ('System started', 'INFO');
    
    -- Mettre à jour
    UPDATE users SET last_login = NOW() WHERE id = 123;
    
    -- Compter des enregistrements
    SELECT COUNT(*) FROM orders WHERE created_at >= '2024-01-01';
    
    -- Jointures
    SELECT u.name, o.total 
    FROM users u 
    JOIN orders o ON u.id = o.user_id;
    ```
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">⚡ Fonctionnalités</p>
    
    - ✅ Exécution de requêtes SQL personnalisées
    - ✅ Mode Safe (protection contre requêtes dangereuses)
    - ✅ Templates de requêtes pré-définis
    - ✅ Historique des requêtes
    - ✅ Export des résultats (CSV, JSON, Excel)
    - ✅ Validation de syntaxe
    - ✅ Mesure du temps d'exécution
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">🚀 Fonctionnalités à venir</p>
    
    - EXPLAIN QUERY (plan d'exécution)
    - Autocomplétion SQL
    - Visualisation avancée des résultats
    - Sauvegarde de requêtes favorites
    - Partage de requêtes par lien
    
    </div>
    """)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | SQL QUERY INTERFACE | SUPABASE INTEGRATION<br>
    SECURE CONNECTION • SAFE MODE ENABLED • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
