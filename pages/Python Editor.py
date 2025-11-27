import streamlit as st
import sys
import io
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Import optionnel de matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Python Terminal",
    page_icon="🐍",
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
        background-color: #0a0a0a !important;
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
    
    .stCheckbox {
        color: #FFAA00 !important;
    }
    
    .stCheckbox span {
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
    
    .terminal-output {
        background: #0a0a0a;
        border: 2px solid #00FF00;
        padding: 15px;
        margin: 10px 0;
        color: #00FF00;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        min-height: 200px;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .terminal-error {
        background: #2a0a0a;
        border: 2px solid #FF0000;
        padding: 15px;
        margin: 10px 0;
        color: #FF0000;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .code-editor-box {
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
    
    .info-box {
        background: #1a1a2a;
        border: 2px solid #00FFFF;
        padding: 15px;
        margin: 10px 0;
        color: #00FFFF;
    }
    
    .warning-box {
        background: #2a2a0a;
        border: 2px solid #FFA500;
        padding: 15px;
        margin: 10px 0;
        color: #FFA500;
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
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333;
        border: 1px solid #FFAA00;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FFAA00;
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
        <div>⬛ BLOOMBERG ENS® TERMINAL - PYTHON INTERACTIVE SHELL</div>
    </div>
    <div>{current_time} UTC • PYTHON {sys.version.split()[0]}</div>
</div>
""", unsafe_allow_html=True)

# Avertissement si matplotlib n'est pas disponible
if not MATPLOTLIB_AVAILABLE:
    st.markdown("""
    <div style="background:#2a2a0a;border:2px solid #FFA500;padding:10px;margin:10px 0;color:#FFA500;font-size:10px;">
        ⚠️ matplotlib n'est pas installé. Les fonctionnalités de visualisation sont limitées.<br>
        Pour l'installer: pip install matplotlib
    </div>
    """, unsafe_allow_html=True)

# =============================================
# INITIALISATION SESSION STATE
# =============================================
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []
if 'terminal_output' not in st.session_state:
    st.session_state.terminal_output = []
if 'variables' not in st.session_state:
    st.session_state.variables = {}
if 'code_templates' not in st.session_state:
    st.session_state.code_templates = {}

# =============================================
# FONCTIONS UTILITAIRES
# =============================================

def execute_python_code(code, use_persistent_vars=True):
    """
    Exécute du code Python et capture la sortie
    """
    # Rediriger stdout et stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_error
    
    result = {
        'success': False,
        'output': '',
        'error': '',
        'returned_value': None,
        'execution_time': 0
    }
    
    try:
        import time
        start_time = time.time()
        
        # Préparer l'environnement d'exécution
        if use_persistent_vars:
            exec_globals = st.session_state.variables.copy()
        else:
            exec_globals = {}
        
        # Ajouter les imports communs
        exec_globals.update({
            'pd': pd,
            'np': np,
            'json': json,
            'st': st
        })
        
        # Ajouter matplotlib seulement si disponible
        if MATPLOTLIB_AVAILABLE:
            exec_globals['plt'] = plt
        
        exec_locals = {}
        
        # Exécuter le code
        exec(code, exec_globals, exec_locals)
        
        # Mettre à jour les variables persistantes
        if use_persistent_vars:
            st.session_state.variables.update(exec_locals)
        
        execution_time = time.time() - start_time
        
        # Capturer la sortie
        output = redirected_output.getvalue()
        error = redirected_error.getvalue()
        
        result['success'] = True
        result['output'] = output
        result['error'] = error
        result['execution_time'] = execution_time
        
        # Récupérer la dernière valeur retournée si elle existe
        if exec_locals:
            last_var = list(exec_locals.values())[-1] if exec_locals else None
            result['returned_value'] = last_var
        
    except Exception as e:
        result['success'] = False
        result['error'] = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
    
    finally:
        # Restaurer stdout et stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result

# =============================================
# INTERFACE PRINCIPALE
# =============================================

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🐍 PYTHON EDITOR", "📦 TEMPLATES", "📊 VARIABLES", "📜 HISTORY"])

# ===== TAB 1: PYTHON EDITOR =====
with tab1:
    st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">🐍 PYTHON CODE EDITOR</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="code-editor-box">', unsafe_allow_html=True)
    
    # Éditeur de code
    st.markdown('<p style="color:#00FF00;font-size:12px;font-weight:bold;margin-bottom:10px;">💻 ENTER YOUR PYTHON CODE</p>', unsafe_allow_html=True)
    
    default_code = """# Exemple de code Python
import pandas as pd
import numpy as np

# Créer des données
data = {
    'nom': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'salaire': [50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)
print("DataFrame créé:")
print(df)

# Calculs
moyenne_age = df['age'].mean()
print(f"\\nÂge moyen: {moyenne_age}")

# Résultat
df"""
    
    python_code = st.text_area(
        "PYTHON CODE",
        value=default_code,
        height=300,
        help="Tapez votre code Python ici. Les bibliothèques pandas, numpy, matplotlib et streamlit sont pré-importées.",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Options d'exécution
    st.markdown('<p style="color:#FFAA00;font-size:11px;font-weight:bold;margin:15px 0 10px 0;">⚙️ EXECUTION OPTIONS</p>', unsafe_allow_html=True)
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        persistent_vars = st.checkbox("PERSISTENT VARIABLES", value=True, 
                                      help="Conserver les variables entre les exécutions")
    
    with col_opt2:
        show_exec_time = st.checkbox("SHOW EXECUTION TIME", value=True)
    
    with col_opt3:
        auto_display = st.checkbox("AUTO-DISPLAY RESULT", value=True,
                                   help="Afficher automatiquement la dernière valeur")
    
    # Boutons d'action
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        run_btn = st.button("▶️ RUN CODE", key="run_code", type="primary")
    
    with col_btn2:
        clear_output_btn = st.button("🗑️ CLEAR OUTPUT", key="clear_output")
    
    with col_btn3:
        clear_vars_btn = st.button("🔄 RESET VARIABLES", key="clear_vars")
    
    with col_btn4:
        save_template_btn = st.button("💾 SAVE AS TEMPLATE", key="save_template")
    
    # Exécution du code
    if run_btn and python_code.strip():
        with st.spinner("⚡ EXECUTING CODE..."):
            result = execute_python_code(python_code, persistent_vars)
            
            # Ajouter à l'historique
            st.session_state.execution_history.insert(0, {
                'code': python_code,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': result['success'],
                'execution_time': result['execution_time']
            })
            
            # Ajouter au terminal output
            terminal_entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'execution',
                'success': result['success'],
                'output': result['output'],
                'error': result['error'],
                'execution_time': result['execution_time']
            }
            st.session_state.terminal_output.append(terminal_entry)
            
            # Afficher les résultats
            if result['success']:
                st.markdown(f"""
                <div class="success-box">
                    ✅ CODE EXECUTED SUCCESSFULLY<br>
                    EXECUTION TIME: {result['execution_time']:.4f}s
                </div>
                """, unsafe_allow_html=True)
                
                # Terminal Output
                st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;margin-top:20px;">📺 TERMINAL OUTPUT</p>', unsafe_allow_html=True)
                
                if result['output']:
                    st.markdown(f"""
                    <div class="terminal-output">
                    <span style="color:#00FFFF;">>>> STDOUT:</span><br>
                    {result['output']}
                    </div>
                    """, unsafe_allow_html=True)
                
                if result['error']:
                    st.markdown(f"""
                    <div class="terminal-output" style="border-color:#FFA500;color:#FFA500;">
                    <span style="color:#FFA500;">>>> STDERR:</span><br>
                    {result['error']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Afficher la valeur retournée
                if auto_display and result['returned_value'] is not None:
                    st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;margin-top:20px;">📊 RETURNED VALUE</p>', unsafe_allow_html=True)
                    
                    # Si c'est un DataFrame, l'afficher avec style
                    if isinstance(result['returned_value'], pd.DataFrame):
                        st.dataframe(
                            result['returned_value'],
                            use_container_width=True,
                            height=400
                        )
                        
                        # Boutons d'export pour DataFrame
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        with col_exp1:
                            csv = result['returned_value'].to_csv(index=False)
                            st.download_button("⬇️ DOWNLOAD CSV", csv, "data.csv", "text/csv")
                        with col_exp2:
                            json_str = result['returned_value'].to_json(orient='records', indent=2)
                            st.download_button("⬇️ DOWNLOAD JSON", json_str, "data.json", "application/json")
                    
                    # Si c'est une figure matplotlib
                    elif MATPLOTLIB_AVAILABLE and isinstance(result['returned_value'], plt.Figure):
                        st.pyplot(result['returned_value'])
                    
                    # Sinon, afficher comme texte
                    else:
                        st.code(str(result['returned_value']), language='python')
                
                if not result['output'] and result['returned_value'] is None:
                    st.markdown("""
                    <div class="info-box">
                        ℹ️ Code executed successfully with no output
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                <div class="terminal-error">
                    ❌ EXECUTION ERROR<br><br>
                    {result['error']}
                </div>
                """, unsafe_allow_html=True)
    
    # Clear output
    if clear_output_btn:
        st.session_state.terminal_output = []
        st.success("✅ Terminal output cleared")
        st.rerun()
    
    # Reset variables
    if clear_vars_btn:
        st.session_state.variables = {}
        st.success("✅ Variables reset")
        st.rerun()
    
    # Save as template
    if save_template_btn and python_code.strip():
        template_name = st.text_input("Template name:", key="template_name_input")
        if template_name:
            st.session_state.code_templates[template_name] = python_code
            st.success(f"✅ Template '{template_name}' saved!")

# ===== TAB 2: TEMPLATES =====
with tab2:
    st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📦 CODE TEMPLATES</p>', unsafe_allow_html=True)
    
    # Templates prédéfinis
    predefined_templates = {
        "🐼 DataFrame Basics": """import pandas as pd
import numpy as np

# Créer un DataFrame
df = pd.DataFrame({
    'A': np.random.rand(10),
    'B': np.random.randint(0, 100, 10),
    'C': ['cat_' + str(i) for i in range(10)]
})

print("DataFrame Info:")
print(df.info())
print("\\nFirst 5 rows:")
print(df.head())

df""",
        
        "🔢 Statistical Analysis": """import pandas as pd
import numpy as np

# Générer des données aléatoires
np.random.seed(42)
data = pd.DataFrame({
    'values': np.random.normal(100, 15, 1000)
})

# Statistiques descriptives
stats = data['values'].describe()
print("Statistiques descriptives:")
print(stats)

# Calculs supplémentaires
median = data['values'].median()
variance = data['values'].var()

print(f"\\nMédiane: {median:.2f}")
print(f"Variance: {variance:.2f}")

data""",
        
        "📈 Time Series": """import pandas as pd
import numpy as np

# Créer une série temporelle
dates = pd.date_range('2024-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

ts = pd.DataFrame({
    'date': dates,
    'value': values
})

ts.set_index('date', inplace=True)

print("Time Series Info:")
print(ts.info())
print("\\nFirst 10 days:")
print(ts.head(10))

ts""",
        
        "🔍 Data Filtering": """import pandas as pd
import numpy as np

# Créer un DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
})

print("Original DataFrame:")
print(df)

# Filtrage
it_dept = df[df['department'] == 'IT']
print("\\nIT Department:")
print(it_dept)

high_salary = df[df['salary'] > 65000]
print("\\nHigh Salary (>65k):")
print(high_salary)

df""",
        
        "🧮 Mathematical Operations": """import numpy as np

# Créer des matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\\nMatrix B:")
print(B)

# Opérations
print("\\nA + B:")
print(A + B)

print("\\nA @ B (multiplication matricielle):")
print(A @ B)

print("\\nDéterminant de A:")
print(np.linalg.det(A))

print("\\nValeurs propres de A:")
print(np.linalg.eigvals(A))

A""",
        
        "📊 Grouping & Aggregation": """import pandas as pd
import numpy as np

# Créer des données de ventes
np.random.seed(42)
df = pd.DataFrame({
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'quantity': np.random.randint(1, 10, 100)
})

print("Données de ventes:")
print(df.head(10))

# Grouper et agréger
grouped = df.groupby(['product', 'region']).agg({
    'sales': ['sum', 'mean', 'count'],
    'quantity': 'sum'
}).round(2)

print("\\nAgrégation par produit et région:")
print(grouped)

grouped"""
    }
    
    # Ajouter le template de visualisation seulement si matplotlib est disponible
    if MATPLOTLIB_AVAILABLE:
        predefined_templates["📊 Data Visualization"] = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Créer des données
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Créer le graphique
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, label='sin(x)', linewidth=2)
ax.plot(x, y2, label='cos(x)', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Sine and Cosine Functions')
ax.legend()
ax.grid(True, alpha=0.3)

print("Plot créé!")
fig"""
    
    st.markdown('<p style="color:#00FFFF;font-size:11px;font-weight:bold;margin:15px 0;">🎯 PREDEFINED TEMPLATES</p>', unsafe_allow_html=True)
    
    col_temp1, col_temp2 = st.columns(2)
    
    for idx, (template_name, template_code) in enumerate(predefined_templates.items()):
        with col_temp1 if idx % 2 == 0 else col_temp2:
            with st.expander(template_name):
                st.code(template_code, language='python')
                if st.button(f"📋 USE THIS TEMPLATE", key=f"use_template_{idx}"):
                    st.info(f"Template loaded! Switch to Python Editor tab.")
    
    # Templates utilisateur
    if st.session_state.code_templates:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<p style="color:#00FFFF;font-size:11px;font-weight:bold;margin:15px 0;">💾 YOUR SAVED TEMPLATES</p>', unsafe_allow_html=True)
        
        for template_name, template_code in st.session_state.code_templates.items():
            with st.expander(f"📌 {template_name}"):
                st.code(template_code, language='python')
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    if st.button(f"📋 USE", key=f"use_saved_{template_name}"):
                        st.info(f"Template '{template_name}' loaded!")
                with col_t2:
                    if st.button(f"🗑️ DELETE", key=f"delete_saved_{template_name}"):
                        del st.session_state.code_templates[template_name]
                        st.rerun()

# ===== TAB 3: VARIABLES =====
with tab3:
    st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📊 SESSION VARIABLES</p>', unsafe_allow_html=True)
    
    if st.session_state.variables:
        st.markdown(f'<p style="color:#00FFFF;font-size:11px;">📦 Total variables: <span style="color:#00FF00;font-weight:bold;">{len(st.session_state.variables)}</span></p>', unsafe_allow_html=True)
        
        for var_name, var_value in st.session_state.variables.items():
            with st.expander(f"🔹 {var_name} ({type(var_value).__name__})"):
                try:
                    # Afficher différemment selon le type
                    if isinstance(var_value, pd.DataFrame):
                        st.dataframe(var_value, use_container_width=True)
                    elif isinstance(var_value, (list, dict, tuple)):
                        st.json(var_value if isinstance(var_value, dict) else {str(i): v for i, v in enumerate(var_value)})
                    else:
                        st.code(str(var_value), language='python')
                except:
                    st.text(f"Cannot display: {type(var_value)}")
        
        if st.button("🗑️ CLEAR ALL VARIABLES"):
            st.session_state.variables = {}
            st.success("✅ All variables cleared")
            st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
            ℹ️ No variables in session yet.<br>
            Run code with "PERSISTENT VARIABLES" option enabled to store variables.
        </div>
        """, unsafe_allow_html=True)

# ===== TAB 4: HISTORY =====
with tab4:
    st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">📜 EXECUTION HISTORY</p>', unsafe_allow_html=True)
    
    if st.session_state.execution_history:
        st.markdown(f'<p style="color:#00FFFF;font-size:11px;">📊 Total executions: <span style="color:#00FF00;font-weight:bold;">{len(st.session_state.execution_history)}</span></p>', unsafe_allow_html=True)
        
        for idx, record in enumerate(st.session_state.execution_history):
            status_icon = "✅" if record['success'] else "❌"
            status_color = "#00FF00" if record['success'] else "#FF0000"
            
            with st.expander(f"{status_icon} Execution #{idx+1} - {record['timestamp']} ({record['execution_time']:.4f}s)"):
                st.code(record['code'], language='python')
                st.markdown(f'<p style="color:{status_color};font-weight:bold;">Status: {"SUCCESS" if record["success"] else "FAILED"}</p>', unsafe_allow_html=True)
                st.text(f"Execution time: {record['execution_time']:.4f}s")
                
                if st.button(f"🔄 RE-RUN", key=f"rerun_{idx}"):
                    st.info("Code copied! Go to Python Editor to execute.")
        
        if st.button("🗑️ CLEAR HISTORY"):
            st.session_state.execution_history = []
            st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
            ℹ️ No execution history yet.<br>
            Start by running some Python code in the editor.
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# DOCUMENTATION
# =============================================
with st.expander("📖 DOCUMENTATION & HELP"):
    st.markdown("""
    <div style="color:#FFAA00;">
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:10px;">🐍 Python Interactive Terminal</p>
    
    Cette interface vous permet d'exécuter du code Python directement dans votre navigateur avec un terminal interactif.
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">📚 Bibliothèques Pré-importées</p>
    
    Les bibliothèques suivantes sont automatiquement disponibles:
    - **pandas** (as pd): Manipulation de données
    - **numpy** (as np): Calculs numériques
    - **json**: Manipulation JSON
    - **streamlit** (as st): Interface Streamlit
    - **matplotlib.pyplot** (as plt): Visualisation de données (si installé)
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">⚙️ Fonctionnalités</p>
    
    - **Persistent Variables**: Les variables sont conservées entre les exécutions
    - **Terminal Output**: Capture de stdout et stderr
    - **Auto-Display**: Affichage automatique de la dernière valeur retournée
    - **Execution Time**: Mesure du temps d'exécution
    - **Code Templates**: Bibliothèque de templates pré-définis
    - **History**: Historique complet des exécutions
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">💡 Exemples d'utilisation</p>
    
    ```python
    # Exemple 1: Créer et manipuler un DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    print(df)
    df  # Affichage automatique
    ```
    
    ```python
    # Exemple 2: Visualisation
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    fig  # Affichage automatique du graphique
    ```
    
    ```python
    # Exemple 3: Variables persistantes
    # Exécution 1:
    x = 10
    
    # Exécution 2 (dans un autre run):
    y = x * 2  # x est toujours disponible!
    print(y)  # Affiche 20
    ```
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">🎯 Conseils</p>
    
    - Utilisez `print()` pour afficher des valeurs intermédiaires
    - La dernière ligne sans assignation sera affichée automatiquement
    - Les DataFrames et figures matplotlib ont un affichage spécial
    - Consultez l'onglet Variables pour voir toutes les variables actives
    - Sauvegardez vos codes fréquents comme templates
    
    <p style="color:#00FFFF;font-weight:bold;font-size:13px;margin-top:20px;">⚠️ Limitations</p>
    
    - Pas d'accès au système de fichiers local
    - Pas d'installation de packages (utilisez les packages pré-installés)
    - Les variables ne persistent pas entre les sessions (refresh de page)
    - Timeout d'exécution selon les limites Streamlit
    
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | PYTHON INTERACTIVE TERMINAL | PYTHON {sys.version.split()[0]}<br>
    SECURE EXECUTION • SANDBOXED ENVIRONMENT • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
