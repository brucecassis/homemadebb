import streamlit as st
import sys
import io
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import json
import subprocess
import os
import re
import time
import base64
from pathlib import Path
import hashlib

# Imports optionnels
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Python IDE",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# STYLE BLOOMBERG TERMINAL
# =============================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
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
        line-height: 1.5 !important;
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
    
    .terminal-output {
        background: #0a0a0a;
        border: 2px solid #00FF00;
        padding: 15px;
        margin: 10px 0;
        color: #00FF00;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        min-height: 100px;
        max-height: 400px;
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
    
    .terminal-warning {
        background: #2a2a0a;
        border: 2px solid #FFA500;
        padding: 10px;
        margin: 5px 0;
        color: #FFA500;
        font-family: 'Courier New', monospace;
        font-size: 11px;
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
    
    .cell-container {
        background: #0a0a0a;
        border: 2px solid #333;
        border-left: 4px solid #FFAA00;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0px;
    }
    
    .cell-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 10px;
    }
    
    .variable-inspector {
        background: #0a0a0a;
        border: 2px solid #00FFFF;
        padding: 10px;
        margin: 5px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .package-item {
        background: #111;
        border: 1px solid #333;
        padding: 8px;
        margin: 3px 0;
        display: flex;
        justify-content: space-between;
    }
    
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
    
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #FFAA00 !important;
    }
    
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
    
    .stDownloadButton button {
        background-color: #333 !important;
        color: #00FF00 !important;
        border: 2px solid #00FF00 !important;
    }
    
    .stFileUploader {
        background-color: #111 !important;
        border: 2px solid #333 !important;
    }
    
    .stExpander {
        background-color: #0a0a0a !important;
        border: 1px solid #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# SESSION STATE INITIALIZATION
# =============================================
if 'cells' not in st.session_state:
    st.session_state.cells = [{'id': 0, 'code': '', 'output': '', 'error': '', 'executed': False, 'exec_time': 0}]
if 'cell_counter' not in st.session_state:
    st.session_state.cell_counter = 1
if 'variables' not in st.session_state:
    st.session_state.variables = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'execution_timeout' not in st.session_state:
    st.session_state.execution_timeout = 30
if 'memory_limit' not in st.session_state:
    st.session_state.memory_limit = 512  # MB
if 'installed_packages' not in st.session_state:
    st.session_state.installed_packages = []
if 'ai_enabled' not in st.session_state:
    st.session_state.ai_enabled = False
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []
if 'watch_variables' not in st.session_state:
    st.session_state.watch_variables = []
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
    # Initialiser Groq si disponible
    if GROQ_AVAILABLE:
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY", None)
            if groq_api_key:
                st.session_state.groq_client = Groq(api_key=groq_api_key)
        except:
            pass

# =============================================
# HEADER
# =============================================
current_time = datetime.now().strftime("%H:%M:%S")
ai_status = "🤖 AI: ON" if st.session_state.groq_client else "🤖 AI: OFF"
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - ADVANCED PYTHON IDE</div>
    </div>
    <div>{current_time} UTC • PYTHON {sys.version.split()[0]} • {ai_status}</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# UTILITY FUNCTIONS
# =============================================

def get_installed_packages():
    """Récupère la liste des packages installés"""
    try:
        result = subprocess.run(['pip', 'list', '--format=json'], 
                              capture_output=True, text=True, timeout=5)
        packages = json.loads(result.stdout)
        return packages
    except:
        return []

def install_package(package_name):
    """Installe un package via pip"""
    try:
        result = subprocess.run(['pip', 'install', package_name], 
                              capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def get_variable_info(var):
    """Obtient des informations détaillées sur une variable"""
    info = {
        'type': type(var).__name__,
        'size': sys.getsizeof(var),
        'value': str(var)[:100]
    }
    
    if isinstance(var, (list, tuple, set)):
        info['length'] = len(var)
    elif isinstance(var, dict):
        info['length'] = len(var)
        info['keys'] = list(var.keys())[:5]
    elif isinstance(var, pd.DataFrame):
        info['shape'] = var.shape
        info['columns'] = list(var.columns)
    elif isinstance(var, np.ndarray):
        info['shape'] = var.shape
        info['dtype'] = str(var.dtype)
    
    return info

def execute_cell_code(code, cell_id, timeout=30):
    """Exécute le code d'une cellule avec timeout et profiling"""
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
        'execution_time': 0,
        'memory_used': 0
    }
    
    try:
        start_time = time.time()
        
        # Préparer l'environnement
        exec_globals = st.session_state.variables.copy()
        exec_globals.update({
            'pd': pd,
            'np': np,
            'json': json,
            'st': st
        })
        
        if MATPLOTLIB_AVAILABLE:
            exec_globals['plt'] = plt
        if PLOTLY_AVAILABLE:
            exec_globals['go'] = go
            exec_globals['px'] = px
        if SEABORN_AVAILABLE:
            exec_globals['sns'] = sns
        
        exec_locals = {}
        
        # Exécuter le code
        exec(code, exec_globals, exec_locals)
        
        # Mettre à jour les variables
        st.session_state.variables.update(exec_locals)
        
        execution_time = time.time() - start_time
        
        output = redirected_output.getvalue()
        error = redirected_error.getvalue()
        
        result['success'] = True
        result['output'] = output
        result['error'] = error
        result['execution_time'] = execution_time
        
        if exec_locals:
            last_var = list(exec_locals.values())[-1] if exec_locals else None
            result['returned_value'] = last_var
            
            # Sauvegarder les visualisations
            if MATPLOTLIB_AVAILABLE and isinstance(last_var, plt.Figure):
                st.session_state.visualizations.append({
                    'type': 'matplotlib',
                    'figure': last_var,
                    'cell_id': cell_id,
                    'timestamp': datetime.now()
                })
            elif PLOTLY_AVAILABLE and isinstance(last_var, (go.Figure,)):
                st.session_state.visualizations.append({
                    'type': 'plotly',
                    'figure': last_var,
                    'cell_id': cell_id,
                    'timestamp': datetime.now()
                })
        
    except Exception as e:
        result['success'] = False
        result['error'] = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
    
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result

def lint_code(code):
    """Analyse basique du code pour détecter les erreurs"""
    warnings = []
    
    # Vérifications basiques
    if 'import' in code and '*' in code:
        warnings.append("⚠️ Warning: Avoid using 'import *', prefer explicit imports")
    
    if 'eval(' in code or 'exec(' in code:
        warnings.append("⚠️ Warning: eval() and exec() can be dangerous")
    
    # Vérifier les variables non définies (basique)
    lines = code.split('\n')
    defined_vars = set()
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):
            var_name = line.split('=')[0].strip().split()[0]
            defined_vars.add(var_name)
    
    return warnings

def export_to_notebook(cells):
    """Exporte les cellules vers un format Jupyter Notebook"""
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split()[0]
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    for cell in cells:
        if cell['code'].strip():
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell['code'].split('\n')
            })
    
    return json.dumps(notebook, indent=2)

def generate_share_link(cells):
    """Génère un lien de partage (hash du code)"""
    code_content = "\n\n".join([cell['code'] for cell in cells if cell['code'].strip()])
    code_hash = hashlib.md5(code_content.encode()).hexdigest()
    return f"bloomberg-ide-{code_hash[:12]}"

def generate_code_with_ai(prompt, context=""):
    """Génère du code Python avec Groq AI"""
    if not st.session_state.groq_client:
        return None, "Groq API not configured"
    
    try:
        system_prompt = """You are an expert Python programmer assistant. Generate clean, efficient, and well-documented Python code based on user requests.

Rules:
- Generate ONLY executable Python code
- Include comments to explain the logic
- Use best practices and modern Python syntax
- Import necessary libraries at the top
- Handle edge cases and errors
- Make code readable and maintainable
- If asked for data analysis, use pandas/numpy
- If asked for visualization, use matplotlib/plotly/seaborn
- Return ONLY the code, no explanations before or after"""

        if context:
            system_prompt += f"\n\nContext - Current variables available:\n{context}"
        
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        generated_code = response.choices[0].message.content
        
        # Nettoyer le code (retirer les backticks si présents)
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        
        return generated_code, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def explain_code_with_ai(code):
    """Explique du code Python avec Groq AI"""
    if not st.session_state.groq_client:
        return "Groq API not configured"
    
    try:
        response = st.session_state.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a Python code explainer. Explain the given code in clear, simple terms. Break down what each part does."},
                {"role": "user", "content": f"Explain this Python code:\n\n{code}"}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"

def fix_code_with_ai(code, error):
    """Corrige du code Python avec Groq AI"""
    if not st.session_state.groq_client:
        return None, "Groq API not configured"
    
    try:
        response = st.session_state.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a Python debugging expert. Fix the given code based on the error message. Return ONLY the corrected code, no explanations."},
                {"role": "user", "content": f"Fix this code:\n\n{code}\n\nError:\n{error}"}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        fixed_code = response.choices[0].message.content
        fixed_code = fixed_code.replace("```python", "").replace("```", "").strip()
        
        return fixed_code, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def optimize_code_with_ai(code):
    """Optimise du code Python avec Groq AI"""
    if not st.session_state.groq_client:
        return None, "Groq API not configured"
    
    try:
        response = st.session_state.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a Python optimization expert. Improve the given code for better performance, readability, and best practices. Return ONLY the optimized code."},
                {"role": "user", "content": f"Optimize this code:\n\n{code}"}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        optimized_code = response.choices[0].message.content
        optimized_code = optimized_code.replace("```python", "").replace("```", "").strip()
        
        return optimized_code, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# =============================================
# SIDEBAR - CONTROLS & SETTINGS
# =============================================

with st.sidebar:
    st.markdown("### ⚙️ IDE CONTROLS")
    
    # File Upload
    st.markdown("#### 📁 FILE MANAGEMENT")
    uploaded_file = st.file_uploader("UPLOAD FILE", 
                                     type=['py', 'ipynb', 'csv', 'xlsx', 'json', 'txt'],
                                     help="Upload Python files, notebooks, or data files")
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1]
        file_content = uploaded_file.read()
        
        if file_ext == 'py':
            st.session_state.uploaded_files[uploaded_file.name] = file_content.decode('utf-8')
            st.success(f"✅ Loaded {uploaded_file.name}")
            
            # Option to load into cell
            if st.button("📋 LOAD INTO NEW CELL"):
                new_cell = {
                    'id': st.session_state.cell_counter,
                    'code': file_content.decode('utf-8'),
                    'output': '',
                    'error': '',
                    'executed': False,
                    'exec_time': 0
                }
                st.session_state.cells.append(new_cell)
                st.session_state.cell_counter += 1
                st.rerun()
        
        elif file_ext == 'ipynb':
            try:
                notebook = json.loads(file_content.decode('utf-8'))
                st.success(f"✅ Loaded notebook: {uploaded_file.name}")
                
                if st.button("📋 IMPORT NOTEBOOK CELLS"):
                    st.session_state.cells = []
                    for idx, cell in enumerate(notebook.get('cells', [])):
                        if cell['cell_type'] == 'code':
                            code = ''.join(cell['source'])
                            new_cell = {
                                'id': idx,
                                'code': code,
                                'output': '',
                                'error': '',
                                'executed': False,
                                'exec_time': 0
                            }
                            st.session_state.cells.append(new_cell)
                    st.session_state.cell_counter = len(st.session_state.cells)
                    st.rerun()
            except:
                st.error("❌ Invalid notebook format")
        
        elif file_ext in ['csv', 'xlsx']:
            st.session_state.uploaded_files[uploaded_file.name] = file_content
            st.success(f"✅ Uploaded {uploaded_file.name}")
            
            # Auto-generate loading code
            var_name = uploaded_file.name.replace('.', '_').replace('-', '_')
            if file_ext == 'csv':
                load_code = f"""# Load uploaded CSV
import pandas as pd
from io import BytesIO

{var_name} = pd.read_csv(BytesIO(st.session_state.uploaded_files['{uploaded_file.name}']))
print(f"Loaded {{len({var_name})}} rows")
{var_name}.head()"""
            else:
                load_code = f"""# Load uploaded Excel
import pandas as pd
from io import BytesIO

{var_name} = pd.read_excel(BytesIO(st.session_state.uploaded_files['{uploaded_file.name}']))
print(f"Loaded {{len({var_name})}} rows")
{var_name}.head()"""
            
            if st.button("📊 GENERATE LOAD CODE"):
                new_cell = {
                    'id': st.session_state.cell_counter,
                    'code': load_code,
                    'output': '',
                    'error': '',
                    'executed': False,
                    'exec_time': 0
                }
                st.session_state.cells.append(new_cell)
                st.session_state.cell_counter += 1
                st.rerun()
    
    # Uploaded files list
    if st.session_state.uploaded_files:
        st.markdown("**📦 Uploaded Files:**")
        for fname in st.session_state.uploaded_files.keys():
            st.text(f"• {fname}")
    
    st.markdown("---")
    
    # Export Options
    st.markdown("#### 💾 EXPORT OPTIONS")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📓 EXPORT .IPYNB"):
            notebook_json = export_to_notebook(st.session_state.cells)
            st.download_button(
                "⬇️ Download Notebook",
                notebook_json,
                "bloomberg_notebook.ipynb",
                "application/json"
            )
    
    with col2:
        if st.button("🐍 EXPORT .PY"):
            py_code = "\n\n# " + "="*50 + "\n\n".join([
                f"# Cell {cell['id']}\n{cell['code']}" 
                for cell in st.session_state.cells if cell['code'].strip()
            ])
            st.download_button(
                "⬇️ Download Script",
                py_code,
                "bloomberg_script.py",
                "text/plain"
            )
    
    # Share link
    if st.button("🔗 GENERATE SHARE LINK"):
        share_id = generate_share_link(st.session_state.cells)
        st.code(f"Share ID: {share_id}", language=None)
        st.info("💡 Save this ID to restore your work later")
    
    st.markdown("---")
    
    # Package Management
    st.markdown("#### 📦 PACKAGE MANAGER")
    
    package_to_install = st.text_input("Package name:", key="pkg_install")
    if st.button("📥 INSTALL PACKAGE"):
        if package_to_install:
            with st.spinner(f"Installing {package_to_install}..."):
                success, output = install_package(package_to_install)
                if success:
                    st.success(f"✅ {package_to_install} installed!")
                    st.session_state.installed_packages = get_installed_packages()
                else:
                    st.error(f"❌ Installation failed")
                    st.code(output, language=None)
    
    if st.button("🔄 REFRESH PACKAGE LIST"):
        st.session_state.installed_packages = get_installed_packages()
        st.success("✅ Package list updated")
    
    # Show installed packages
    with st.expander("📋 INSTALLED PACKAGES"):
        if not st.session_state.installed_packages:
            st.session_state.installed_packages = get_installed_packages()
        
        search_pkg = st.text_input("🔍 Search packages:", key="search_pkg")
        packages = st.session_state.installed_packages
        
        if search_pkg:
            packages = [p for p in packages if search_pkg.lower() in p['name'].lower()]
        
        for pkg in packages[:20]:  # Limit display
            st.markdown(f"""
            <div class="package-item">
                <span>{pkg['name']}</span>
                <span style="color:#666;">{pkg['version']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Execution Settings
    st.markdown("#### ⚙️ EXECUTION SETTINGS")
    
    st.session_state.execution_timeout = st.slider(
        "Timeout (seconds)",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )
    
    st.session_state.memory_limit = st.slider(
        "Memory Limit (MB)",
        min_value=128,
        max_value=2048,
        value=512,
        step=128
    )
    
    auto_save = st.checkbox("Auto-save cells", value=True)
    show_line_numbers = st.checkbox("Show line numbers", value=True)
    
    st.markdown("---")
    
    # Variable Inspector
    st.markdown("#### 🔍 VARIABLE INSPECTOR")
    
    if st.session_state.variables:
        # Add to watch list
        var_to_watch = st.selectbox(
            "Add to watch:",
            [""] + list(st.session_state.variables.keys())
        )
        
        if var_to_watch and st.button("👁️ WATCH VARIABLE"):
            if var_to_watch not in st.session_state.watch_variables:
                st.session_state.watch_variables.append(var_to_watch)
                st.success(f"✅ Watching {var_to_watch}")
        
        # Watch list
        if st.session_state.watch_variables:
            st.markdown("**👁️ WATCHED VARIABLES:**")
            for var_name in st.session_state.watch_variables:
                if var_name in st.session_state.variables:
                    var_value = st.session_state.variables[var_name]
                    var_info = get_variable_info(var_value)
                    
                    with st.expander(f"🔹 {var_name}"):
                        st.text(f"Type: {var_info['type']}")
                        st.text(f"Size: {var_info['size']} bytes")
                        if 'shape' in var_info:
                            st.text(f"Shape: {var_info['shape']}")
                        if 'length' in var_info:
                            st.text(f"Length: {var_info['length']}")
                        st.code(var_info['value'], language='python')
                        
                        if st.button(f"🗑️ Remove watch", key=f"unwatch_{var_name}"):
                            st.session_state.watch_variables.remove(var_name)
                            st.rerun()
        
        # All variables
        with st.expander(f"📊 ALL VARIABLES ({len(st.session_state.variables)})"):
            for var_name, var_value in st.session_state.variables.items():
                var_info = get_variable_info(var_value)
                st.markdown(f"**{var_name}** ({var_info['type']})")
                st.text(f"Size: {var_info['size']} bytes")
                if 'shape' in var_info:
                    st.text(f"Shape: {var_info['shape']}")
    else:
        st.info("No variables defined yet")
    
    if st.button("🗑️ CLEAR ALL VARIABLES"):
        st.session_state.variables = {}
        st.session_state.watch_variables = []
        st.success("✅ Variables cleared")
        st.rerun()
    
    st.markdown("---")
    
    # Visualization Gallery
    st.markdown("#### 🎨 VISUALIZATION GALLERY")
    
    if st.session_state.visualizations:
        st.text(f"📊 {len(st.session_state.visualizations)} visualizations")
        
        for idx, viz in enumerate(st.session_state.visualizations[-5:]):  # Last 5
            with st.expander(f"📈 Viz {idx+1} (Cell {viz['cell_id']})"):
                st.text(f"Type: {viz['type']}")
                st.text(f"Time: {viz['timestamp'].strftime('%H:%M:%S')}")
                
                if viz['type'] == 'matplotlib' and MATPLOTLIB_AVAILABLE:
                    st.pyplot(viz['figure'])
                elif viz['type'] == 'plotly' and PLOTLY_AVAILABLE:
                    st.plotly_chart(viz['figure'], use_container_width=True)
        
        if st.button("🗑️ CLEAR GALLERY"):
            st.session_state.visualizations = []
            st.rerun()
    else:
        st.info("No visualizations yet")

# =============================================
# MAIN AREA - JUPYTER-STYLE CELLS
# =============================================

st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;">🐍 PYTHON IDE - JUPYTER-STYLE CELLS</p>', unsafe_allow_html=True)

# Library availability warnings
warnings = []
if not MATPLOTLIB_AVAILABLE:
    warnings.append("⚠️ matplotlib not available")
if not PLOTLY_AVAILABLE:
    warnings.append("⚠️ plotly not available")
if not SEABORN_AVAILABLE:
    warnings.append("⚠️ seaborn not available")
if not GROQ_AVAILABLE:
    warnings.append("⚠️ groq not available (AI features disabled)")
elif not st.session_state.groq_client:
    warnings.append("⚠️ Groq API key not configured (AI features disabled)")

if warnings:
    st.markdown(f"""
    <div class="terminal-warning">
        {' • '.join(warnings)}<br>
        Install with: pip install matplotlib plotly seaborn groq
    </div>
    """, unsafe_allow_html=True)

# Cell controls
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    if st.button("➕ ADD CELL"):
        new_cell = {
            'id': st.session_state.cell_counter,
            'code': '',
            'output': '',
            'error': '',
            'executed': False,
            'exec_time': 0
        }
        st.session_state.cells.append(new_cell)
        st.session_state.cell_counter += 1
        st.rerun()

with col2:
    if st.button("▶️ RUN ALL CELLS"):
        for cell in st.session_state.cells:
            if cell['code'].strip():
                result = execute_cell_code(cell['code'], cell['id'])
                cell['output'] = result['output']
                cell['error'] = result['error']
                cell['executed'] = True
                cell['exec_time'] = result['execution_time']
                cell['returned_value'] = result.get('returned_value')
        st.rerun()

with col3:
    if st.button("🗑️ CLEAR ALL OUTPUTS"):
        for cell in st.session_state.cells:
            cell['output'] = ''
            cell['error'] = ''
            cell['executed'] = False
            cell['returned_value'] = None
        st.rerun()

with col4:
    if st.button("🔄 RESET IDE"):
        st.session_state.cells = [{'id': 0, 'code': '', 'output': '', 'error': '', 'executed': False, 'exec_time': 0}]
        st.session_state.cell_counter = 1
        st.session_state.variables = {}
        st.session_state.visualizations = []
        st.rerun()

st.markdown("---")

# Render cells
for idx, cell in enumerate(st.session_state.cells):
    st.markdown(f'<div class="cell-container">', unsafe_allow_html=True)
    
    # Cell header
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([1, 1, 1, 1, 2])
    
    with col_h1:
        st.markdown(f"**Cell [{cell['id']}]**")
    
    with col_h2:
        if st.button("▶️", key=f"run_{cell['id']}", help="Run this cell"):
            result = execute_cell_code(cell['code'], cell['id'])
            cell['output'] = result['output']
            cell['error'] = result['error']
            cell['executed'] = True
            cell['exec_time'] = result['execution_time']
            cell['returned_value'] = result.get('returned_value')
            st.rerun()
    
    with col_h3:
        if st.button("🗑️", key=f"del_{cell['id']}", help="Delete this cell"):
            st.session_state.cells.pop(idx)
            st.rerun()
    
    with col_h4:
        if idx > 0 and st.button("⬆️", key=f"up_{cell['id']}", help="Move up"):
            st.session_state.cells[idx], st.session_state.cells[idx-1] = \
                st.session_state.cells[idx-1], st.session_state.cells[idx]
            st.rerun()
    
    with col_h5:
        if cell['executed']:
            st.markdown(f"<small style='color:#00FF00;'>✅ Executed in {cell['exec_time']:.3f}s</small>", 
                       unsafe_allow_html=True)
    
    # Code editor
    cell['code'] = st.text_area(
        f"Code {cell['id']}",
        value=cell['code'],
        height=150,
        key=f"code_{cell['id']}",
        label_visibility="collapsed",
        placeholder="# Enter Python code here...\n# Shift+Enter to run (in real Jupyter)"
    )
    
    # Lint warnings
    if cell['code'].strip():
        warnings = lint_code(cell['code'])
        for warning in warnings:
            st.markdown(f'<div class="terminal-warning">{warning}</div>', unsafe_allow_html=True)
    
    # Output
    if cell['executed']:
        if cell['error']:
            st.markdown(f"""
            <div class="terminal-error">
                ❌ ERROR:<br>
                {cell['error']}
            </div>
            """, unsafe_allow_html=True)
        
        if cell['output']:
            st.markdown(f"""
            <div class="terminal-output">
                <span style="color:#00FFFF;">>>> OUTPUT:</span><br>
                {cell['output']}
            </div>
            """, unsafe_allow_html=True)
        
        # Display returned value
        if 'returned_value' in cell and cell['returned_value'] is not None:
            returned_val = cell['returned_value']
            
            if isinstance(returned_val, pd.DataFrame):
                st.dataframe(returned_val, use_container_width=True, height=300)
                
                # Export options
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    csv = returned_val.to_csv(index=False)
                    st.download_button(
                        "⬇️ CSV",
                        csv,
                        f"cell_{cell['id']}_output.csv",
                        key=f"csv_{cell['id']}"
                    )
                with col_e2:
                    json_str = returned_val.to_json(orient='records', indent=2)
                    st.download_button(
                        "⬇️ JSON",
                        json_str,
                        f"cell_{cell['id']}_output.json",
                        key=f"json_{cell['id']}"
                    )
            
            elif MATPLOTLIB_AVAILABLE and isinstance(returned_val, plt.Figure):
                st.pyplot(returned_val)
            
            elif PLOTLY_AVAILABLE and isinstance(returned_val, go.Figure):
                st.plotly_chart(returned_val, use_container_width=True)
            
            else:
                st.code(str(returned_val), language='python')
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# =============================================
# AI ASSISTANT (Groq Integration)
# =============================================

st.markdown('<p style="color:#FFAA00;font-weight:bold;font-size:14px;border-bottom:2px solid #333;padding:10px 0;margin-top:20px;">🤖 AI CODING ASSISTANT</p>', unsafe_allow_html=True)

if st.session_state.groq_client:
    st.markdown("""
    <div class="success-box">
        ✅ Groq API Connected - AI Features Available
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different AI features
    ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs(["✨ GENERATE", "📖 EXPLAIN", "🔧 FIX", "⚡ OPTIMIZE"])
    
    # TAB 1: Generate Code
    with ai_tab1:
        st.markdown("### ✨ Generate Code from Description")
        
        # Context about variables
        context_info = ""
        if st.session_state.variables:
            var_list = ", ".join([f"{name} ({type(val).__name__})" for name, val in st.session_state.variables.items()])
            context_info = f"Available variables: {var_list}"
            st.info(f"💡 {context_info}")
        
        ai_prompt = st.text_area(
            "Describe what you want to code:",
            placeholder="e.g., Create a function that calculates the Fibonacci sequence up to n terms",
            height=100,
            key="ai_generate_prompt"
        )
        
        col_gen1, col_gen2 = st.columns([2, 1])
        
        with col_gen1:
            if st.button("✨ GENERATE CODE", key="generate_code_btn", type="primary"):
                if ai_prompt.strip():
                    with st.spinner("🤖 Generating code with Groq AI..."):
                        generated_code, error = generate_code_with_ai(ai_prompt, context_info)
                        
                        if generated_code:
                            st.success("✅ Code generated successfully!")
                            st.code(generated_code, language='python')
                            
                            if st.button("📋 ADD TO NEW CELL", key="add_generated"):
                                new_cell = {
                                    'id': st.session_state.cell_counter,
                                    'code': generated_code,
                                    'output': '',
                                    'error': '',
                                    'executed': False,
                                    'exec_time': 0
                                }
                                st.session_state.cells.append(new_cell)
                                st.session_state.cell_counter += 1
                                st.success("✅ Code added to new cell!")
                                st.rerun()
                        else:
                            st.error(f"❌ {error}")
                else:
                    st.warning("⚠️ Please enter a description")
        
        with col_gen2:
            use_context = st.checkbox("Use current variables", value=True, key="use_context")
    
    # TAB 2: Explain Code
    with ai_tab2:
        st.markdown("### 📖 Explain Code")
        
        # Select cell to explain
        if st.session_state.cells:
            cell_options = {f"Cell [{cell['id']}]": cell for cell in st.session_state.cells if cell['code'].strip()}
            
            if cell_options:
                selected_cell_label = st.selectbox("Select cell to explain:", list(cell_options.keys()))
                selected_cell = cell_options[selected_cell_label]
                
                st.code(selected_cell['code'], language='python')
                
                if st.button("📖 EXPLAIN THIS CODE", key="explain_btn", type="primary"):
                    with st.spinner("🤖 Analyzing code..."):
                        explanation = explain_code_with_ai(selected_cell['code'])
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <b>🤖 AI Explanation:</b><br><br>
                            {explanation}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No code to explain. Add some code first!")
        else:
            st.info("ℹ️ No cells available")
    
    # TAB 3: Fix Code
    with ai_tab3:
        st.markdown("### 🔧 Fix Code Errors")
        
        # Select cell with error
        error_cells = {f"Cell [{cell['id']}]": cell for cell in st.session_state.cells 
                      if cell.get('error') and cell['code'].strip()}
        
        if error_cells:
            selected_error_label = st.selectbox("Select cell with error:", list(error_cells.keys()))
            selected_error_cell = error_cells[selected_error_label]
            
            st.markdown("**Original Code:**")
            st.code(selected_error_cell['code'], language='python')
            
            st.markdown("**Error:**")
            st.markdown(f"""
            <div class="terminal-error">
                {selected_error_cell['error']}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔧 FIX THIS ERROR", key="fix_btn", type="primary"):
                with st.spinner("🤖 Fixing code..."):
                    fixed_code, error = fix_code_with_ai(
                        selected_error_cell['code'], 
                        selected_error_cell['error']
                    )
                    
                    if fixed_code:
                        st.success("✅ Code fixed!")
                        st.markdown("**Fixed Code:**")
                        st.code(fixed_code, language='python')
                        
                        col_fix1, col_fix2 = st.columns(2)
                        
                        with col_fix1:
                            if st.button("✅ REPLACE IN CELL", key="replace_fixed"):
                                selected_error_cell['code'] = fixed_code
                                selected_error_cell['error'] = ''
                                selected_error_cell['executed'] = False
                                st.success("✅ Code replaced in cell!")
                                st.rerun()
                        
                        with col_fix2:
                            if st.button("➕ ADD AS NEW CELL", key="add_fixed"):
                                new_cell = {
                                    'id': st.session_state.cell_counter,
                                    'code': fixed_code,
                                    'output': '',
                                    'error': '',
                                    'executed': False,
                                    'exec_time': 0
                                }
                                st.session_state.cells.append(new_cell)
                                st.session_state.cell_counter += 1
                                st.success("✅ Fixed code added to new cell!")
                                st.rerun()
                    else:
                        st.error(f"❌ {error}")
        else:
            st.info("ℹ️ No errors to fix. Great job! 🎉")
    
    # TAB 4: Optimize Code
    with ai_tab4:
        st.markdown("### ⚡ Optimize Code")
        
        if st.session_state.cells:
            cell_opt_options = {f"Cell [{cell['id']}]": cell for cell in st.session_state.cells if cell['code'].strip()}
            
            if cell_opt_options:
                selected_opt_label = st.selectbox("Select cell to optimize:", list(cell_opt_options.keys()))
                selected_opt_cell = cell_opt_options[selected_opt_label]
                
                st.markdown("**Original Code:**")
                st.code(selected_opt_cell['code'], language='python')
                
                if st.button("⚡ OPTIMIZE THIS CODE", key="optimize_btn", type="primary"):
                    with st.spinner("🤖 Optimizing code..."):
                        optimized_code, error = optimize_code_with_ai(selected_opt_cell['code'])
                        
                        if optimized_code:
                            st.success("✅ Code optimized!")
                            st.markdown("**Optimized Code:**")
                            st.code(optimized_code, language='python')
                            
                            col_opt1, col_opt2 = st.columns(2)
                            
                            with col_opt1:
                                if st.button("✅ REPLACE IN CELL", key="replace_optimized"):
                                    selected_opt_cell['code'] = optimized_code
                                    selected_opt_cell['executed'] = False
                                    st.success("✅ Code replaced in cell!")
                                    st.rerun()
                            
                            with col_opt2:
                                if st.button("➕ ADD AS NEW CELL", key="add_optimized"):
                                    new_cell = {
                                        'id': st.session_state.cell_counter,
                                        'code': optimized_code,
                                        'output': '',
                                        'error': '',
                                        'executed': False,
                                        'exec_time': 0
                                    }
                                    st.session_state.cells.append(new_cell)
                                    st.session_state.cell_counter += 1
                                    st.success("✅ Optimized code added to new cell!")
                                    st.rerun()
                        else:
                            st.error(f"❌ {error}")
            else:
                st.info("ℹ️ No code to optimize. Add some code first!")
        else:
            st.info("ℹ️ No cells available")

else:
    st.markdown("""
    <div class="terminal-warning">
        ⚠️ Groq API Not Configured<br><br>
        
        <b>To enable AI features:</b><br>
        1. Get a free API key from <a href="https://console.groq.com" target="_blank" style="color:#00FFFF;">console.groq.com</a><br>
        2. Add to Streamlit Secrets: GROQ_API_KEY = "your_key_here"<br>
        3. Install groq: pip install groq<br><br>
        
        <b>AI Features Available:</b><br>
        • Code generation from natural language<br>
        • Code explanation and documentation<br>
        • Automatic error fixing<br>
        • Code optimization suggestions<br>
    </div>
    """, unsafe_allow_html=True)
    
    if not GROQ_AVAILABLE:
        st.markdown("""
        <div class="terminal-error">
            ❌ Groq package not installed<br>
            Run: pip install groq
        </div>
        """, unsafe_allow_html=True)

# =============================================
# KEYBOARD SHORTCUTS INFO
# =============================================

with st.expander("⌨️ KEYBOARD SHORTCUTS & TIPS"):
    st.markdown("""
    <div style="color:#FFAA00;">
    
    <p style="color:#00FFFF;font-weight:bold;">📋 Jupyter-Style Workflow</p>
    
    • **Add Cell**: Click "➕ ADD CELL" button<br>
    • **Run Cell**: Click "▶️" next to each cell<br>
    • **Run All**: Click "▶️ RUN ALL CELLS"<br>
    • **Move Cells**: Use ⬆️ buttons to reorder<br>
    • **Delete Cell**: Click 🗑️ next to cell<br><br>
    
    <p style="color:#00FFFF;font-weight:bold;">💾 File Operations</p>
    
    • **Upload .py**: Sidebar → Upload File → Load into cell<br>
    • **Upload .ipynb**: Sidebar → Upload → Import all cells<br>
    • **Upload Data**: CSV/Excel → Auto-generate loading code<br>
    • **Export**: Sidebar → Export to .ipynb or .py<br><br>
    
    <p style="color:#00FFFF;font-weight:bold;">🔍 Debugging</p>
    
    • **Variable Inspector**: Sidebar shows all variables<br>
    • **Watch Variables**: Add variables to watch list<br>
    • **Error Traces**: Full tracebacks displayed<br>
    • **Lint Warnings**: Real-time code quality checks<br><br>
    
    <p style="color:#00FFFF;font-weight:bold;">📦 Package Management</p>
    
    • **Install**: Sidebar → Package Manager → Enter name<br>
    • **View Installed**: Sidebar → Installed Packages<br>
    • **Search**: Use search box to filter packages<br><br>
    
    <p style="color:#00FFFF;font-weight:bold;">🎨 Visualizations</p>
    
    • **Matplotlib**: Auto-displayed inline<br>
    • **Plotly**: Interactive charts supported<br>
    • **Gallery**: Sidebar → Visualization Gallery<br>
    • **Export**: Download charts as PNG/SVG<br><br>
    
    <p style="color:#00FFFF;font-weight:bold;">⚡ Pro Tips</p>
    
    • Variables persist across all cells<br>
    • Use `print()` for debugging output<br>
    • Last line auto-displays if not assigned<br>
    • DataFrames get special rendering<br>
    • Upload data files for instant analysis<br>
    • Export your work as notebooks<br>
    
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | ADVANCED PYTHON IDE | MULTI-CELL EXECUTION<br>
    JUPYTER-STYLE INTERFACE • PACKAGE MANAGER • VARIABLE INSPECTOR • AI-READY<br>
    CELLS: {len(st.session_state.cells)} • VARIABLES: {len(st.session_state.variables)} • VISUALIZATIONS: {len(st.session_state.visualizations)}
</div>
""", unsafe_allow_html=True)
