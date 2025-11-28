import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from supabase import create_client, Client

# ============================================================================
# PARAMETRES SUPABASE
# ============================================================================

SUPABASE_URL = "https://gbrefcefeavmqupulzyw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdicmVmY2VmZWF2bXF1cHVsenl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0OTA2NjksImV4cCI6MjA3OTA2NjY2OX0.WsA-3so0J52hAyZTIddVT0qqLuvcxjHYTZ4XkZ5mMio"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✓ Connexion à Supabase réussie")

# ============================================================================
# RECUPERATION DES DONNEES
# ============================================================================

TABLE_NAME = "morgan_stanley_h4_data"

print(f"\n📊 Récupération des données de '{TABLE_NAME}'...")

# Récupérer toutes les données (limitées à 10000 pour éviter les problèmes)
response = supabase.table(TABLE_NAME).select("*").order("date", desc=False).limit(10000).execute()

if not response.data:
    print("❌ Aucune donnée trouvée dans la table!")
    exit()

# Convertir en DataFrame
df = pd.DataFrame(response.data)

print(f"✓ {len(df)} lignes récupérées")
print(f"📅 Période: {df['date'].min()} à {df['date'].max()}")

# ============================================================================
# PREPARATION DES DONNEES
# ============================================================================

# Convertir la date en datetime et définir comme index
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# S'assurer que les colonnes sont en float
for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprimer les lignes avec des NaN
df = df.dropna(subset=['open', 'high', 'low', 'close'])

print(f"✓ Données nettoyées: {len(df)} bougies valides")

# ============================================================================
# CHOIX DU TYPE DE GRAPHIQUE
# ============================================================================

print("\n" + "="*60)
print("CHOISIR LE TYPE DE GRAPHIQUE")
print("="*60)
print("1 - Courbe (ligne de prix de clôture)")
print("2 - Bougies japonaises (candlestick)")
print("="*60)

choix = input("Ton choix (1 ou 2): ").strip()

# ============================================================================
# TRACAGE DU GRAPHIQUE
# ============================================================================

if choix == "1":
    # COURBE
    print("\n📈 Génération du graphique en courbe...")
    
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['close'], linewidth=1.5, color='#2962FF')
    plt.title(f'Morgan Stanley (MS) - Prix de clôture H4', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Prix ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    filename = "morgan_stanley_courbe.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {filename}")
    
    plt.show()

elif choix == "2":
    # BOUGIES JAPONAISES
    print("\n🕯️ Génération du graphique en bougies...")
    
    # Style personnalisé
    mc = mpf.make_marketcolors(
        up='#26a69a',      # Vert pour hausse
        down='#ef5350',    # Rouge pour baisse
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='lightgray',
        facecolor='white',
        figcolor='white'
    )
    
    # Tracer les bougies avec volume
    mpf.plot(
        df,
        type='candle',
        style=s,
        title=f'Morgan Stanley (MS) - Bougies H4',
        ylabel='Prix ($)',
        volume=True,
        ylabel_lower='Volume',
        figsize=(15, 8),
        tight_layout=True,
        savefig='morgan_stanley_bougies.png'
    )
    
    print(f"✅ Graphique sauvegardé: morgan_stanley_bougies.png")

else:
    print("❌ Choix invalide!")

print("\n✅ Terminé!")
