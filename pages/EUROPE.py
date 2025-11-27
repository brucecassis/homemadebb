import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Europe Economic Data",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Bloomberg style
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    .bloomberg-header {
        background: #FFAA00;
        padding: 5px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 14px;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        border-bottom: 2px solid #FFAA00;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
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
    
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFAA00 !important;
        font-size: 10px !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button {
        background-color: #333;
        color: #FFAA00;
        font-weight: bold;
        border: 1px solid #FFAA00;
        padding: 6px 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0px;
        font-size: 10px;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000;
    }
    
    .api-info-box {
        background-color: #0a0a0a;
        border-left: 3px solid #00FFFF;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .country-flag {
        font-size: 24px;
        margin-right: 10px;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - EUROPEAN ECONOMIC DATA</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# Fonctions API ECB
@st.cache_data(ttl=3600)
def get_ecb_data(series_key, start_date=None):
    """Récupère données ECB Statistical Data Warehouse"""
    try:
        base_url = "https://sdw-wsrest.ecb.europa.eu/service/data"
        
        params = {
            'format': 'jsondata',
            'detail': 'dataonly'
        }
        
        if start_date:
            params['startPeriod'] = start_date
        
        url = f"{base_url}/{series_key}"
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'dataSets' in data and len(data['dataSets']) > 0:
                observations = data['dataSets'][0].get('series', {})
                
                if observations:
                    series_data = list(observations.values())[0].get('observations', {})
                    
                    dates = []
                    values = []
                    
                    for key, value in series_data.items():
                        dates.append(data['structure']['dimensions']['observation'][0]['values'][int(key)]['id'])
                        values.append(value[0])
                    
                    df = pd.DataFrame({
                        'date': pd.to_datetime(dates),
                        'value': values
                    })
                    
                    return df.sort_values('date')
        
        return None
        
    except Exception as e:
        st.error(f"ECB API Error: {e}")
        return None

# Configuration des pays
COUNTRIES_CONFIG = {
    'france': {
        'name': 'France',
        'flag': '🇫🇷',
        'indicators': {
            'gdp_growth': ('0.4%', '+0.1%', 'QoQ'),
            'inflation': ('2.9%', '+0.3%', 'YoY'),
            'unemployment': ('7.3%', '-0.2%', 'ILO'),
            'trade_balance': ('-€8.2B', '-€1.1B', 'Monthly')
        },
        'statistical_office': {
            'name': 'INSEE',
            'full_name': 'Institut national de la statistique et des études économiques',
            'website': 'https://www.insee.fr/',
            'api_url': 'https://api.insee.fr/catalogue/',
            'requires_auth': True,
            'key_series': {
                '001688527': 'PIB Trimestriel (GDP)',
                '001759970': 'IPC (CPI)',
                '001688613': 'Taux de Chômage (Unemployment)',
                '001769682': 'Production Industrielle (Industrial Production)',
                '001641151': 'Balance Commerciale (Trade Balance)'
            }
        },
        'central_bank': 'Banque de France',
        'stock_index': {'name': 'CAC 40', 'symbol': '^FCHI', 'value': '7,521', 'change': '+1.2%'},
        'bond_10y': {'name': 'OAT 10Y', 'value': '3.12%', 'change': '+0.05%'},
        'sectors': {
            'Industry': {'Weight': '13.5%', 'Growth': '+0.3%', 'Employment': '2.8M'},
            'Construction': {'Weight': '5.8%', 'Growth': '-0.1%', 'Employment': '1.5M'},
            'Services': {'Weight': '78.9%', 'Growth': '+0.5%', 'Employment': '18.2M'},
            'Agriculture': {'Weight': '1.8%', 'Growth': '-0.2%', 'Employment': '0.7M'}
        }
    },
    'switzerland': {
        'name': 'Switzerland',
        'flag': '🇨🇭',
        'indicators': {
            'gdp_growth': ('0.3%', '+0.1%', 'QoQ'),
            'inflation': ('1.4%', '-0.2%', 'YoY'),
            'unemployment': ('2.0%', '+0.1%', 'SECO'),
            'kof_barometer': ('101.2', '+1.8', 'Leading Indicator')
        },
        'statistical_office': {
            'name': 'BFS/FSO',
            'full_name': 'Federal Statistical Office / Office fédéral de la statistique',
            'website': 'https://www.bfs.admin.ch/',
            'api_url': 'https://opendata.swiss/',
            'requires_auth': False,
            'key_datasets': {
                'je-e-06.02.01.01': 'Unemployment Rate (SECO)',
                'px-x-0602010000_101': 'GDP Quarterly',
                'cc-e-05.02.55': 'Consumer Price Index',
                'je-e-06.02.02.07': 'Job Vacancies',
                'su-e-11.03.03': 'Production Index'
            }
        },
        'central_bank': 'SNB - Swiss National Bank',
        'stock_index': {'name': 'SMI', 'symbol': '^SSMI', 'value': '11,234', 'change': '+0.8%'},
        'bond_10y': {'name': 'Swiss Govt 10Y', 'value': '0.85%', 'change': '+0.03%'},
        'special_indicators': {
            'SNB Policy Rate': '1.75%',
            'CHF/EUR': '0.93',
            'Foreign Reserves': 'CHF 716B',
            'KOF Barometer': '101.2'
        }
    },
    'italy': {
        'name': 'Italy',
        'flag': '🇮🇹',
        'indicators': {
            'gdp_growth': ('0.2%', '-0.1%', 'QoQ'),
            'inflation': ('1.9%', '-0.3%', 'YoY'),
            'unemployment': ('7.8%', '-0.2%', 'ISTAT'),
            'trade_balance': ('+€5.2B', '+€0.8B', 'Monthly')
        },
        'statistical_office': {
            'name': 'ISTAT',
            'full_name': 'Istituto Nazionale di Statistica',
            'website': 'https://www.istat.it/',
            'api_url': 'https://www.istat.it/en/data/',
            'requires_auth': False,
            'key_series': {
                'DCSC_PIL1': 'PIL Trimestrale (GDP)',
                'DCSC_INFLAZIONE': 'Indice dei prezzi al consumo (CPI)',
                'DCCV_TAXDISOCCU1': 'Tasso di disoccupazione (Unemployment)',
                'DCSC_PRODIND': 'Produzione industriale',
                'DCSC_COMMESTERO': 'Commercio estero (Trade)'
            }
        },
        'central_bank': 'Banca d\'Italia',
        'stock_index': {'name': 'FTSE MIB', 'symbol': 'FTSEMIB.MI', 'value': '34,567', 'change': '+0.6%'},
        'bond_10y': {'name': 'BTP 10Y', 'value': '4.15%', 'change': '+0.08%'},
        'sectors': {
            'Industry': {'Weight': '16.3%', 'Growth': '+0.2%', 'Employment': '4.2M'},
            'Construction': {'Weight': '4.9%', 'Growth': '+0.4%', 'Employment': '1.4M'},
            'Services': {'Weight': '73.9%', 'Growth': '+0.3%', 'Employment': '16.8M'},
            'Agriculture': {'Weight': '2.1%', 'Growth': '-0.1%', 'Employment': '0.9M'}
        }
    },
    'germany': {
        'name': 'Germany',
        'flag': '🇩🇪',
        'indicators': {
            'gdp_growth': ('0.1%', '-0.2%', 'QoQ'),
            'inflation': ('2.7%', '+0.4%', 'YoY'),
            'unemployment': ('3.0%', '+0.1%', 'BA'),
            'trade_balance': ('+€20.4B', '+€2.1B', 'Monthly')
        },
        'statistical_office': {
            'name': 'Destatis',
            'full_name': 'Statistisches Bundesamt',
            'website': 'https://www.destatis.de/',
            'api_url': 'https://www-genesis.destatis.de/genesis/online',
            'requires_auth': True,
            'key_series': {
                '81000-0001': 'BIP (GDP)',
                '61111-0001': 'Verbraucherpreisindex (CPI)',
                '13231-0001': 'Arbeitslosenquote (Unemployment)',
                '42151-0001': 'Produktion im Produzierenden Gewerbe',
                '51000-0001': 'Außenhandel (Foreign Trade)'
            }
        },
        'central_bank': 'Deutsche Bundesbank',
        'stock_index': {'name': 'DAX', 'symbol': '^GDAXI', 'value': '17,892', 'change': '+1.1%'},
        'bond_10y': {'name': 'Bund 10Y', 'value': '2.65%', 'change': '+0.04%'},
        'sectors': {
            'Industry': {'Weight': '23.8%', 'Growth': '-0.1%', 'Employment': '8.1M'},
            'Construction': {'Weight': '6.2%', 'Growth': '-0.3%', 'Employment': '2.6M'},
            'Services': {'Weight': '68.6%', 'Growth': '+0.3%', 'Employment': '32.5M'},
            'Agriculture': {'Weight': '0.7%', 'Growth': '+0.1%', 'Employment': '0.6M'}
        }
    },
    'netherlands': {
        'name': 'Netherlands',
        'flag': '🇳🇱',
        'indicators': {
            'gdp_growth': ('0.5%', '+0.2%', 'QoQ'),
            'inflation': ('2.8%', '+0.3%', 'YoY'),
            'unemployment': ('3.6%', '-0.1%', 'CBS'),
            'trade_balance': ('+€7.8B', '+€0.5B', 'Monthly')
        },
        'statistical_office': {
            'name': 'CBS',
            'full_name': 'Centraal Bureau voor de Statistiek',
            'website': 'https://www.cbs.nl/',
            'api_url': 'https://opendata.cbs.nl/',
            'requires_auth': False,
            'key_datasets': {
                '84105NED': 'BBP (GDP)',
                '83131NED': 'Consumentenprijsindex (CPI)',
                '80590ned': 'Werkloosheid (Unemployment)',
                '82801NED': 'Productie industrie',
                '82008NED': 'Internationale handel (Trade)'
            }
        },
        'central_bank': 'De Nederlandsche Bank',
        'stock_index': {'name': 'AEX', 'symbol': '^AEX', 'value': '892', 'change': '+0.9%'},
        'bond_10y': {'name': 'Dutch 10Y', 'value': '2.85%', 'change': '+0.03%'},
        'sectors': {
            'Industry': {'Weight': '12.8%', 'Growth': '+0.4%', 'Employment': '1.1M'},
            'Construction': {'Weight': '4.5%', 'Growth': '+0.2%', 'Employment': '0.5M'},
            'Services': {'Weight': '78.2%', 'Growth': '+0.6%', 'Employment': '7.8M'},
            'Agriculture': {'Weight': '1.6%', 'Growth': '+0.1%', 'Employment': '0.2M'}
        }
    },
    'spain': {
        'name': 'Spain',
        'flag': '🇪🇸',
        'indicators': {
            'gdp_growth': ('0.6%', '+0.3%', 'QoQ'),
            'inflation': ('3.1%', '+0.5%', 'YoY'),
            'unemployment': ('11.8%', '-0.4%', 'INE'),
            'trade_balance': ('-€3.5B', '-€0.3B', 'Monthly')
        },
        'statistical_office': {
            'name': 'INE',
            'full_name': 'Instituto Nacional de Estadística',
            'website': 'https://www.ine.es/',
            'api_url': 'https://www.ine.es/dyngs/IOE/es/',
            'requires_auth': False,
            'key_series': {
                'IPI': 'PIB (GDP)',
                'IPC': 'Índice de Precios al Consumo (CPI)',
                'EPA': 'Encuesta de Población Activa (Labor Force)',
                'IPRI': 'Índice de Producción Industrial',
                'COMEXT': 'Comercio Exterior (Foreign Trade)'
            }
        },
        'central_bank': 'Banco de España',
        'stock_index': {'name': 'IBEX 35', 'symbol': '^IBEX', 'value': '11,234', 'change': '+1.3%'},
        'bond_10y': {'name': 'Spanish 10Y', 'value': '3.45%', 'change': '+0.06%'},
        'sectors': {
            'Industry': {'Weight': '13.2%', 'Growth': '+0.5%', 'Employment': '2.5M'},
            'Construction': {'Weight': '5.8%', 'Growth': '+0.8%', 'Employment': '1.3M'},
            'Services': {'Weight': '74.3%', 'Growth': '+0.7%', 'Employment': '15.8M'},
            'Agriculture': {'Weight': '2.8%', 'Growth': '+0.2%', 'Employment': '0.8M'}
        }
    },
    'portugal': {
        'name': 'Portugal',
        'flag': '🇵🇹',
        'indicators': {
            'gdp_growth': ('0.7%', '+0.4%', 'QoQ'),
            'inflation': ('2.6%', '+0.2%', 'YoY'),
            'unemployment': ('6.8%', '-0.3%', 'INE'),
            'trade_balance': ('-€2.1B', '-€0.2B', 'Monthly')
        },
        'statistical_office': {
            'name': 'INE Portugal',
            'full_name': 'Instituto Nacional de Estatística',
            'website': 'https://www.ine.pt/',
            'api_url': 'https://www.ine.pt/xportal/xmain?xpid=INE',
            'requires_auth': False,
            'key_series': {
                'CN_PIB': 'PIB (GDP)',
                'IPC': 'Índice de Preços no Consumidor (CPI)',
                'EMPR': 'Taxa de Desemprego (Unemployment)',
                'PROD_IND': 'Produção Industrial',
                'COM_EXT': 'Comércio Internacional'
            }
        },
        'central_bank': 'Banco de Portugal',
        'stock_index': {'name': 'PSI 20', 'symbol': 'PSI20.LS', 'value': '6,234', 'change': '+0.7%'},
        'bond_10y': {'name': 'Portuguese 10Y', 'value': '3.25%', 'change': '+0.05%'},
        'sectors': {
            'Industry': {'Weight': '14.6%', 'Growth': '+0.6%', 'Employment': '1.1M'},
            'Construction': {'Weight': '4.2%', 'Growth': '+0.9%', 'Employment': '0.4M'},
            'Services': {'Weight': '76.8%', 'Growth': '+0.8%', 'Employment': '3.8M'},
            'Agriculture': {'Weight': '2.1%', 'Growth': '+0.3%', 'Employment': '0.4M'}
        }
    },
    'uk': {
        'name': 'United Kingdom',
        'flag': '🇬🇧',
        'indicators': {
            'gdp_growth': ('0.3%', '+0.1%', 'QoQ'),
            'inflation': ('3.9%', '+0.6%', 'YoY'),
            'unemployment': ('4.2%', '+0.1%', 'ONS'),
            'trade_balance': ('-£2.8B', '-£0.4B', 'Monthly')
        },
        'statistical_office': {
            'name': 'ONS',
            'full_name': 'Office for National Statistics',
            'website': 'https://www.ons.gov.uk/',
            'api_url': 'https://api.ons.gov.uk/',
            'requires_auth': False,
            'key_datasets': {
                'ABMI': 'GDP',
                'D7G7': 'CPI',
                'MGSX': 'Unemployment Rate',
                'K222': 'Industrial Production',
                'BOKI': 'Trade Balance'
            }
        },
        'central_bank': 'Bank of England',
        'stock_index': {'name': 'FTSE 100', 'symbol': '^FTSE', 'value': '8,234', 'change': '+0.5%'},
        'bond_10y': {'name': 'Gilt 10Y', 'value': '4.25%', 'change': '+0.07%'},
        'sectors': {
            'Industry': {'Weight': '9.8%', 'Growth': '+0.1%', 'Employment': '2.7M'},
            'Construction': {'Weight': '6.1%', 'Growth': '+0.3%', 'Employment': '2.3M'},
            'Services': {'Weight': '80.4%', 'Growth': '+0.4%', 'Employment': '27.2M'},
            'Agriculture': {'Weight': '0.6%', 'Growth': '-0.1%', 'Employment': '0.5M'}
        }
    },
    'norway': {
        'name': 'Norway',
        'flag': '🇳🇴',
        'indicators': {
            'gdp_growth': ('0.4%', '+0.2%', 'QoQ'),
            'inflation': ('4.8%', '+0.7%', 'YoY'),
            'unemployment': ('3.8%', '+0.2%', 'SSB'),
            'oil_production': ('1.8M bpd', '+0.1M', 'Daily')
        },
        'statistical_office': {
            'name': 'SSB',
            'full_name': 'Statistics Norway / Statistisk sentralbyrå',
            'website': 'https://www.ssb.no/',
            'api_url': 'https://data.ssb.no/api/',
            'requires_auth': False,
            'key_datasets': {
                '09170': 'BNP (GDP)',
                '03013': 'Konsumprisindeks (CPI)',
                '09545': 'Arbeidsledighet (Unemployment)',
                '09170': 'Industriproduksjon',
                '08804': 'Petroleumsproduksjon'
            }
        },
        'central_bank': 'Norges Bank',
        'stock_index': {'name': 'OBX', 'symbol': 'OBX.OL', 'value': '1,234', 'change': '+0.8%'},
        'bond_10y': {'name': 'Norwegian 10Y', 'value': '3.85%', 'change': '+0.06%'},
        'special_indicators': {
            'Oil Fund': 'NOK 15.9T',
            'Brent Crude': '$85/bbl',
            'NOK/EUR': '11.45'
        }
    },
    'sweden': {
        'name': 'Sweden',
        'flag': '🇸🇪',
        'indicators': {
            'gdp_growth': ('0.2%', '-0.1%', 'QoQ'),
            'inflation': ('5.5%', '+0.8%', 'YoY'),
            'unemployment': ('7.6%', '+0.3%', 'SCB'),
            'trade_balance': ('+SEK 8.2B', '+SEK 1.1B', 'Monthly')
        },
        'statistical_office': {
            'name': 'SCB',
            'full_name': 'Statistics Sweden / Statistiska centralbyrån',
            'website': 'https://www.scb.se/',
            'api_url': 'https://api.scb.se/',
            'requires_auth': False,
            'key_datasets': {
                'NR0103': 'BNP (GDP)',
                'PR0101': 'Konsumentprisindex (CPI)',
                'AM0401': 'Arbetslöshet (Unemployment)',
                'IN0101': 'Industriproduktion',
                'HA0201': 'Utrikeshandel (Foreign Trade)'
            }
        },
        'central_bank': 'Sveriges Riksbank',
        'stock_index': {'name': 'OMX Stockholm 30', 'symbol': '^OMX', 'value': '2,456', 'change': '+0.6%'},
        'bond_10y': {'name': 'Swedish 10Y', 'value': '2.75%', 'change': '+0.04%'},
        'sectors': {
            'Industry': {'Weight': '15.2%', 'Growth': '+0.2%', 'Employment': '0.9M'},
            'Construction': {'Weight': '6.3%', 'Growth': '-0.2%', 'Employment': '0.4M'},
            'Services': {'Weight': '72.8%', 'Growth': '+0.3%', 'Employment': '4.2M'},
            'Agriculture': {'Weight': '1.2%', 'Growth': '+0.1%', 'Employment': '0.1M'}
        }
    },
    'austria': {
        'name': 'Austria',
        'flag': '🇦🇹',
        'indicators': {
            'gdp_growth': ('0.2%', '+0.1%', 'QoQ'),
            'inflation': ('3.4%', '+0.5%', 'YoY'),
            'unemployment': ('5.1%', '-0.1%', 'AMS'),
            'trade_balance': ('+€1.2B', '+€0.2B', 'Monthly')
        },
        'statistical_office': {
            'name': 'Statistics Austria',
            'full_name': 'Statistik Austria',
            'website': 'https://www.statistik.at/',
            'api_url': 'https://www.statistik.at/opendata/',
            'requires_auth': False,
            'key_datasets': {
                'OGD_vgr001_VGRJahre_1': 'BIP (GDP)',
                'OGD_vpi15_VPI_2015_1': 'Verbraucherpreisindex (CPI)',
                'OGD_alu_UEB_ALQ_1': 'Arbeitslosenquote (Unemployment)',
                'OGD_prodn_PRO_1': 'Produktionsindex',
                'OGD_f1531_Aussenhandel_1': 'Außenhandel (Foreign Trade)'
            }
        },
        'central_bank': 'Oesterreichische Nationalbank',
        'stock_index': {'name': 'ATX', 'symbol': '^ATX', 'value': '3,567', 'change': '+0.4%'},
        'bond_10y': {'name': 'Austrian 10Y', 'value': '3.05%', 'change': '+0.05%'},
        'sectors': {
            'Industry': {'Weight': '18.5%', 'Growth': '+0.3%', 'Employment': '0.7M'},
            'Construction': {'Weight': '6.8%', 'Growth': '+0.1%', 'Employment': '0.3M'},
            'Services': {'Weight': '70.2%', 'Growth': '+0.2%', 'Employment': '3.2M'},
            'Agriculture': {'Weight': '1.2%', 'Growth': '+0.1%', 'Employment': '0.2M'}
        }
    },
    'hungary': {
        'name': 'Hungary',
        'flag': '🇭🇺',
        'indicators': {
            'gdp_growth': ('-0.2%', '-0.5%', 'QoQ'),
            'inflation': ('6.5%', '+1.2%', 'YoY'),
            'unemployment': ('4.5%', '+0.2%', 'KSH'),
            'trade_balance': ('+€0.8B', '+€0.1B', 'Monthly')
        },
        'statistical_office': {
            'name': 'KSH',
            'full_name': 'Központi Statisztikai Hivatal',
            'website': 'https://www.ksh.hu/',
            'api_url': 'https://www.ksh.hu/stadat',
            'requires_auth': False,
            'key_datasets': {
                'QNA_GDP': 'GDP',
                'STA_CPI': 'Fogyasztói árindex (CPI)',
                'STA_UNE': 'Munkanélküliségi ráta (Unemployment)',
                'STA_IND': 'Ipari termelés',
                'STA_TRA': 'Külkereskedelem (Foreign Trade)'
            }
        },
        'central_bank': 'Magyar Nemzeti Bank',
        'stock_index': {'name': 'BUX', 'symbol': 'BUX', 'value': '72,345', 'change': '+1.2%'},
        'bond_10y': {'name': 'Hungarian 10Y', 'value': '6.85%', 'change': '+0.12%'},
        'sectors': {
            'Industry': {'Weight': '21.3%', 'Growth': '-0.3%', 'Employment': '1.1M'},
            'Construction': {'Weight': '4.8%', 'Growth': '-0.5%', 'Employment': '0.4M'},
            'Services': {'Weight': '65.2%', 'Growth': '+0.1%', 'Employment': '3.2M'},
            'Agriculture': {'Weight': '3.5%', 'Growth': '+0.2%', 'Employment': '0.5M'}
        }
    },
    'belgium': {
        'name': 'Belgium',
        'flag': '🇧🇪',
        'indicators': {
            'gdp_growth': ('0.3%', '+0.2%', 'QoQ'),
            'inflation': ('2.3%', '+0.1%', 'YoY'),
            'unemployment': ('5.5%', '-0.1%', 'Statbel'),
            'trade_balance': ('+€1.5B', '+€0.3B', 'Monthly')
        },
        'statistical_office': {
            'name': 'Statbel',
            'full_name': 'Statistics Belgium',
            'website': 'https://statbel.fgov.be/',
            'api_url': 'https://bestat.statbel.fgov.be/',
            'requires_auth': False,
            'key_datasets': {
                'BBP_NAT': 'BBP (GDP)',
                'CPI': 'Consumptieprijsindex (CPI)',
                'WERKL': 'Werkloosheidsgraad (Unemployment)',
                'INDPROD': 'Industriële productie',
                'BUITHAND': 'Buitenlandse handel (Foreign Trade)'
            }
        },
        'central_bank': 'National Bank of Belgium',
        'stock_index': {'name': 'BEL 20', 'symbol': '^BFX', 'value': '3,789', 'change': '+0.5%'},
        'bond_10y': {'name': 'Belgian 10Y', 'value': '3.15%', 'change': '+0.04%'},
        'sectors': {
            'Industry': {'Weight': '13.8%', 'Growth': '+0.2%', 'Employment': '0.6M'},
            'Construction': {'Weight': '5.2%', 'Growth': '+0.3%', 'Employment': '0.3M'},
            'Services': {'Weight': '77.4%', 'Growth': '+0.4%', 'Employment': '3.9M'},
            'Agriculture': {'Weight': '0.7%', 'Growth': '+0.1%', 'Employment': '0.1M'}
        }
    },
    'luxembourg': {
        'name': 'Luxembourg',
        'flag': '🇱🇺',
        'indicators': {
            'gdp_growth': ('0.5%', '+0.3%', 'QoQ'),
            'inflation': ('2.0%', '+0.2%', 'YoY'),
            'unemployment': ('5.3%', '+0.1%', 'STATEC'),
            'financial_sector': ('€1.2T', '+€50B', 'Assets')
        },
        'statistical_office': {
            'name': 'STATEC',
            'full_name': 'Institut national de la statistique et des études économiques',
            'website': 'https://statistiques.public.lu/',
            'api_url': 'https://statistiques.public.lu/stat/ReportFolders/',
            'requires_auth': False,
            'key_datasets': {
                'PIB': 'PIB (GDP)',
                'IPCN': 'Indice des prix à la consommation (CPI)',
                'CHOMAGE': 'Taux de chômage (Unemployment)',
                'FINANCE': 'Secteur financier',
                'COMMERCE': 'Commerce extérieur'
            }
        },
        'central_bank': 'Banque centrale du Luxembourg',
        'stock_index': {'name': 'LuxX', 'symbol': 'LUXX', 'value': '1,567', 'change': '+0.3%'},
        'bond_10y': {'name': 'Luxembourg 10Y', 'value': '2.95%', 'change': '+0.03%'},
        'sectors': {
            'Finance': {'Weight': '26.5%', 'Growth': '+0.6%', 'Employment': '48K'},
            'Industry': {'Weight': '6.2%', 'Growth': '+0.2%', 'Employment': '22K'},
            'Services': {'Weight': '65.8%', 'Growth': '+0.5%', 'Employment': '210K'},
            'Agriculture': {'Weight': '0.3%', 'Growth': '+0.1%', 'Employment': '3K'}
        }
    }
}

def render_country_tab(country_key, config):
    """Fonction générique pour afficher l'onglet d'un pays"""
    
    st.markdown(f"### {config['flag']} {config['name'].upper()} ECONOMIC INDICATORS")
    
    # Data sources box
    st.markdown(f"""
    <div class="api-info-box">
        <p style="margin: 0; font-size: 10px; color: #00FFFF; font-weight: bold;">
        📊 DATA SOURCES
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>{config['statistical_office']['name']}:</strong> {config['statistical_office']['full_name']}</li>
            <li><strong>{config['central_bank']}:</strong> Central Bank / Monetary Authority</li>
            <li><strong>Eurostat:</strong> European Union Statistics (where applicable)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Refresh button
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 REFRESH", use_container_width=True, key=f"refresh_{country_key}"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # Key Indicators
    st.markdown(f"#### 📊 KEY {config['name'].upper()} INDICATORS (DEMO)")
    
    if 'indicators' in config and config['indicators']:
        cols = st.columns(min(4, len(config['indicators'])))
        for idx, (label, (value, delta, caption)) in enumerate(config['indicators'].items()):
            with cols[idx % 4]:
                st.metric(label.upper().replace('_', ' '), value, delta, help=caption)
    else:
        st.info("Indicators data not available for this country")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # Statistical Office API
    st.markdown(f"#### 📊 {config['statistical_office']['name']} API ACCESS")
    
    # Build series list HTML
    series_html = ""
    if 'key_series' in config['statistical_office']:
        for code, name in config['statistical_office']['key_series'].items():
            series_html += f"            <li><code>{code}</code> - {name}</li>\n"
    elif 'key_datasets' in config['statistical_office']:
        for code, name in config['statistical_office']['key_datasets'].items():
            series_html += f"            <li><code>{code}</code> - {name}</li>\n"
    
    auth_note = "🔑 Requires API key/account" if config['statistical_office']['requires_auth'] else "💡 No API key required - Open data!"
    
    st.markdown(f"""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 {config['statistical_office']['name']} - {config['statistical_office']['full_name']}
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Website:</strong> {config['statistical_office']['website']}
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API/Data Portal:</strong> {config['statistical_office']['api_url']}
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Key Series/Datasets:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
{series_html}
        </ul>
        <p style="margin: 10px 0 0 0; font-size: 9px; color: #00FFFF;">
        {auth_note}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sectoral Data
    if 'sectors' in config:
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        st.markdown(f"#### 🏭 {config['name'].upper()} SECTORAL INDICATORS")
        
        sector_df = pd.DataFrame(config['sectors']).T.reset_index()
        sector_df.columns = ['Sector'] + list(config['sectors'][list(config['sectors'].keys())[0]].keys())
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
    
    # Market Data
    if 'stock_index' in config and 'bond_10y' in config:
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        st.markdown(f"#### 📈 {config['name'].upper()} MARKET INDICATORS")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.metric(
                config['stock_index']['name'], 
                config['stock_index']['value'], 
                config['stock_index']['change']
            )
            st.caption(f"Stock Index (Symbol: {config['stock_index']['symbol']})")
        
        with col_m2:
            st.metric(
                config['bond_10y']['name'], 
                config['bond_10y']['value'], 
                config['bond_10y']['change']
            )
            st.caption("Government Bond 10-Year Yield")
    
    # Special indicators if available
    if 'special_indicators' in config:
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        st.markdown(f"#### 🎯 SPECIAL {config['name'].upper()} INDICATORS")
        
        special_cols = st.columns(len(config['special_indicators']))
        for idx, (label, value) in enumerate(config['special_indicators'].items()):
            with special_cols[idx]:
                st.metric(label, value)

# ONGLETS PRINCIPAUX
tab_list = ["🇪🇺 EUROPEAN UNION"] + [f"{cfg['flag']} {cfg['name'].upper()}" for cfg in COUNTRIES_CONFIG.values()]
tabs = st.tabs(tab_list)

# Tab 0: European Union Overview
with tabs[0]:
    st.markdown("### 🇪🇺 EUROPEAN UNION ECONOMIC INDICATORS")
    
    st.markdown("""
    <div class="api-info-box">
        <p style="margin: 0; font-size: 10px; color: #00FFFF; font-weight: bold;">
        📊 DATA SOURCES
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>ECB SDW:</strong> European Central Bank Statistical Data Warehouse</li>
            <li><strong>Eurostat:</strong> Statistical Office of the European Union</li>
            <li><strong>OECD:</strong> Organisation for Economic Co-operation and Development</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Refresh button
    col_eu1, col_eu2 = st.columns([5, 1])
    with col_eu2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_eu"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # Key Eurozone Indicators
    st.markdown("#### 📊 KEY EUROZONE INDICATORS")
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 ECB STATISTICAL DATA WAREHOUSE
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Access:</strong> https://sdw.ecb.europa.eu/
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API Endpoint:</strong> https://sdw-wsrest.ecb.europa.eu/service/data/
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Key Series Examples:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><code>ICP/M.U2.N.000000.4.ANR</code> - HICP Inflation YoY</li>
            <li><code>FM/D.U2.EUR.4F.KR.MRR_FR.LEV</code> - ECB Main Refinancing Rate</li>
            <li><code>MNA/Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N</code> - GDP Eurozone</li>
            <li><code>STS/M.I8.Y.UNEH.RTT000.4.000</code> - Unemployment Rate</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard with demo indicators
    st.markdown("#### 📊 EUROZONE DASHBOARD (DEMO)")
    
    cols_eu = st.columns(4)
    demo_indicators = [
        ('HICP INFLATION', '2.4%', '+0.2%', 'YoY'),
        ('ECB RATE', '4.50%', '0.0%', 'Deposit Facility'),
        ('UNEMPLOYMENT', '6.5%', '-0.1%', 'Eurozone'),
        ('GDP GROWTH', '0.5%', '+0.1%', 'QoQ')
    ]
    
    for idx, (label, value, delta, caption) in enumerate(demo_indicators):
        with cols_eu[idx]:
            st.metric(label, value, delta, help=caption)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # Country Breakdown
    st.markdown("#### 🌍 EUROZONE COUNTRY BREAKDOWN")
    
    eurozone_summary = []
    for key, cfg in COUNTRIES_CONFIG.items():
        # Only include Eurozone countries
        if key in ['france', 'germany', 'italy', 'spain', 'netherlands', 'belgium', 
                   'austria', 'portugal', 'luxembourg']:
            eurozone_summary.append({
                'Country': f"{cfg['flag']} {cfg['name']}",
                'Inflation': cfg['indicators'].get('inflation', ('N/A', '', ''))[0],
                'Unemployment': cfg['indicators'].get('unemployment', ('N/A', '', ''))[0],
                'GDP Growth': cfg['indicators'].get('gdp_growth', ('N/A', '', ''))[0]
            })
    
    summary_df = pd.DataFrame(eurozone_summary)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Eurostat Integration
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 📊 EUROSTAT DATA ACCESS")
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 EUROSTAT API
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Website:</strong> https://ec.europa.eu/eurostat
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API:</strong> https://ec.europa.eu/eurostat/api/dissemination/
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Popular Datasets:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><code>prc_hicp_midx</code> - HICP Monthly Index</li>
            <li><code>namq_10_gdp</code> - GDP and Main Components</li>
            <li><code>une_rt_m</code> - Unemployment by Sex and Age</li>
            <li><code>sts_inpr_m</code> - Industrial Production Index</li>
            <li><code>ert_bil_eur_m</code> - Bilateral Exchange Rates</li>
        </ul>
        <p style="margin: 10px 0 0 0; font-size: 9px; color: #00FFFF;">
        💡 No API key required - Open data!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tabs 1-14: Individual Countries
for idx, (country_key, config) in enumerate(COUNTRIES_CONFIG.items(), 1):
    with tabs[idx]:
        render_country_tab(country_key, config)

# Integration Section
st.markdown('<div style="border-top: 2px solid #FFAA00; margin: 20px 0;"></div>', unsafe_allow_html=True)
st.markdown("### 🔗 DATA INTEGRATION & TOOLS")

integration_cols = st.columns(3)

with integration_cols[0]:
    st.markdown("#### 📊 QUICK LINKS")
    st.markdown("""
    **🇪🇺 Pan-European:**
    - [ECB SDW](https://sdw.ecb.europa.eu/)
    - [Eurostat](https://ec.europa.eu/eurostat)
    - [OECD](https://data.oecd.org/)
    
    **📈 Market Data:**
    - [Yahoo Finance](https://finance.yahoo.com/)
    - [Investing.com](https://www.investing.com/)
    - [Trading Economics](https://tradingeconomics.com/)
    """)

with integration_cols[1]:
    st.markdown("#### 🛠️ PYTHON LIBRARIES")
    st.markdown("""
    **Recommended packages:**
```bash
# European data
pip install eurostat
pip install pandas-datareader

# Country-specific
pip install pynsee  # France
pip install wbdata  # World Bank

# General
pip install requests pandas
pip install plotly streamlit
```
    """)

with integration_cols[2]:
    st.markdown("#### 📈 NEXT STEPS")
    st.markdown("""
    **Implementation Guide:**
    
    1. **API Setup:**
       - Check which APIs need credentials
       - Register for free accounts
       
    2. **Data Collection:**
       - Test API connections
       - Verify data formats
       
    3. **Dashboard Building:**
       - Combine multiple sources
       - Add visualizations
       - Implement caching
    """)

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | EUROPEAN ECONOMIC DATA | LAST UPDATE: {last_update}
    <br>
    Data sources: ECB, Eurostat, National Statistical Offices, OECD
    <br>
    {len(COUNTRIES_CONFIG)} Countries Covered
</div>
""", unsafe_allow_html=True)
