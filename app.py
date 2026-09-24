import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Configurazione della pagina
st.set_page_config(page_title="Simulatore PAC Storico", page_icon="📈", layout="wide")

# --- INIEZIONE CSS DESIGN SYSTEM MONEYBOX HUB ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800&family=Quicksand:wght@600;700;800&display=swap');

        /* Font Generali */
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif !important;
            color: #4A4A4A;
        }

        /* Titoli */
        h1, h2, h3 {
            font-family: 'Quicksand', sans-serif !important;
            font-weight: 800 !important;
            color: #2D3748 !important;
        }

        /* Badge Custom */
        .mb-badge {
            display: inline-block;
            background: rgba(152, 215, 194, 0.25); /* Menta pastello */
            color: #2C7A5E;
            font-family: 'Quicksand', sans-serif;
            font-weight: 800;
            font-size: 14px;
            padding: 8px 16px;
            border-radius: 50px;
            margin-bottom: 16px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* Sidebar in Crema */
        [data-testid="stSidebar"] {
            background-color: #FDFBF7 !important;
            border-right: 1px solid #f0eee9;
        }

        /* Stile delle Metriche (Card) */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #f0eee9;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(45, 55, 72, 0.05);
            text-align: center;
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Quicksand', sans-serif;
            font-weight: 700;
            color: #718096;
            text-transform: uppercase;
            font-size: 13px !important;
            display: flex;
            justify-content: center;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Montserrat', sans-serif;
            font-weight: 800;
            color: #2D3748;
            font-size: 26px !important;
        }

        /* Input Fields e Sliders (Accenti Terracotta) */
        .stSlider > div > div > div > div { background-color: #D9534F !important; }
        .stNumberInput input, .stTextInput input {
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
        .stNumberInput input:focus, .stTextInput input:focus {
            border-color: #9ED5E6 !important;
            box-shadow: 0 0 0 3px rgba(158, 213, 230, 0.3) !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- HEADER APP ---
st.markdown("<div class='mb-badge'>📈 Backtest Storico</div>", unsafe_allow_html=True)
st.title("Simulatore PAC Storico")
st.markdown("Scopri quanto avresti dovuto investire mensilmente nel passato per raggiungere il tuo obiettivo finanziario, basandoti sui dati reali del mercato.")
st.markdown("---")

# --- BARRA LATERALE PER I PARAMETRI ---
st.sidebar.header("⚙️ Imposta il tuo Piano")
TICKER = st.sidebar.text_input("Ticker di Riferimento", "^GSPC", help="Usa ^GSPC per S&P 500, VWCE.DE per Vanguard All-World, ecc.")
DURATA_ANNI = st.sidebar.slider("Durata dell'Investimento (Anni)", min_value=5, max_value=40, value=20, step=1)
OBIETTIVO_FINALE = st.sidebar.number_input("Obiettivo Finale (€)", min_value=10000, max_value=10000000, value=1000000, step=50000)

# --- DOWNLOAD DATI CON CACHE ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="1970-01-01", progress=False)
    if data.empty:
        return None
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close'][ticker] if 'Adj Close' in data.columns.levels[0] else data['Close'][ticker]
    else:
        prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    
    return prices

prices = load_data(TICKER)

if prices is None:
    st.error("⚠️ Nessun dato trovato per questo Ticker. Verifica di averlo scritto correttamente (es. usa il suffisso delle piazze europee come .MI o .DE).")
else:
    # Pre-elaborazione dati (Primo giorno del mese)
    try:
        monthly_prices = prices.resample('MS').first().dropna()
    except Exception:
        monthly_prices = prices.resample('M').first().dropna()

    monthly_returns = monthly_prices.pct_change().dropna()
    
    # --- ALGORITMO BACKTEST ---
    mesi_totali = DURATA_ANNI * 12
    rolling_results = []

    for i in range(len(monthly_returns) - mesi_totali + 1):
        start_date = monthly_returns.index[i]
        end_date = monthly_returns.index[i + mesi_totali - 1]
        
        period_returns = monthly_returns.iloc[i : i + mesi_totali]
        cum_factors = np.cumprod(1 + period_returns.values[::-1])[::-1]
        s_factor = np.sum(cum_factors)
        monthly_required = OBIETTIVO_FINALE / s_factor
        
        rolling_results.append({
            'Start_Year': start_date.year,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Monthly_Contribution': monthly_required
        })

    if not rolling_results:
        st.warning(f"Lo storico di questo strumento non è lungo abbastanza per simulare {DURATA_ANNI} anni. Riduci la durata o cambia Ticker.")
    else:
        df_rolling = pd.DataFrame(rolling_results)

        # Calcoli statistici
        media_val = df_rolling['Monthly_Contribution'].mean()
        mediana_val = df_rolling['Monthly_Contribution'].median()
        q1_val = df_rolling['Monthly_Contribution'].quantile(0.25)
        q3_val = df_rolling['Monthly_Contribution'].quantile(0.75)

        # --- METRICHE VISUALI ---
        st.markdown("### 🎯 Risultati Statistici")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mediana (Caso Centrale)", f"€ {mediana_val:,.0f}")
        col2.metric("1° Quartile (Mercato Toro)", f"€ {q1_val:,.0f}")
        col3.metric("3° Quartile (Mercato Orso)", f"€ {q3_val:,.0f}")
        col4.metric("Media Aritmetica", f"€ {media_val:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- GRAFICI (STILE MONEYBOX) ---
        st.markdown("### 📉 Analisi Visiva")
        
        # Colori Brand
        color_primary = "#D9534F" # Terracotta
        color_secondary = "#9ED5E6" # Azzurro
        color_success = "#2C7A5E" # Menta Scuro (per testi/linee su sfondi chiari)
        bg_color = "#FFFFFF"

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5), facecolor=bg_color)
            ax1.set_facecolor(bg_color)
            
            # Linea Principale
            ax1.plot(df_rolling['Start_Date'], df_rolling['Monthly_Contribution'], color=color_primary, linewidth=2.5)
            # Linea Mediana
            ax1.axhline(mediana_val, color=color_success, linestyle='--', linewidth=2, label=f'Mediana: €{mediana_val:,.0f}')
            
            ax1.set_title('Versamento Storico Necessario', pad=15, fontdict={'fontsize': 14, 'fontweight': 'bold', 'color': '#2D3748'})
            ax1.set_ylabel('Versamento Mensile (€)', color='#718096', fontweight='bold')
            ax1.tick_params(colors='#718096')
            
            # Griglia morbida
            ax1.grid(color='#f0eee9', linestyle='-', linewidth=1, alpha=0.7)
            for spine in ax1.spines.values():
                spine.set_visible(False)
                
            ax1.legend(frameon=False, labelcolor='#4A4A4A')
            st.pyplot(fig1)

        with fig_col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor=bg_color)
            ax2.set_facecolor(bg_color)
            
            # Istogramma
            ax2.hist(df_rolling['Monthly_Contribution'], bins=20, density=True, color=color_secondary, alpha=0.6, edgecolor='#FFFFFF', linewidth=1.5)
            
            # Curva KDE
            kde = gaussian_kde(df_rolling['Monthly_Contribution'])
            x_grid = np.linspace(df_rolling['Monthly_Contribution'].min() * 0.9, df_rolling['Monthly_Contribution'].max() * 1.1, 300)
            ax2.plot(x_grid, kde(x_grid), color=color_primary, linewidth=2.5)
            
            # Linea Mediana
            ax2.axvline(mediana_val, color=color_success, linestyle='--', linewidth=2)
            
            ax2.set_title('Distribuzione Probabilistica', pad=15, fontdict={'fontsize': 14, 'fontweight': 'bold', 'color': '#2D3748'})
            ax2.tick_params(colors='#718096')
            ax2.set_yticks([]) # Nascondiamo l'asse Y per l'istogramma, rende il tutto più "pulito"
            
            # Griglia morbida
            ax2.grid(color='#f0eee9', linestyle='-', linewidth=1, alpha=0.7, axis='x')
            for spine in ax2.spines.values():
                spine.set_visible(False)
                
            st.pyplot(fig2)
            
        st.markdown(f"<p style='text-align: center; color: #718096; font-size: 13px; font-weight: 500;'>Dati analizzati dal {monthly_prices.index[0].strftime('%Y')} al {monthly_prices.index[-1].strftime('%Y')} per finestre di {DURATA_ANNI} anni.</p>", unsafe_allow_html=True)
