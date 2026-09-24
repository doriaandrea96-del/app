import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Configurazione della pagina
st.set_page_config(page_title="Simulatore PAC Storico", layout="wide")

# --- CSS ADATTIVO PER TEMA AUTOMATICO (LIGHT & DARK COMPATIBILE) ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800&family=Quicksand:wght@600;700;800&display=swap');

        /* Font Generali */
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif !important;
        }

        /* Titoli */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Quicksand', sans-serif !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }

        /* Badge Custom Adattivo */
        .mb-badge {
            display: inline-block;
            background: rgba(158, 213, 230, 0.25);
            color: #2A5A69;
            font-family: 'Quicksand', sans-serif;
            font-weight: 800;
            font-size: 13px;
            padding: 6px 14px;
            border-radius: 50px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Metric Cards Adattive (Traslucide per adattarsi sia a Light che Dark mode) */
        [data-testid="stMetric"] {
            background-color: rgba(125, 125, 125, 0.08) !important;
            border: 1px solid rgba(125, 125, 125, 0.18) !important;
            border-radius: 16px !important;
            padding: 16px 10px !important;
            text-align: center;
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
        }
        
        /* Leggibilità Titoli nei Riquadri (Metric Labels) */
        [data-testid="stMetricLabel"] {
            justify-content: center !important;
        }
        [data-testid="stMetricLabel"] p {
            font-family: 'Quicksand', sans-serif !important;
            font-weight: 700 !important;
            font-size: 12px !important;
            text-transform: uppercase;
            opacity: 0.85;
        }
        
        [data-testid="stMetricValue"] div {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 800 !important;
            font-size: 22px !important;
        }

        /* Pulizia dello Slider */
        div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
            font-weight: 700 !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- HEADER SITO ---
st.markdown("<div class='mb-badge'>Backtest Storico</div>", unsafe_allow_html=True)
st.title("Simulatore PAC Storico")
st.markdown("Scopri quanto avresti dovuto investire mensilmente nel passato per raggiungere il tuo obiettivo finanziario, basandoti sui dati reali del mercato.")
st.markdown("<br>", unsafe_allow_html=True)

# --- BARRA LATERALE PER I PARAMETRI ---
st.sidebar.header("Imposta il tuo Piano")
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
    st.error("Nessun dato trovato per questo Ticker. Verifica di averlo scritto correttamente (es. usa il suffisso delle piazze europee come .MI o .DE).")
else:
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

        # Calcoli statistici completi
        min_val = df_rolling['Monthly_Contribution'].min()
        max_val = df_rolling['Monthly_Contribution'].max()
        media_val = df_rolling['Monthly_Contribution'].mean()
        mediana_val = df_rolling['Monthly_Contribution'].median()
        q1_val = df_rolling['Monthly_Contribution'].quantile(0.25)
        q3_val = df_rolling['Monthly_Contribution'].quantile(0.75)

        # --- METRICHE VISUALI (CON MIN E MAX) ---
        st.markdown("### Risultati Statistici")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Minimo (Migliore)", f"€ {min_val:,.0f}")
        col2.metric("1° Quartile (Toro)", f"€ {q1_val:,.0f}")
        col3.metric("Mediana (Centrale)", f"€ {mediana_val:,.0f}")
        col4.metric("Media Aritmetica", f"€ {media_val:,.0f}")
        col5.metric("3° Quartile (Orso)", f"€ {q3_val:,.0f}")
        col6.metric("Massimo (Peggiore)", f"€ {max_val:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- GRAFICI STORICI ---
        st.markdown("### Analisi Storica Versamenti")
        
        color_primary = "#D9534F"   # Terracotta
        color_secondary = "#9ED5E6" # Azzurro Pastello
        color_success = "#2C7A5E"   # Verde Menta Scuro

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            fig1, ax1 = plt.subplots(figsize=(8, 4.5), facecolor='none')
            ax1.set_facecolor('none')
            
            ax1.plot(df_rolling['Start_Date'], df_rolling['Monthly_Contribution'], color=color_primary, linewidth=2.5)
            ax1.axhline(mediana_val, color=color_success, linestyle='--', linewidth=2, label=f'Mediana: €{mediana_val:,.0f}')
            
            ax1.set_title('Versamento Storico Necessario', pad=15, fontdict={'fontsize': 13, 'fontweight': 'bold'})
            ax1.set_ylabel('Versamento Mensile (€)', fontweight='bold')
            
            ax1.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
            for spine in ax1.spines.values():
                spine.set_visible(False)
                
            ax1.legend(frameon=False)
            st.pyplot(fig1)

        with fig_col2:
            fig2, ax2 = plt.subplots(figsize=(8, 4.5), facecolor='none')
            ax2.set_facecolor('none')
            
            ax2.hist(df_rolling['Monthly_Contribution'], bins=20, density=True, color=color_secondary, alpha=0.6, edgecolor='none')
            
            kde = gaussian_kde(df_rolling['Monthly_Contribution'])
            x_grid = np.linspace(df_rolling['Monthly_Contribution'].min() * 0.9, df_rolling['Monthly_Contribution'].max() * 1.1, 300)
            ax2.plot(x_grid, kde(x_grid), color=color_primary, linewidth=2.5)
            
            ax2.axvline(mediana_val, color=color_success, linestyle='--', linewidth=2)
            
            ax2.set_title('Distribuzione Probabilistica', pad=15, fontdict={'fontsize': 13, 'fontweight': 'bold'})
            ax2.set_yticks([])
            
            ax2.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2, axis='x')
            for spine in ax2.spines.values():
                spine.set_visible(False)
                
            st.pyplot(fig2)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- NUOVO GRAFICO: ULTIMO PERIODO DISPONIBILE ---
        st.markdown("### Valore Portafoglio (Ultimo Periodo Disponibile)")
        
        last_returns = monthly_returns.iloc[-mesi_totali:]
        last_start = last_returns.index[0]
        last_end = last_returns.index[-1]
        last_req_contrib = df_rolling.iloc[-1]['Monthly_Contribution']
        
        port_vals = []
        inv_vals = []
        curr_v = 0
        curr_i = 0
        for dt, ret in last_returns.items():
            curr_i += last_req_contrib
            curr_v = (curr_v + last_req_contrib) * (1 + ret)
            port_vals.append(curr_v)
            inv_vals.append(curr_i)
            
        fig3, ax3 = plt.subplots(figsize=(12, 4.5), facecolor='none')
        ax3.set_facecolor('none')
        
        ax3.plot(last_returns.index, port_vals, color=color_primary, linewidth=2.5, label='Valore Portafoglio')
        ax3.plot(last_returns.index, inv_vals, color=color_secondary, linewidth=2, linestyle='--', label='Capitale Versato')
        
        ax3.set_title(f"Andamento Reale per la Finestra {last_start.strftime('%m/%Y')} - {last_end.strftime('%m/%Y')} (Rata: €{last_req_contrib:,.0f}/mese)", pad=15, fontdict={'fontsize': 13, 'fontweight': 'bold'})
        ax3.set_ylabel('Valore (€)', fontweight='bold')
        ax3.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        for spine in ax3.spines.values():
            spine.set_visible(False)
            
        ax3.legend(frameon=False)
        st.pyplot(fig3)
            
        st.markdown(f"<p style='text-align: center; opacity: 0.7; font-size: 13px; font-weight: 500;'>Dati analizzati dal {monthly_prices.index[0].strftime('%Y')} al {monthly_prices.index[-1].strftime('%Y')} per finestre di {DURATA_ANNI} anni.</p>", unsafe_allow_html=True)
