import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Configurazione della pagina
st.set_page_config(page_title="Simulatore PAC Storico", page_icon="📈", layout="wide")

st.title("📊 Simulatore PAC Storico")
st.markdown("Scopri quanto avresti dovuto investire mensilmente nel passato per raggiungere il tuo obiettivo finanziario, basandoti sui dati reali del mercato.")

# --- BARRA LATERALE PER I PARAMETRI ---
st.sidebar.header("⚙️ Imposta il tuo Piano")
TICKER = st.sidebar.text_input("Ticker di Riferimento", "^GSPC", help="Usa ^GSPC per S&P 500, VWCE.DE per Vanguard All-World, ecc.")
DURATA_ANNI = st.sidebar.slider("Durata dell'Investimento (Anni)", min_value=5, max_value=40, value=20, step=1)
OBIETTIVO_FINALE = st.sidebar.number_input("Obiettivo Finale (€)", min_value=10000, max_value=10000000, value=1000000, step=50000)

# --- DOWNLOAD DATI CON CACHE ---
# Il decoratore evita di scaricare i dati a ogni click dell'utente
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

        # --- GRAFICI ---
        st.markdown("### 📉 Analisi Visiva")
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Creiamo due colonne per affiancare i grafici su desktop (si impilano su mobile)
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(df_rolling['Start_Date'], df_rolling['Monthly_Contribution'], color='#1a73e8', linewidth=2)
            ax1.axhline(mediana_val, color='#188038', linestyle='-', linewidth=2, label=f'Mediana: €{mediana_val:,.0f}')
            ax1.set_title('Versamento Storico Necessario', pad=10)
            ax1.set_ylabel('Versamento Mensile (€)')
            ax1.legend()
            st.pyplot(fig1)

        with fig_col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.hist(df_rolling['Monthly_Contribution'], bins=20, density=True, color='#aecbfa', edgecolor='#1a73e8', alpha=0.7)
            kde = gaussian_kde(df_rolling['Monthly_Contribution'])
            x_grid = np.linspace(df_rolling['Monthly_Contribution'].min() * 0.9, df_rolling['Monthly_Contribution'].max() * 1.1, 300)
            ax2.plot(x_grid, kde(x_grid), color='#0d47a1', linewidth=2)
            ax2.axvline(mediana_val, color='#188038', linestyle='-', linewidth=2)
            ax2.set_title('Distribuzione Probabilistica', pad=10)
            st.pyplot(fig2)
            
        st.markdown(f"*Dati analizzati dal {monthly_prices.index[0].strftime('%Y')} al {monthly_prices.index[-1].strftime('%Y')} per finestre di {DURATA_ANNI} anni.*")
