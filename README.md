# 🔍 SERP Analyzer Pro

Uno strumento di analisi SERP (Search Engine Results Page) sviluppato con Streamlit per analizzare la concorrenza e le tipologie di pagine nei risultati di ricerca di Google.

## ✨ Caratteristiche

- **Analisi SERP Completa**: Analizza i risultati di ricerca per multiple keyword
- **Classificazione Automatica**: Utilizza OpenAI GPT-4 per classificare automaticamente le tipologie di pagine
- **Visualizzazioni Interattive**: Grafici e tabelle interattive con Plotly
- **Report Excel**: Genera report completi in formato Excel
- **Interfaccia Web Intuitiva**: Interface user-friendly con Streamlit
- **Analisi Competitor**: Identifica i competitor principali per le tue keyword
- **People Also Ask**: Raccoglie e analizza le domande correlate
- **Related Searches**: Trova query correlate per l'espansione delle keyword

## 🚀 Demo Online

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)

## 📋 Prerequisiti

- Account [Serper.dev](https://serper.dev) (per le ricerche SERP)
- Account [OpenAI](https://openai.com) (per la classificazione delle pagine)
- Python 3.8+

## 🛠️ Installazione Locale

1. **Clona il repository**
```bash
git clone https://github.com/tuo-username/serp-analyzer-pro.git
cd serp-analyzer-pro
```

2. **Installa le dipendenze**
```bash
pip install -r requirements.txt
```

3. **Avvia l'applicazione**
```bash
streamlit run app.py
```

4. **Apri il browser** su `http://localhost:8501`

## 🔧 Configurazione

### API Keys Necessarie

1. **Serper API Key**
   - Registrati su [serper.dev](https://serper.dev)
   - Ottieni la tua API key gratuita (100 ricerche/mese)

2. **OpenAI API Key**
   - Crea un account su [OpenAI](https://platform.openai.com)
   - Genera una API key dalla dashboard

### Parametri Configurabili

- **Paese**: Seleziona il mercato geografico (IT, US, UK, DE, FR, ES)
- **Lingua**: Imposta la lingua dei risultati
- **Numero Risultati**: Da 5 a 20 risultati per query

## 📊 Tipologie di Pagine Riconosciute

L'analizzatore classifica automaticamente le pagine in:

- **Homepage**: Pagine principali dei siti
- **Pagina di Categoria**: Elenchi di prodotti/servizi
- **Pagina Prodotto**: Dettagli di singoli prodotti/servizi
- **Articolo di Blog**: Content marketing e informazioni
- **Pagina di Servizi**: Descrizioni di servizi aziendali
- **Altro**: Pagine non classificabili nelle categorie precedenti

## 📈 Report Generati

Il tool genera report Excel con i seguenti fogli:

1. **Top Domains**: Classifica dei domini per occorrenze
2. **Tipologie di Pagine**: Distribuzione delle tipologie
3. **Competitor e Tipologie**: Analisi dettagliata per competitor
4. **Tipologie per Query**: Breakdown per singola keyword
5. **People Also Ask**: Domande correlate raccolte
6. **Related Queries**: Suggerimenti di keyword correlate

## 🎯 Casi d'Uso

- **SEO Competitive Analysis**: Analizza la concorrenza per le tue keyword target
- **Content Strategy**: Identifica che tipo di contenuti rankano meglio
- **Keyword Research**: Scopri nuove opportunità di keyword
- **Market Research**: Comprendi il panorama competitivo del tuo settore
- **SERP Monitoring**: Monitora i cambiamenti nei risultati di ricerca

## 📱 Deployment su Streamlit Cloud

1. **Fork questo repository** sul tuo account GitHub

2. **Connetti a Streamlit Cloud**
   - Vai su [share.streamlit.io](https://share.streamlit.io)
   - Connetti il tuo repository GitHub
   - Seleziona `app.py` come file principale

3. **Configura i Secrets**
   - Nelle impostazioni dell'app, aggiungi i secrets:
   ```toml
   [secrets]
   SERPER_API_KEY = "la-tua-serper-api-key"
   OPENAI_API_KEY = "la-tua-openai-api-key"
   ```

## 📝 Changelog

### v1.0.0
- Rilascio iniziale
- Analisi SERP completa
- Classificazione automatica pagine
- Export Excel
- Interfaccia Streamlit


---

**Sviluppato da Daniele Pisciottano 🦕**
