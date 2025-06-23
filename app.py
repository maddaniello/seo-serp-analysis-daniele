import streamlit as st
import requests
import pandas as pd
import time
import json
from collections import Counter, defaultdict
from urllib.parse import urlparse
from openai import OpenAI
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# Configurazione della pagina
st.set_page_config(
    page_title="SERP Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class SERPAnalyzer:
    def __init__(self, serper_api_key, openai_api_key):
        self.serper_api_key = serper_api_key
        self.openai_api_key = openai_api_key
        self.serper_url = "https://google.serper.dev/search"
        self.client = OpenAI(api_key=openai_api_key)

    def fetch_serp_results(self, query, country="it", language="it", num_results=10):
        """Effettua la ricerca SERP tramite Serper API"""
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = json.dumps({
            "q": query,
            "num": num_results,
            "gl": country,
            "hl": language
        })
        
        try:
            response = requests.post(self.serper_url, headers=headers, data=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Errore API per query '{query}': {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Errore di connessione: {e}")
            return None

    def classify_page_type_gpt(self, url, title, snippet=""):
        """Classifica la tipologia della pagina usando OpenAI"""
        prompt = f"""Classifica la tipologia della seguente pagina web basandoti sull'URL, titolo e snippet:

        URL: {url}
        Titolo: {title}
        Snippet: {snippet}

        Le possibili categorie sono:
        - Homepage
        - Pagina di Categoria
        - Pagina Prodotto
        - Articolo di Blog
        - Pagina di Servizi
        - Altro

        Rispondi solo con la categoria esatta, senza testo aggiuntivo.
        La risposta deve essere in italiano.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Classifica la tipologia di pagina web in modo preciso e conciso."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=15,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Errore OpenAI: {e}")
            return "Altro"

    def parse_results(self, data, query):
        """Analizza i risultati SERP"""
        domain_page_types = defaultdict(lambda: defaultdict(int))
        domain_occurences = defaultdict(int)
        query_page_types = defaultdict(list)
        paa_questions = []
        related_queries = []
        paa_to_queries = defaultdict(set)
        related_to_queries = defaultdict(set)
        paa_to_domains = defaultdict(set)

        # Elaborazione risultati organici
        if "organic" in data:
            for result in data["organic"]:
                domain = urlparse(result["link"]).netloc
                url = result["link"]
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                page_type = self.classify_page_type_gpt(url, title, snippet)

                domain_page_types[domain][page_type] += 1
                domain_occurences[domain] += 1
                query_page_types[query].append(page_type)

        # People Also Ask
        if "peopleAlsoAsk" in data:
            for paa in data["peopleAlsoAsk"]:
                paa_text = paa["question"]
                paa_questions.append(paa_text)
                paa_to_queries[paa_text].add(query)
                paa_to_domains[paa_text].update([domain for domain in domain_page_types.keys()])

        # Related Searches
        if "relatedSearches" in data:
            for related in data["relatedSearches"]:
                related_text = related["query"]
                related_queries.append(related_text)
                related_to_queries[related_text].add(query)

        return (domain_page_types, domain_occurences, query_page_types, 
                paa_questions, related_queries, paa_to_queries, 
                related_to_queries, paa_to_domains)

    def create_excel_report(self, domains_counter, domain_occurences, query_page_types, 
                           domain_page_types, paa_questions, related_queries, 
                           paa_to_queries, related_to_queries, paa_to_domains):
        """Crea il report Excel"""
        
        # DataFrame per domini e tipologie
        domain_page_types_list = []
        page_type_counter = Counter()

        for domain, page_type_dict in domain_page_types.items():
            domain_data = {
                "Competitor": domain, 
                "Numero occorrenze": domain_occurences[domain]
            }
            
            for page_type in ['Homepage', 'Pagina di Categoria', 'Pagina Prodotto', 
                            'Articolo di Blog', 'Pagina di Servizi', 'Altro']:
                domain_data[page_type] = page_type_dict.get(page_type, 0)
                page_type_counter[page_type] += domain_data[page_type]
            
            domain_page_types_list.append(domain_data)

        domain_page_types_df = pd.DataFrame(domain_page_types_list)

        # Top Domains
        domains_df = pd.DataFrame(domains_counter.items(), columns=["Dominio", "Occorrenze"])
        total_queries = sum(domains_counter.values())
        domains_df["% Presenza"] = (domains_df["Occorrenze"] / total_queries * 100).round(2)

        # Query page types
        query_page_type_data = []
        for query, page_types in query_page_types.items():
            for page_type, count in Counter(page_types).items():
                query_page_type_data.append({
                    "Query": query, 
                    "Tipologia Pagina": page_type, 
                    "Occorrenze": count
                })
        query_page_type_df = pd.DataFrame(query_page_type_data)

        # PAA DataFrame
        paa_df = pd.DataFrame(paa_questions, columns=["People Also Ask"])
        paa_df["Keyword che lo attivano"] = paa_df["People Also Ask"].map(
            lambda x: ", ".join(paa_to_queries[x])
        )

        # Related Queries DataFrame
        related_df = pd.DataFrame(related_queries, columns=["Related Query"])
        related_df["Keyword che lo attivano"] = related_df["Related Query"].map(
            lambda x: ", ".join(related_to_queries[x])
        )

        # Page Types DataFrame
        page_type_df = pd.DataFrame(page_type_counter.items(), 
                                  columns=["Tipologia Pagina", "Occorrenze"])

        # Creazione file Excel in memoria
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            domains_df.to_excel(writer, sheet_name="Top Domains", index=False)
            page_type_df.to_excel(writer, sheet_name="Tipologie di Pagine", index=False)
            domain_page_types_df.to_excel(writer, sheet_name="Competitor e Tipologie", index=False)
            query_page_type_df.to_excel(writer, sheet_name="Tipologie per Query", index=False)
            paa_df.to_excel(writer, sheet_name="People Also Ask", index=False)
            related_df.to_excel(writer, sheet_name="Related Queries", index=False)

        return output.getvalue(), domains_df, page_type_df, domain_page_types_df

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 SERP Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar per configurazione
    st.sidebar.header("⚙️ Configurazione")
    
    # API Keys
    serper_api_key = st.sidebar.text_input(
        "Serper API Key", 
        type="password",
        help="Inserisci la tua API key di Serper.dev"
    )
    
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Inserisci la tua API key di OpenAI"
    )

    # Parametri di ricerca
    st.sidebar.subheader("🌍 Parametri di Ricerca")
    country = st.sidebar.selectbox(
        "Paese",
        ["it", "us", "uk", "de", "fr", "es"],
        index=0,
        help="Seleziona il paese per la ricerca"
    )
    
    language = st.sidebar.selectbox(
        "Lingua",
        ["it", "en", "de", "fr", "es"],
        index=0,
        help="Seleziona la lingua dei risultati"
    )
    
    num_results = st.sidebar.slider(
        "Numero di risultati per query",
        min_value=5,
        max_value=20,
        value=10,
        help="Numero di risultati da analizzare per ogni query"
    )

    # Input queries
    st.header("📝 Inserisci le Query")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        queries_input = st.text_area(
            "Query da analizzare (una per riga)",
            height=200,
            placeholder="Inserisci le tue keyword qui...\nUna per ogni riga\n\nEsempio:\ncorso python\ncorso programmazione\nlearn python online"
        )
    
    with col2:
        st.markdown("### 💡 Suggerimenti")
        st.info("""
        • Una query per riga
        • Massimo 1000 query
        • Evita caratteri speciali
        • Usa query specifiche per il tuo settore
        """)

    # Pulsante di avvio
    if st.button("🚀 Avvia Analisi", type="primary", use_container_width=True):
        if not serper_api_key or not openai_api_key:
            st.error("⚠️ Inserisci entrambe le API keys!")
            return
        
        if not queries_input.strip():
            st.error("⚠️ Inserisci almeno una query!")
            return

        queries = [q.strip() for q in queries_input.strip().split('\n') if q.strip()]
        
        if len(queries) > 1000:
            st.error("⚠️ Massimo 1000 query per volta!")
            return

        # Inizializzazione analyzer
        analyzer = SERPAnalyzer(serper_api_key, openai_api_key)
        
        # Progress bar e containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Contenitori per risultati live
        live_results = st.container()
        
        # Variabili per accumulo dati
        all_domains = []
        query_page_types = defaultdict(list)
        domain_page_types = defaultdict(lambda: defaultdict(int))
        domain_occurences = defaultdict(int)
        paa_questions = []
        related_queries = []
        paa_to_queries = defaultdict(set)
        related_to_queries = defaultdict(set)
        paa_to_domains = defaultdict(set)

        # Elaborazione queries
        for i, query in enumerate(queries):
            status_text.text(f"🔍 Analizzando: {query} ({i+1}/{len(queries)})")
            
            results = analyzer.fetch_serp_results(query, country, language, num_results)
            
            if results:
                (domain_page_types_query, domain_occurences_query, query_page_types_query,
                 paa_questions_query, related_queries_query, paa_to_queries_query,
                 related_to_queries_query, paa_to_domains_query) = analyzer.parse_results(results, query)
                
                # Accumulo dati
                for domain, page_types in domain_page_types_query.items():
                    for page_type, count in page_types.items():
                        domain_page_types[domain][page_type] += count
                
                for domain, count in domain_occurences_query.items():
                    domain_occurences[domain] += count
                
                for query_key, page_types in query_page_types_query.items():
                    query_page_types[query_key].extend(page_types)
                
                paa_questions.extend(paa_questions_query)
                related_queries.extend(related_queries_query)
                paa_to_queries.update(paa_to_queries_query)
                related_to_queries.update(related_to_queries_query)
                paa_to_domains.update(paa_to_domains_query)
                all_domains.extend(domain_page_types_query.keys())
            
            # Update progress
            progress_bar.progress((i + 1) / len(queries))
            time.sleep(1)  # Rate limiting

        status_text.text("✅ Analisi completata! Generazione report...")

        # Creazione report
        domains_counter = Counter(all_domains)
        excel_data, domains_df, page_type_df, domain_page_types_df = analyzer.create_excel_report(
            domains_counter, domain_occurences, query_page_types, domain_page_types,
            paa_questions, related_queries, paa_to_queries, related_to_queries, paa_to_domains
        )

        status_text.text("📊 Visualizzazione risultati...")

        # Risultati
        st.markdown("---")
        st.header("📊 Risultati Analisi")

        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Query Analizzate", len(queries))
        with col2:
            st.metric("Domini Trovati", len(domains_counter))
        with col3:
            st.metric("PAA Questions", len(set(paa_questions)))
        with col4:
            st.metric("Related Queries", len(set(related_queries)))

        # Grafici
        st.subheader("📈 Top Domini")
        if not domains_df.empty:
            fig_domains = px.bar(
                domains_df.head(15), 
                x="Dominio", 
                y="Occorrenze",
                title="Top 15 Domini per Occorrenze"
            )
            fig_domains.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_domains, use_container_width=True)

        st.subheader("🏷️ Distribuzione Tipologie di Pagine")
        if not page_type_df.empty:
            fig_pie = px.pie(
                page_type_df, 
                values="Occorrenze", 
                names="Tipologia Pagina",
                title="Distribuzione delle Tipologie di Pagine"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Tabelle risultati
        st.subheader("📋 Tabelle Dettagliate")
        
        tab1, tab2, tab3 = st.tabs(["Top Domini", "Tipologie Pagine", "Competitor Analysis"])
        
        with tab1:
            st.dataframe(domains_df, use_container_width=True)
        
        with tab2:
            st.dataframe(page_type_df, use_container_width=True)
        
        with tab3:
            st.dataframe(domain_page_types_df, use_container_width=True)

        # Download Excel
        st.subheader("💾 Download Report")
        st.download_button(
            label="📥 Scarica Report Excel Completo",
            data=excel_data,
            file_name=f"serp_analysis_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        progress_bar.empty()
        status_text.text("🎉 Analisi completata con successo!")

if __name__ == "__main__":
    main()
