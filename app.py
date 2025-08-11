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
import asyncio
import aiohttp
import concurrent.futures
import threading
from functools import lru_cache
import re
import pickle
import os
from datetime import datetime
import gc

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
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key != "dummy" else None
        self.classification_cache = {}
        self.use_ai = True
        self.batch_size = 5
        # Aggiungi timeout e retry
        self.request_timeout = 30
        self.max_retries = 3
        self.retry_delay = 2

    def fetch_serp_results_with_retry(self, query, country="it", language="it", num_results=10):
        """Effettua la ricerca SERP con retry in caso di errore"""
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
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.serper_url, 
                    headers=headers, 
                    data=payload,
                    timeout=self.request_timeout
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (attempt + 1)
                    st.warning(f"Rate limit raggiunto per '{query}'. Attendo {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.warning(f"Errore API per query '{query}': {response.status_code}")
            except requests.exceptions.Timeout:
                st.warning(f"Timeout per query '{query}', tentativo {attempt + 1}/{self.max_retries}")
            except Exception as e:
                st.warning(f"Errore per query '{query}': {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        return None

    def fetch_serp_results(self, query, country="it", language="it", num_results=10):
        """Wrapper retrocompatibile"""
        return self.fetch_serp_results_with_retry(query, country, language, num_results)

    @lru_cache(maxsize=1000)
    def classify_page_type_rule_based(self, url, title, snippet=""):
        """Classificazione veloce basata su regole per casi comuni"""
        url_lower = url.lower()
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        # Homepage patterns
        if (url_lower.count('/') <= 3 and 
            ('home' in url_lower or url_lower.endswith('.com') or url_lower.endswith('.it') or
             'homepage' in title_lower or 'home page' in title_lower)):
            return "Homepage"
        
        # Product page patterns
        product_patterns = ['product', 'prodotto', 'item', 'articolo', '/p/', 'buy', 'acquista', 'shop']
        if any(pattern in url_lower for pattern in product_patterns):
            return "Pagina Prodotto"
        
        # Category page patterns  
        category_patterns = ['category', 'categoria', 'catalogo', 'catalog', 'collection', 'collezione', 'products', 'prodotti']
        if any(pattern in url_lower for pattern in category_patterns):
            return "Pagina di Categoria"
            
        # Blog patterns
        blog_patterns = ['blog', 'news', 'notizie', 'articolo', 'post', 'article', '/blog/', 'magazine']
        if any(pattern in url_lower for pattern in blog_patterns):
            return "Articolo di Blog"
            
        # Services patterns
        service_patterns = ['service', 'servizio', 'servizi', 'services', 'consulenza', 'consulting']
        if any(pattern in url_lower for pattern in service_patterns):
            return "Pagina di Servizi"
            
        return None

    def classify_page_type_gpt(self, url, title, snippet=""):
        """Classificazione con OpenAI solo per casi complessi"""
        # Prima prova la classificazione rule-based
        rule_based_result = self.classify_page_type_rule_based(url, title, snippet)
        if rule_based_result:
            return rule_based_result
            
        # Cache check
        cache_key = f"{url}_{title}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # Prompt ottimizzato per velocità
        prompt = f"""Classifica SOLO con una di queste categorie:
        
URL: {url}
Titolo: {title}

Categorie: Homepage, Pagina di Categoria, Pagina Prodotto, Articolo di Blog, Pagina di Servizi, Altro

Rispondi solo con la categoria."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            self.classification_cache[cache_key] = result
            return result
        except Exception as e:
            st.warning(f"Errore OpenAI: {e}")
            return "Altro"

    def classify_batch_openai(self, pages_data):
        """Classificazione in batch per ridurre le chiamate API"""
        if not pages_data or not self.use_ai or not self.client:
            return {}
            
        # Raggruppa per classificazione batch
        batch_size = min(len(pages_data), self.batch_size)
        batch_prompt = "Classifica ogni pagina con una di queste categorie: Homepage, Pagina di Categoria, Pagina Prodotto, Articolo di Blog, Pagina di Servizi, Altro\n\n"
        
        for i, (url, title, snippet) in enumerate(pages_data[:batch_size]):
            batch_prompt += f"{i+1}. URL: {url}\n   Titolo: {title}\n\n"
        
        batch_prompt += "Rispondi nel formato: 1. Categoria, 2. Categoria, ecc."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": batch_prompt}],
                max_tokens=100,
                temperature=0
            )
            
            # Parse della risposta batch
            results = {}
            response_text = response.choices[0].message.content.strip()
            lines = response_text.split('\n')
            
            for i, line in enumerate(lines):
                if str(i+1) in line and i < len(pages_data):
                    for category in ["Homepage", "Pagina di Categoria", "Pagina Prodotto", 
                                   "Articolo di Blog", "Pagina di Servizi", "Altro"]:
                        if category in line:
                            url, title, snippet = pages_data[i]
                            cache_key = f"{url}_{title}"
                            results[cache_key] = category
                            break
            
            return results
        except Exception as e:
            st.warning(f"Errore batch OpenAI: {e}")
            return {}

    def cluster_keywords_with_custom(self, keywords, custom_clusters):
        """Clusterizza le keyword usando cluster personalizzati come priorità"""
        if not self.client or not self.use_ai:
            return self.cluster_keywords_simple_custom(keywords, custom_clusters)
        
        # Dividi in batch per evitare prompt troppo lunghi
        batch_size = 50
        all_clusters = {}
        
        # Inizializza cluster personalizzati
        for cluster_name in custom_clusters:
            all_clusters[cluster_name] = []
        
        for i in range(0, len(keywords), batch_size):
            batch_keywords = keywords[i:i+batch_size]
            
            prompt = f"""Ruolo: Esperto di analisi semantica e architettura siti web
Capacità: Specialista in clustering di keyword basato su strutture di siti web esistenti.

Compito: Assegna ogni keyword al cluster più appropriato, dando PRIORITÀ ai cluster predefiniti del sito.

CLUSTER PREDEFINITI (USA QUESTI COME PRIORITÀ):
{chr(10).join([f"- {cluster}" for cluster in custom_clusters])}

Keyword da classificare:
{chr(10).join([f"- {kw}" for kw in batch_keywords])}

Istruzioni:
1. PRIORITÀ ASSOLUTA: Cerca di assegnare ogni keyword a uno dei cluster predefiniti se semanticamente correlata
2. Solo se una keyword NON può essere associata a nessun cluster predefinito, crea un nuovo cluster
3. Ogni cluster deve avere almeno 3 keyword (per quelli nuovi)
4. Se una keyword non si adatta a nessun cluster, mettila in "Generale"

Formato di risposta:
Cluster: [Nome Cluster Predefinito o Nuovo]
- keyword1
- keyword2
- keyword3

Cluster: [Altro Cluster]
- keyword4
- keyword5"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                
                # Parse della risposta
                response_text = response.choices[0].message.content.strip()
                clusters = self.parse_clustering_response_custom(response_text, custom_clusters)
                
                # Merge dei risultati
                for cluster_name, cluster_keywords in clusters.items():
                    if cluster_name in all_clusters:
                        all_clusters[cluster_name].extend(cluster_keywords)
                    else:
                        all_clusters[cluster_name] = cluster_keywords
                
            except Exception as e:
                st.warning(f"Errore clustering personalizzato batch {i//batch_size + 1}: {e}")
                # Fallback per questo batch
                simple_clusters = self.cluster_keywords_simple_custom(batch_keywords, custom_clusters)
                for cluster_name, cluster_keywords in simple_clusters.items():
                    if cluster_name in all_clusters:
                        all_clusters[cluster_name].extend(cluster_keywords)
                    else:
                        all_clusters[cluster_name] = cluster_keywords
        
        # Pulisci cluster vuoti
        final_clusters = {k: v for k, v in all_clusters.items() if v}
        
        return final_clusters

    def cluster_keywords_simple_custom(self, keywords, custom_clusters):
        """Clustering semplice con cluster personalizzati (fallback)"""
        clusters = {}
        
        # Inizializza cluster personalizzati
        for cluster_name in custom_clusters:
            clusters[cluster_name] = []
        
        unassigned_keywords = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            assigned = False
            
            # Prova ad assegnare a cluster personalizzati
            for cluster_name in custom_clusters:
                cluster_words = cluster_name.lower().split()
                if any(word in keyword_lower or keyword_lower in word for word in cluster_words):
                    clusters[cluster_name].append(keyword)
                    assigned = True
                    break
            
            if not assigned:
                unassigned_keywords.append(keyword)
        
        # Raggruppa keyword non assegnate
        if unassigned_keywords:
            auto_clusters = self.cluster_keywords_simple(unassigned_keywords)
            clusters.update(auto_clusters)
        
        # Rimuovi cluster vuoti
        final_clusters = {k: v for k, v in clusters.items() if v}
        
        return final_clusters

    def parse_clustering_response_custom(self, response_text, custom_clusters):
        """Parse della risposta di clustering personalizzato"""
        clusters = {}
        current_cluster = None
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Cluster:'):
                current_cluster = line.replace('Cluster:', '').strip()
                clusters[current_cluster] = []
            elif line.startswith('-') and current_cluster:
                keyword = line.replace('-', '').strip()
                if keyword:
                    clusters[current_cluster].append(keyword)
        
        # Per cluster personalizzati, accetta anche cluster con meno di 5 keyword
        # ma per quelli nuovi mantieni il minimo
        valid_clusters = {}
        small_keywords = []
        
        for cluster_name, keywords in clusters.items():
            if cluster_name in custom_clusters:
                # Cluster personalizzati: accetta qualsiasi size
                valid_clusters[cluster_name] = keywords
            elif len(keywords) >= 3:
                # Cluster nuovi: minimo 3 keyword
                valid_clusters[cluster_name] = keywords
            else:
                small_keywords.extend(keywords)
        
        if small_keywords:
            if "Generale" in valid_clusters:
                valid_clusters["Generale"].extend(small_keywords)
            else:
                valid_clusters["Generale"] = small_keywords
        
        return valid_clusters

    def cluster_keywords_semantic(self, keywords):
        """Clusterizza le keyword per gruppi semantici usando OpenAI"""
        if not self.client or not self.use_ai:
            return self.cluster_keywords_simple(keywords)
        
        batch_size = 50
        all_clusters = {}
        
        for i in range(0, len(keywords), batch_size):
            batch_keywords = keywords[i:i+batch_size]
            
            prompt = f"""Ruolo: Esperto di analisi semantica
Capacità: Possiedi competenze approfondite in linguistica computazionale, analisi semantica e clustering di parole chiave.

Compito: Clusterizza il seguente elenco di keyword raggruppando quelle appartenenti allo stesso gruppo semantico. Ogni cluster deve contenere ALMENO 5 keyword per essere valido.

Elenco keyword da analizzare:
{chr(10).join([f"- {kw}" for kw in batch_keywords])}

Istruzioni:
1. Raggruppa le keyword per similarità semantica, significato e contesto d'uso
2. Ogni cluster deve avere almeno 5 keyword
3. Se una keyword non ha abbastanza correlate, inseriscila nel cluster "Generale"
4. Dai un nome descrittivo a ogni cluster

Formato di risposta:
Cluster: [Nome Cluster]
- keyword1
- keyword2
- keyword3
- keyword4
- keyword5

Cluster: [Nome Cluster 2]
- keyword6
- keyword7
[etc...]"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                
                response_text = response.choices[0].message.content.strip()
                clusters = self.parse_clustering_response(response_text)
                all_clusters.update(clusters)
                
            except Exception as e:
                st.warning(f"Errore clustering OpenAI batch {i//batch_size + 1}: {e}")
                simple_clusters = self.cluster_keywords_simple(batch_keywords)
                all_clusters.update(simple_clusters)
        
        return all_clusters

    def cluster_keywords_simple(self, keywords):
        """Clustering semplice basato su parole comuni (fallback)"""
        clusters = defaultdict(list)
        
        for keyword in keywords:
            words = keyword.lower().split()
            main_word = words[0] if words else keyword
            
            assigned = False
            for cluster_name in clusters:
                if any(word in cluster_name.lower() or cluster_name.lower() in word for word in words):
                    clusters[cluster_name].append(keyword)
                    assigned = True
                    break
            
            if not assigned:
                clusters[f"Cluster {main_word.capitalize()}"].append(keyword)
        
        final_clusters = {}
        small_clusters = []
        
        for cluster_name, cluster_keywords in clusters.items():
            if len(cluster_keywords) >= 5:
                final_clusters[cluster_name] = cluster_keywords
            else:
                small_clusters.extend(cluster_keywords)
        
        if small_clusters:
            final_clusters["Generale"] = small_clusters
        
        return final_clusters

    def parse_clustering_response(self, response_text):
        """Parse della risposta di clustering da OpenAI"""
        clusters = {}
        current_cluster = None
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Cluster:'):
                current_cluster = line.replace('Cluster:', '').strip()
                clusters[current_cluster] = []
            elif line.startswith('-') and current_cluster:
                keyword = line.replace('-', '').strip()
                if keyword:
                    clusters[current_cluster].append(keyword)
        
        valid_clusters = {}
        small_keywords = []
        
        for cluster_name, keywords in clusters.items():
            if len(keywords) >= 5:
                valid_clusters[cluster_name] = keywords
            else:
                small_keywords.extend(keywords)
        
        if small_keywords:
            valid_clusters["Generale"] = small_keywords
        
        return valid_clusters

    def parse_results(self, data, query):
        """Analizza i risultati SERP con classificazione ottimizzata"""
        domain_page_types = defaultdict(lambda: defaultdict(int))
        domain_occurences = defaultdict(int)
        query_page_types = defaultdict(list)
        paa_questions = []
        related_queries = []
        paa_to_queries = defaultdict(set)
        related_to_queries = defaultdict(set)
        paa_to_domains = defaultdict(set)

        pages_to_classify = []
        pages_info = []
        
        if "organic" in data:
            for result in data["organic"]:
                domain = urlparse(result["link"]).netloc
                url = result["link"]
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                page_type = self.classify_page_type_rule_based(url, title, snippet)
                
                if page_type:
                    domain_page_types[domain][page_type] += 1
                    domain_occurences[domain] += 1
                    query_page_types[query].append(page_type)
                else:
                    pages_to_classify.append((url, title, snippet))
                    pages_info.append((domain, url, title, snippet))

        if pages_to_classify and self.use_ai:
            batch_results = self.classify_batch_openai(pages_to_classify)
            
            for domain, url, title, snippet in pages_info:
                cache_key = f"{url}_{title}"
                page_type = batch_results.get(cache_key, "Altro")
                
                domain_page_types[domain][page_type] += 1
                domain_occurences[domain] += 1
                query_page_types[query].append(page_type)
        elif pages_to_classify and not self.use_ai:
            for domain, url, title, snippet in pages_info:
                page_type = "Altro"
                domain_page_types[domain][page_type] += 1
                domain_occurences[domain] += 1
                query_page_types[query].append(page_type)

        if "peopleAlsoAsk" in data:
            for paa in data["peopleAlsoAsk"]:
                paa_text = paa["question"]
                paa_questions.append(paa_text)
                paa_to_queries[paa_text].add(query)
                paa_to_domains[paa_text].update([domain for domain in domain_page_types.keys()])

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
                           paa_to_queries, related_to_queries, paa_to_domains, keyword_clusters=None):
        """Crea il report Excel"""
        
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

        domains_df = pd.DataFrame(domains_counter.items(), columns=["Dominio", "Occorrenze"])
        total_queries = sum(domains_counter.values())
        domains_df["% Presenza"] = (domains_df["Occorrenze"] / total_queries * 100).round(2)

        query_page_type_data = []
        for query, page_types in query_page_types.items():
            for page_type, count in Counter(page_types).items():
                query_page_type_data.append({
                    "Query": query, 
                    "Tipologia Pagina": page_type, 
                    "Occorrenze": count
                })
        query_page_type_df = pd.DataFrame(query_page_type_data)

        paa_df = pd.DataFrame(paa_questions, columns=["People Also Ask"])
        paa_df["Keyword che lo attivano"] = paa_df["People Also Ask"].map(
            lambda x: ", ".join(paa_to_queries[x])
        )

        related_df = pd.DataFrame(related_queries, columns=["Related Query"])
        related_df["Keyword che lo attivano"] = related_df["Related Query"].map(
            lambda x: ", ".join(related_to_queries[x])
        )

        page_type_df = pd.DataFrame(page_type_counter.items(), 
                                  columns=["Tipologia Pagina", "Occorrenze"])

        clustering_df = pd.DataFrame()
        if keyword_clusters:
            clustering_data = []
            for cluster_name, keywords in keyword_clusters.items():
                for keyword in keywords:
                    clustering_data.append({
                        "Cluster": cluster_name,
                        "Keyword": keyword
                    })
            clustering_df = pd.DataFrame(clustering_data)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            domains_df.to_excel(writer, sheet_name="Top Domains", index=False)
            page_type_df.to_excel(writer, sheet_name="Tipologie di Pagine", index=False)
            domain_page_types_df.to_excel(writer, sheet_name="Competitor e Tipologie", index=False)
            query_page_type_df.to_excel(writer, sheet_name="Tipologie per Query", index=False)
            paa_df.to_excel(writer, sheet_name="People Also Ask", index=False)
            related_df.to_excel(writer, sheet_name="Related Queries", index=False)
            if not clustering_df.empty:
                clustering_df.to_excel(writer, sheet_name="Keyword Clustering", index=False)

        return output.getvalue(), domains_df, page_type_df, domain_page_types_df, clustering_df

def save_checkpoint(data, checkpoint_name):
    """Salva un checkpoint dei dati"""
    try:
        with open(f"{checkpoint_name}.pkl", 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Errore nel salvataggio checkpoint: {e}")
        return False

def load_checkpoint(checkpoint_name):
    """Carica un checkpoint dei dati"""
    try:
        if os.path.exists(f"{checkpoint_name}.pkl"):
            with open(f"{checkpoint_name}.pkl", 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Errore nel caricamento checkpoint: {e}")
    return None

def process_batch_queries(analyzer, queries_batch, country, language, num_results, 
                         all_domains, query_page_types, domain_page_types, 
                         domain_occurences, paa_questions, related_queries,
                         paa_to_queries, related_to_queries, paa_to_domains):
    """Processa un batch di query"""
    for query in queries_batch:
        results = analyzer.fetch_serp_results(query, country, language, num_results)
        
        if results:
            (domain_page_types_query, domain_occurences_query, query_page_types_query,
             paa_questions_query, related_queries_query, paa_to_queries_query,
             related_to_queries_query, paa_to_domains_query) = analyzer.parse_results(results, query)
            
            # Aggiorna i dati globali
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
            
            # Libera memoria
            del results
            gc.collect()

def main():
    st.markdown('<h1 class="main-header">🔍 SERP Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Inizializza session state per checkpoint
    if 'checkpoint_data' not in st.session_state:
        st.session_state.checkpoint_data = None
    if 'processed_queries' not in st.session_state:
        st.session_state.processed_queries = set()

    st.sidebar.header("⚙️ Configurazione")
    
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
    
    st.sidebar.subheader("⚡ Opzioni Velocità")
    use_ai_classification = st.sidebar.checkbox(
        "Usa AI per classificazione avanzata",
        value=True,
        help="Disabilita per analisi ultra-veloce (solo regole)"
    )
    
    enable_keyword_clustering = st.sidebar.checkbox(
        "Abilita clustering semantico keyword",
        value=True,
        help="Raggruppa le keyword per gruppi semantici"
    )
    
    batch_size = st.sidebar.slider(
        "Dimensione batch AI",
        min_value=1,
        max_value=10,
        value=5,
        help="Pagine da classificare insieme (più alto = più veloce)"
    ) if use_ai_classification else 1

    # Opzioni avanzate per grandi volumi
    st.sidebar.subheader("🔧 Opzioni Avanzate")
    
    enable_checkpoints = st.sidebar.checkbox(
        "Abilita checkpoint automatici",
        value=True,
        help="Salva progressi ogni N query per recupero in caso di errori"
    )
    
    checkpoint_frequency = st.sidebar.number_input(
        "Frequenza checkpoint (query)",
        min_value=10,
        max_value=100,
        value=50,
        help="Salva checkpoint ogni N query processate"
    ) if enable_checkpoints else 50
    
    batch_processing = st.sidebar.checkbox(
        "Elaborazione a batch",
        value=True,
        help="Processa query in batch per migliore gestione memoria"
    )
    
    queries_per_batch = st.sidebar.number_input(
        "Query per batch",
        min_value=10,
        max_value=100,
        value=25,
        help="Numero di query da processare per batch"
    ) if batch_processing else 25

    # Gestione checkpoint esistenti
    if enable_checkpoints:
        st.sidebar.subheader("📁 Checkpoint")
        checkpoint_files = [f for f in os.listdir('.') if f.endswith('.pkl') and f.startswith('serp_checkpoint_')]
        
        if checkpoint_files:
            selected_checkpoint = st.sidebar.selectbox(
                "Carica checkpoint esistente",
                ["Nessuno"] + checkpoint_files
            )
            
            if selected_checkpoint != "Nessuno" and st.sidebar.button("Carica Checkpoint"):
                checkpoint_data = load_checkpoint(selected_checkpoint.replace('.pkl', ''))
                if checkpoint_data:
                    st.session_state.checkpoint_data = checkpoint_data
                    st.success(f"✅ Checkpoint caricato: {len(checkpoint_data.get('processed_queries', []))} query già processate")

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

    if enable_keyword_clustering:
        st.header("🏗️ Cluster Personalizzati (Opzionale)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_clusters_input = st.text_area(
                "Nomi delle pagine/categorie del tuo sito (una per riga)",
                height=150,
                placeholder="Inserisci i nomi delle tue pagine principali...\nUna per ogni riga\n\nEsempio:\nServizi SEO\nCorsi Online\nConsulenza Marketing\nBlog Aziendale\nChi Siamo"
            )
        
        with col2:
            st.markdown("### 🎯 Cluster Strategici")
            st.info("""
            • Nomi delle tue pagine principali
            • Categorie del sito
            • Servizi offerti
            • Sezioni importanti
            • Lascia vuoto per clustering automatico
            """)

    if st.button("🚀 Avvia Analisi", type="primary", use_container_width=True):
        if use_ai_classification and (not serper_api_key or not openai_api_key):
            st.error("⚠️ Inserisci entrambe le API keys per l'analisi AI!")
            return
        elif not use_ai_classification and not serper_api_key:
            st.error("⚠️ Inserisci almeno la Serper API key!")
            return
        
        if not queries_input.strip():
            st.error("⚠️ Inserisci almeno una query!")
            return

        queries = [q.strip() for q in queries_input.strip().split('\n') if q.strip()]
        
        if len(queries) > 1000:
            st.error("⚠️ Massimo 1000 query per volta!")
            return

        # Rimuovi query già processate se caricate da checkpoint
        if st.session_state.checkpoint_data:
            processed = st.session_state.checkpoint_data.get('processed_queries', set())
            queries = [q for q in queries if q not in processed]
            st.info(f"📊 Riprendendo analisi: {len(queries)} query rimanenti da processare")

        custom_clusters = []
        if enable_keyword_clustering and 'custom_clusters_input' in locals() and custom_clusters_input.strip():
            custom_clusters = [c.strip() for c in custom_clusters_input.strip().split('\n') if c.strip()]

        if use_ai_classification:
            analyzer = SERPAnalyzer(serper_api_key, openai_api_key)
            st.info("🤖 Modalità AI attivata - Classificazione avanzata delle pagine")
        else:
            analyzer = SERPAnalyzer(serper_api_key, "dummy")
            st.info("⚡ Modalità Veloce attivata - Solo classificazione basata su regole")
        
        analyzer.use_ai = use_ai_classification
        analyzer.batch_size = batch_size
        
        # Inizializza o carica dati da checkpoint
        if st.session_state.checkpoint_data:
            all_domains = st.session_state.checkpoint_data.get('all_domains', [])
            query_page_types = st.session_state.checkpoint_data.get('query_page_types', defaultdict(list))
            domain_page_types = st.session_state.checkpoint_data.get('domain_page_types', defaultdict(lambda: defaultdict(int)))
            domain_occurences = st.session_state.checkpoint_data.get('domain_occurences', defaultdict(int))
            paa_questions = st.session_state.checkpoint_data.get('paa_questions', [])
            related_queries = st.session_state.checkpoint_data.get('related_queries', [])
            paa_to_queries = st.session_state.checkpoint_data.get('paa_to_queries', defaultdict(set))
            related_to_queries = st.session_state.checkpoint_data.get('related_to_queries', defaultdict(set))
            paa_to_domains = st.session_state.checkpoint_data.get('paa_to_domains', defaultdict(set))
            processed_queries = st.session_state.checkpoint_data.get('processed_queries', set())
            keyword_clusters = st.session_state.checkpoint_data.get('keyword_clusters', {})
        else:
            all_domains = []
            query_page_types = defaultdict(list)
            domain_page_types = defaultdict(lambda: defaultdict(int))
            domain_occurences = defaultdict(int)
            paa_questions = []
            related_queries = []
            paa_to_queries = defaultdict(set)
            related_to_queries = defaultdict(set)
            paa_to_domains = defaultdict(set)
            processed_queries = set()
            keyword_clusters = {}
        
        # Clustering keyword (se non già fatto)
        if enable_keyword_clustering and not keyword_clusters:
            status_text = st.empty()
            
            if custom_clusters:
                status_text.text(f"🏗️ Clustering con {len(custom_clusters)} cluster personalizzati...")
                try:
                    keyword_clusters = analyzer.cluster_keywords_with_custom(queries, custom_clusters)
                    st.success(f"✅ Cluster creati: {len(keyword_clusters)} (inclusi {len([k for k in keyword_clusters.keys() if k in custom_clusters])} personalizzati)")
                except Exception as e:
                    st.warning(f"Errore durante il clustering personalizzato: {e}")
                    keyword_clusters = {}
            else:
                status_text.text("🧠 Clustering semantico automatico delle keyword...")
                try:
                    keyword_clusters = analyzer.cluster_keywords_semantic(queries)
                    st.success(f"✅ Identificati {len(keyword_clusters)} cluster semantici!")
                except Exception as e:
                    st.warning(f"Errore durante il clustering: {e}")
                    keyword_clusters = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Container per metriche in tempo reale
        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            metric_queries = col1.empty()
            metric_domains = col2.empty()
            metric_paa = col3.empty()
            metric_errors = col4.empty()
        
        total_queries = len(queries)
        queries_processed = 0
        errors_count = 0
        
        try:
            # Processa query in batch
            if batch_processing:
                for batch_start in range(0, len(queries), queries_per_batch):
                    batch_end = min(batch_start + queries_per_batch, len(queries))
                    batch_queries = queries[batch_start:batch_end]
                    
                    status_text.text(f"🔄 Processando batch {batch_start//queries_per_batch + 1}/{(len(queries)-1)//queries_per_batch + 1}")
                    
                    # Processa il batch
                    for i, query in enumerate(batch_queries):
                        if query in processed_queries:
                            continue
                            
                        status_text.text(f"🔍 Analizzando: {query} ({queries_processed+1}/{total_queries})")
                        
                        try:
                            results = analyzer.fetch_serp_results(query, country, language, num_results)
                            
                            if results:
                                (domain_page_types_query, domain_occurences_query, query_page_types_query,
                                 paa_questions_query, related_queries_query, paa_to_queries_query,
                                 related_to_queries_query, paa_to_domains_query) = analyzer.parse_results(results, query)
                                
                                # Aggiorna i dati
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
                                
                                processed_queries.add(query)
                                queries_processed += 1
                            else:
                                errors_count += 1
                        
                        except Exception as e:
                            st.warning(f"Errore processando '{query}': {e}")
                            errors_count += 1
                        
                        # Aggiorna metriche
                        metric_queries.metric("Query Processate", f"{queries_processed}/{total_queries}")
                        metric_domains.metric("Domini Trovati", len(set(all_domains)))
                        metric_paa.metric("PAA Questions", len(set(paa_questions)))
                        metric_errors.metric("Errori", errors_count)
                        
                        progress_bar.progress(queries_processed / total_queries)
                        
                        # Salva checkpoint se necessario
                        if enable_checkpoints and queries_processed % checkpoint_frequency == 0:
                            checkpoint_data = {
                                'all_domains': all_domains,
                                'query_page_types': dict(query_page_types),
                                'domain_page_types': dict(domain_page_types),
                                'domain_occurences': dict(domain_occurences),
                                'paa_questions': paa_questions,
                                'related_queries': related_queries,
                                'paa_to_queries': dict(paa_to_queries),
                                'related_to_queries': dict(related_to_queries),
                                'paa_to_domains': dict(paa_to_domains),
                                'processed_queries': processed_queries,
                                'keyword_clusters': keyword_clusters
                            }
                            
                            checkpoint_name = f"serp_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            if save_checkpoint(checkpoint_data, checkpoint_name):
                                status_text.text(f"💾 Checkpoint salvato: {checkpoint_name}")
                        
                        # Attesa tra query
                        sleep_time = 0.5 if not use_ai_classification else 1.0
                        time.sleep(sleep_time)
                    
                    # Libera memoria dopo ogni batch
                    gc.collect()
            
            else:
                # Modalità standard (non batch)
                for i, query in enumerate(queries):
                    if query in processed_queries:
                        continue
                        
                    status_text.text(f"🔍 Analizzando: {query} ({i+1}/{len(queries)})")
                    
                    try:
                        results = analyzer.fetch_serp_results(query, country, language, num_results)
                        
                        if results:
                            (domain_page_types_query, domain_occurences_query, query_page_types_query,
                             paa_questions_query, related_queries_query, paa_to_queries_query,
                             related_to_queries_query, paa_to_domains_query) = analyzer.parse_results(results, query)
                            
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
                            
                            processed_queries.add(query)
                            queries_processed += 1
                    
                    except Exception as e:
                        st.warning(f"Errore processando '{query}': {e}")
                        errors_count += 1
                    
                    progress_bar.progress((i + 1) / len(queries))
                    
                    sleep_time = 0.5 if not use_ai_classification else 1.0
                    time.sleep(sleep_time)

            status_text.text("✅ Analisi completata! Generazione report...")

            # Generazione report finale
            domains_counter = Counter(all_domains)
            excel_data, domains_df, page_type_df, domain_page_types_df, clustering_df = analyzer.create_excel_report(
                domains_counter, domain_occurences, query_page_types, domain_page_types,
                paa_questions, related_queries, paa_to_queries, related_to_queries, paa_to_domains, keyword_clusters
            )

            status_text.text("📊 Visualizzazione risultati...")

            st.markdown("---")
            st.header("📊 Risultati Analisi")

            # Visualizzazioni
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Top Domini")
                if not domains_df.empty:
                    fig_domains = px.bar(
                        domains_df.head(10), 
                        x="Dominio", 
                        y="Occorrenze",
                        title="Top 10 Domini per Occorrenze"
                    )
                    fig_domains.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_domains, use_container_width=True)

            with col2:
                st.subheader("🏷️ Distribuzione Tipologie")
                if not page_type_df.empty:
                    fig_pie = px.pie(
                        page_type_df, 
                        values="Occorrenze", 
                        names="Tipologia Pagina",
                        title="Tipologie di Pagine"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            if keyword_clusters:
                st.subheader("🧠 Analisi Cluster Semantici")
                
                cluster_sizes = {name: len(keywords) for name, keywords in keyword_clusters.items()}
                cluster_df = pd.DataFrame(list(cluster_sizes.items()), columns=["Cluster", "Numero Keyword"])
                
                fig_clusters = px.bar(
                    cluster_df.sort_values("Numero Keyword", ascending=False),
                    x="Cluster",
                    y="Numero Keyword", 
                    title="Distribuzione Keyword per Cluster"
                )
                fig_clusters.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_clusters, use_container_width=True)

            # Tabelle
            st.subheader("📋 Tabelle Dettagliate")
            
            tabs = ["Top Domini", "Tipologie Pagine", "Competitor Analysis"]
            if keyword_clusters:
                tabs.append("Keyword Clustering")
            
            if len(tabs) == 4:
                tab1, tab2, tab3, tab4 = st.tabs(tabs)
            else:
                tab1, tab2, tab3 = st.tabs(tabs)
                tab4 = None
            
            with tab1:
                st.dataframe(domains_df, use_container_width=True)
            
            with tab2:
                st.dataframe(page_type_df, use_container_width=True)
            
            with tab3:
                st.dataframe(domain_page_types_df, use_container_width=True)
            
            if tab4 and not clustering_df.empty:
                with tab4:
                    st.dataframe(clustering_df, use_container_width=True)

            # Download
            st.subheader("💾 Download Report")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Scarica Report Excel Completo",
                    data=excel_data,
                    file_name=f"serp_analysis_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                if enable_checkpoints:
                    # Salva checkpoint finale
                    final_checkpoint_data = {
                        'all_domains': all_domains,
                        'query_page_types': dict(query_page_types),
                        'domain_page_types': dict(domain_page_types),
                        'domain_occurences': dict(domain_occurences),
                        'paa_questions': paa_questions,
                        'related_queries': related_queries,
                        'paa_to_queries': dict(paa_to_queries),
                        'related_to_queries': dict(related_to_queries),
                        'paa_to_domains': dict(paa_to_domains),
                        'processed_queries': processed_queries,
                        'keyword_clusters': keyword_clusters
                    }
                    
                    checkpoint_name = f"serp_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    if st.button("💾 Salva Risultati come Checkpoint"):
                        if save_checkpoint(final_checkpoint_data, checkpoint_name):
                            st.success(f"✅ Checkpoint finale salvato: {checkpoint_name}.pkl")

            progress_bar.empty()
            status_text.text("🎉 Analisi completata con successo!")
            
            # Pulisci checkpoint temporanei se completato con successo
            if enable_checkpoints:
                st.info("💡 Puoi eliminare i checkpoint temporanei se l'analisi è stata completata con successo.")
        
        except Exception as e:
            st.error(f"❌ Errore durante l'analisi: {e}")
            
            # Salva checkpoint di emergenza
            if enable_checkpoints:
                emergency_checkpoint_data = {
                    'all_domains': all_domains,
                    'query_page_types': dict(query_page_types),
                    'domain_page_types': dict(domain_page_types),
                    'domain_occurences': dict(domain_occurences),
                    'paa_questions': paa_questions,
                    'related_queries': related_queries,
                    'paa_to_queries': dict(paa_to_queries),
                    'related_to_queries': dict(related_to_queries),
                    'paa_to_domains': dict(paa_to_domains),
                    'processed_queries': processed_queries,
                    'keyword_clusters': keyword_clusters
                }
                
                emergency_checkpoint_name = f"serp_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if save_checkpoint(emergency_checkpoint_data, emergency_checkpoint_name):
                    st.warning(f"⚠️ Checkpoint di emergenza salvato: {emergency_checkpoint_name}.pkl")
                    st.info(f"Processate {len(processed_queries)} query prima dell'errore. Puoi riprendere l'analisi caricando il checkpoint.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>SEO SERP Analyzer PRO v2.0 - Ottimizzato per grandi volumi - Sviluppato da Daniele Pisciottano e il suo amico Claude 🦕</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
