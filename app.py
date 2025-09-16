import streamlit as st
import requests
import urllib.parse
import re

# --- Configurazione dell'App ---
st.set_page_config(
    page_title="Assistente EmPULIA",
    page_icon="🤖",
    layout="wide"
)

# URL del backend FastAPI e della cartella dei documenti
BACKEND_URL = "http://127.0.0.1:8000/ask"
DOCS_BASE_URL = "http://127.0.0.1:8000/docs/"

# --- Funzione Helper per Processare le Fonti ---
def process_answer_for_links(answer_text: str) -> str:
    """
    Trova la sezione 'Fonti' nella risposta e converte i nomi dei file in link Markdown.
    Funziona anche se la sezione non è preceduta da --- o **.
    """
    # Cerca la sezione Fonti (accetta varianti)
    match = re.search(r'(\*\*Fonti:\*\*|Fonti:)\s*\n([\s\S]+)', answer_text, re.IGNORECASE)
    if not match:
        return answer_text

    main_answer = answer_text[:match.start(1)]
    sources_text = match.group(2)

    # Prendi solo le righe che sembrano fonti (fino a una riga vuota o fine testo)
    sources_lines = []
    for line in sources_text.strip().split('\n'):
        file_name = line.strip().lstrip('*- ').strip()
        if not file_name or file_name.lower().startswith("informazioni aggiuntive"):
            break
        sources_lines.append(file_name)

    processed_sources = []
    for file_name in sources_lines:
        if file_name:
            safe_file_name = urllib.parse.quote(file_name)
            link_url = f"{DOCS_BASE_URL}{safe_file_name}"
            processed_sources.append(f"* [{file_name}]({link_url})")

    if processed_sources:
        return main_answer + "**Fonti:**\n" + "\n".join(processed_sources)
    else:
        return answer_text

# --- Interfaccia Utente ---
st.title("🤖 Assistente AI per la Piattaforma EmPULIA")
st.caption("Fai una domanda sulla documentazione tecnica e ricevi una risposta accurata con le relative fonti.")

# Inizializza la cronologia della chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra i messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input dell'utente
if prompt := st.chat_input("Come posso creare una commissione di gara?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sto cercando la risposta nei documenti..."):
            try:
                payload = {"question": prompt}
                response = requests.post(BACKEND_URL, json=payload, timeout=300)
                response.raise_for_status()
                
                result = response.json()
                raw_answer = result.get("answer", "Non ho ricevuto una risposta valida dal sistema.")
                
                # ### <<< CORREZIONE FONDAMENTALE >>> ###
                # Processa la risposta ricevuta per trasformare il testo delle fonti in link
                answer_with_links = process_answer_for_links(raw_answer)
                
                st.markdown(answer_with_links, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer_with_links})

            except requests.exceptions.ConnectionError:
                error_message = "**Errore di Connessione**\n\nImpossibile raggiungere il sistema. Assicurati che il server backend (`main.py`) sia in esecuzione."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            except Exception as e:
                error_message = f"**Si è verificato un errore inaspettato:** {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})