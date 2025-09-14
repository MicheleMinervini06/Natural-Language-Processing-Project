import google.generativeai as genai
import json
import os
import time
from typing import Dict, Any, Optional

# --- Configurazione Globale ---
try:
    GOOGLE_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    print("API Key di Google Gemini configurata.")
except KeyError:
    print("ERRORE: La variabile d'ambiente GEMINI_API_KEY non è impostata.")

LLM_MODEL_SYNTHESIS = "gemini-2.5-pro"

# Configurazione del Retriever
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password"
ALL_CHUNKS_FILE_PATH = 'data/processed/processed_chunks_toc_enhanced.json'

# --- Variabili Globali per Gestione Dinamica ---
_retriever_instance = None
_current_data_type = None
_analyze_function = None
_KnowledgeRetriever = None

def setup_pipeline(use_raw_data: bool = True):
    """Configura dinamicamente la pipeline in base al tipo di dati."""
    global _current_data_type, _analyze_function, _KnowledgeRetriever, _retriever_instance
    
    if _current_data_type == use_raw_data and _KnowledgeRetriever is not None:
        return True # Già configurato

    if _retriever_instance:
        close_retriever_connection()
    
    _current_data_type = use_raw_data
    
    if use_raw_data:
        print("Configurazione pipeline per dati RAW...")
        try:
            from query_analyzer_rawData import analyze_user_question
            from knowledge_retriever_rawData import KnowledgeRetriever
            _analyze_function = analyze_user_question
            _KnowledgeRetriever = KnowledgeRetriever
            print("Moduli per dati raw importati con successo.")
        except ImportError as e:
            print(f"Errore nell'importazione dei moduli raw: {e}")
            return False
    else:
        print("Configurazione pipeline per dati AGGREGATI...")
        try:
            from query_analyzer import analyze_user_question
            from knowledge_retriever import KnowledgeRetriever
            _analyze_function = analyze_user_question
            _KnowledgeRetriever = KnowledgeRetriever
            print("Moduli per dati aggregati importati con successo.")
        except ImportError as e:
            print(f"Errore nell'importazione dei moduli aggregati: {e}")
            return False
    
    return True

def get_neo4j_config(use_raw_data: bool) -> Dict[str, str]:
    """Restituisce la configurazione Neo4j appropriata."""
    base_config = {
        "uri": NEO4J_URI,
        "user": NEO4J_USER,
        "password": NEO4J_PASSWORD
    }
    if use_raw_data:
        base_config["database"] = "test"
    else:
        base_config["database"] = "testaggregated"
    return base_config

def get_retriever_instance(use_raw_data: bool = True) -> Optional:
    """Funzione Singleton per creare e restituire una singola istanza del retriever."""
    global _retriever_instance
    
    if _current_data_type != use_raw_data or _KnowledgeRetriever is None:
        if not setup_pipeline(use_raw_data):
            return None
    
    if _retriever_instance is None:
        print(f"Inizializzazione del Knowledge Retriever ({'raw' if use_raw_data else 'aggregated'})...")
        config = get_neo4j_config(use_raw_data)
        _retriever_instance = _KnowledgeRetriever(
            config["uri"], config["user"], config["password"], 
            config["database"], ALL_CHUNKS_FILE_PATH
        )
        if not _retriever_instance.driver:
            print("Inizializzazione del Retriever fallita.")
            _retriever_instance = None
    
    return _retriever_instance

def close_retriever_connection():
    """Chiude la connessione del retriever."""
    global _retriever_instance
    if _retriever_instance:
        print("Chiusura connessione del Knowledge Retriever...")
        _retriever_instance.close()
        _retriever_instance = None

def call_gemini_with_retries(prompt, model_name=LLM_MODEL_SYNTHESIS, max_retries=3, delay=5):
    """Chiama Gemini per la sintesi della risposta finale."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Errore in call_gemini_with_retries: {e}")
        return ""

def build_answer_generation_prompt(user_question, graph_context, text_context):
    """Costruisce il prompt per la generazione della risposta finale."""
    return f"""
Sei un assistente esperto sulla piattaforma di e-procurement EmPULIA.
Il tuo compito è rispondere alla domanda dell'utente in modo chiaro, conciso e basandoti ESCLUSIVAMENTE sulle informazioni fornite nel contesto. Non inventare informazioni.

**Domanda dell'Utente:**
{user_question}

**Contesto dal Knowledge Graph (Entità e Relazioni):**
{graph_context}
code
Code
**Contesto dal Testo Originale (Se disponibile):**
{text_context}
code
Code
**Istruzioni per la Risposta:**
1.  Sintetizza le informazioni da entrambi i contesti per formulare una risposta completa.
2.  Se il contesto non contiene informazioni sufficienti, dichiara: "Non ho trovato informazioni specifiche per rispondere alla tua domanda."
3.  Non fare riferimento al "Knowledge Graph" o ai "documenti" nella tua risposta. Parla direttamente all'utente.
4.  Usa un linguaggio professionale e, se utile, elenchi puntati.
5.  DEVI aggiungere alla fine della risposta le fonti usate per generare la risposta, in questo modo:
**Fonti:**
- Fonte 1: [inserisci il nome del documento]
- Fonte 2: [inserisci il nome del documento]
etc.
Trovi queste informazioni nel contesto del testo originale. ASSICURATI di inserire solo i riferimenti a documenti effetivamente usati per la risposta.
**Risposta Finale:**
"""
    
def validate_context(retrieved_context):
    """Verifica se il contesto recuperato è vuoto."""
    graph_context = retrieved_context.get("graph_context", "")
    text_context = retrieved_context.get("text_context", "")
    graph_empty = (not graph_context or "Nessuna" in graph_context)
    text_empty = (not text_context or "Nessun" in text_context)
    return not (graph_empty and text_empty)

def run_qa_pipeline(user_question: str, use_raw_data: bool = True) -> Dict[str, Any]:
    """
    Orchestra l'intera pipeline di Q&A per una singola domanda.
    """
    retriever = get_retriever_instance(use_raw_data)
    if not retriever:
        return {
            "question": user_question,
            "answer": "Mi dispiace, non riesco a connettermi alla mia base di conoscenza.",
            "contexts": [], "error": "Knowledge Retriever non inizializzato."
        }

    if not _analyze_function:
        return {
            "question": user_question,
            "answer": "Errore nella configurazione del sistema di analisi.",
            "contexts": [], "error": "Funzione di analisi non configurata."
        }

    # 1. Analisi della domanda
    analysis = _analyze_function(user_question)
    
    # ### <<< CORREZIONE FONDAMENTALE >>> ###
    # Passiamo il dizionario 'analysis' al retriever, non più la stringa 'user_question'.
    if analysis:
        retrieved_context = retriever.retrieve_knowledge(analysis, retrieve_text=True)
    else:
        # Se l'analisi fallisce, crea un contesto vuoto per evitare errori
        print("L'analisi della domanda ha fallito. Procedo con un contesto vuoto.")
        retrieved_context = {
            "graph_context": "Analisi della domanda fallita.",
            "text_context": ""
        }
    
    # 3. Generazione della risposta
    graph_context = retrieved_context.get("graph_context", "")
    text_context = retrieved_context.get("text_context", "")
    
    if not validate_context(retrieved_context):
        final_answer = "Non ho trovato informazioni specifiche per rispondere alla tua domanda."
    else:
        prompt = build_answer_generation_prompt(user_question, graph_context, text_context)
        final_answer = call_gemini_with_retries(prompt)
        if not final_answer:
            final_answer = "Si è verificato un problema nella generazione della risposta."

    # 4. Assemblaggio output per valutazione/logging
    contexts_list = []
    if graph_context and "Nessuna" not in graph_context:
        contexts_list.append(graph_context)
    if text_context and "Nessun" not in text_context:
        contexts_list.append(text_context)
        
    result_package = {
        "question": user_question,
        "answer": final_answer,
        "contexts": contexts_list
    }
    
    return result_package

def answer_user_question(user_question: str, use_raw_data: bool = True) -> str:
    """Funzione wrapper per l'utente finale. Restituisce solo la stringa della risposta."""
    result = run_qa_pipeline(user_question, use_raw_data)
    return result.get("answer", "Si è verificato un errore inaspettato.")

# --- Esempio di Utilizzo e Test ---
if __name__ == "__main__":
    import atexit
    atexit.register(close_retriever_connection)
    
    #test_question = "Sono un utente e ho appena ricevuto le nuove credenziali. Qual è il primo passo che devo compiere dopo aver fatto l'accesso?"
    test_question = "Come faccio a rinnovare la mia patente di guida?"
    print(f"--- TEST PIPELINE COMPLETA (DATI RAW) ---")
    print(f"Domanda: {test_question}\n")
    
    full_output = run_qa_pipeline(test_question, use_raw_data=True)
    
    # Stampa un output più leggibile per il test
    print("\n--- RISULTATO DEL TEST ---")
    print(f"DOMANDA: {full_output.get('question')}")
    print(f"\nRISPOSTA:\n{full_output.get('answer')}")
    print("\nCONTESTI USATI:")
    #for i, ctx in enumerate(full_output.get('contexts', [])):
     #   print(f"--- Contesto {i+1} ---\n{ctx}\n--------------------")