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

LLM_MODEL_SYNTHESIS = "gemini-2.0-flash-exp"

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
    """
    Configura dinamicamente la pipeline in base al tipo di dati.
    
    Args:
        use_raw_data (bool): True per dati raw, False per dati aggregati
    """
    global _current_data_type, _analyze_function, _KnowledgeRetriever, _retriever_instance
    
    # Reset dell'istanza precedente se diversa
    if _current_data_type != use_raw_data and _retriever_instance:
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
    """
    Restituisce la configurazione Neo4j appropriata per il tipo di dati.
    
    Args:
        use_raw_data (bool): True per dati raw, False per dati aggregati
        
    Returns:
        Dict con configurazione Neo4j
    """
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
    """
    Funzione Singleton-like per creare e restituire una singola istanza del retriever.
    
    Args:
        use_raw_data (bool): True per dati raw, False per dati aggregati
    """
    global _retriever_instance
    
    # Setup della pipeline se necessario
    if _current_data_type != use_raw_data:
        if not setup_pipeline(use_raw_data):
            return None
    
    if _retriever_instance is None:
        print(f"Inizializzazione del Knowledge Retriever ({'raw' if use_raw_data else 'aggregated'})...")
        
        config = get_neo4j_config(use_raw_data)
        _retriever_instance = _KnowledgeRetriever(
            config["uri"], 
            config["user"], 
            config["password"], 
            config["database"], 
            ALL_CHUNKS_FILE_PATH
        )
        
        if not _retriever_instance.driver:
            print("Inizializzazione del Retriever fallita.")
            _retriever_instance = None
    
    return _retriever_instance

def close_retriever_connection():
    """Funzione per chiudere la connessione del retriever alla fine."""
    global _retriever_instance
    if _retriever_instance:
        print("Chiusura connessione del Knowledge Retriever...")
        _retriever_instance.close()
        _retriever_instance = None

def call_gemini_with_retries(prompt, model_name=LLM_MODEL_SYNTHESIS, max_retries=3, delay=5):
    generation_config = {"temperature": 0.2}
    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Errore in call_gemini_with_retries: {e}")
        return ""

def build_answer_generation_prompt(user_question, graph_context, text_context):
    return f"Domanda: {user_question}\nContesto: {graph_context}\n{text_context}\nRisposta:"
    
def validate_context(retrieved_context):
    graph_context = retrieved_context.get("graph_context", "")
    text_context = retrieved_context.get("text_context", "")
    graph_empty = (not graph_context or "Nessuna" in graph_context)
    text_empty = (not text_context or "Nessun" in text_context)
    return not (graph_empty and text_empty)

def save_interaction_log(user_question, context, answer, log_file="interaction_log.json"):
    pass

def run_qa_pipeline(user_question: str, use_raw_data: bool = True) -> Dict[str, Any]:
    """
    Orchestra l'intera pipeline di Q&A per una singola domanda.
    
    Args:
        user_question (str): La domanda dell'utente
        use_raw_data (bool): True per dati raw, False per dati aggregati
    """
    retriever = get_retriever_instance(use_raw_data)
    if not retriever:
        return {
            "question": user_question,
            "answer": "Mi dispiace, non riesco a connettermi alla mia base di conoscenza in questo momento.",
            "contexts": [],
            "error": "Knowledge Retriever non inizializzato."
        }

    # Verifica che la funzione di analisi sia disponibile
    if not _analyze_function:
        return {
            "question": user_question,
            "answer": "Errore nella configurazione del sistema di analisi.",
            "contexts": [],
            "error": "Funzione di analisi non configurata."
        }

    # 1. Analisi della domanda
    analysis = _analyze_function(user_question)
    
    # 2. Recupero della conoscenza
    retrieved_context = retriever.retrieve_knowledge(user_question, retrieve_text=True)
    
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
        "contexts": contexts_list,
        "analysis": analysis,
        "retrieved_context": retrieved_context,
        "data_type": "raw" if use_raw_data else "aggregated",
        "error": None
    }
    
    save_interaction_log(user_question, retrieved_context, final_answer)
    
    return result_package

def answer_user_question(user_question: str, use_raw_data: bool = True) -> str:
    """
    Funzione wrapper per l'utente finale. Restituisce solo la stringa della risposta.
    
    Args:
        user_question (str): La domanda dell'utente
        use_raw_data (bool): True per dati raw, False per dati aggregati
    """
    result = run_qa_pipeline(user_question, use_raw_data)
    return result.get("answer", "Si è verificato un errore inaspettato.")

# --- Esempio di Utilizzo e Test ---
if __name__ == "__main__":
    import atexit
    
    # Registra la funzione di cleanup
    atexit.register(close_retriever_connection)
    
    # Test della pipeline completa con dati raw
    test_question = "Cos'è il DGUE e dove si usa?"
    
    print(f"--- TEST PIPELINE COMPLETA (DATI RAW) ---")
    print(f"Domanda: {test_question}\n")
    
    full_output = run_qa_pipeline(test_question, use_raw_data=True)
    print(json.dumps(full_output, indent=2, ensure_ascii=False))