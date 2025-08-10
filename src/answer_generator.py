import google.generativeai as genai
import json
import os
import time
from typing import Dict, Any, Optional

# --- Importazioni dei Componenti ---
# Assumiamo che siano nella stessa cartella 'src'
from query_analyzer import analyze_user_question
from knowledge_retriever import KnowledgeRetriever

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
NEO4J_DATABASE = "test"
ALL_CHUNKS_FILE_PATH = 'data/processed/processed_chunks_toc_enhanced.json'

# --- Variabile Globale per l'Istanza del Retriever ---
_retriever_instance = None

def get_retriever_instance() -> Optional[KnowledgeRetriever]:
    """
    Funzione Singleton-like per creare e restituire una singola istanza del retriever.
    """
    global _retriever_instance
    if _retriever_instance is None:
        print("Inizializzazione del Knowledge Retriever...")
        _retriever_instance = KnowledgeRetriever(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, ALL_CHUNKS_FILE_PATH
        )
        if not _retriever_instance.driver:
            print("Inizializzazione del Retriever fallita.")
            _retriever_instance = None # Resetta se fallisce
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

#TODO Check if log are needed
def save_interaction_log(user_question, context, answer, log_file="interaction_log.json"):
    pass

def run_qa_pipeline(user_question: str) -> Dict[str, Any]:
    """
    Orchestra l'intera pipeline di Q&A per una singola domanda.
    Questa è la funzione "core" che restituisce l'output strutturato.
    """
    retriever = get_retriever_instance()
    if not retriever:
        # Costruisci un output di errore standard
        return {
            "question": user_question,
            "answer": "Mi dispiace, non riesco a connettermi alla mia base di conoscenza in questo momento.",
            "contexts": [],
            "error": "Knowledge Retriever non inizializzato."
        }

    # 1. Analisi della domanda
    analysis = analyze_user_question(user_question)
    
    # 2. Recupero della conoscenza
    retrieved_context = retriever.retrieve_knowledge(analysis, retrieve_text=True)
    
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
        "retrieved_context": retrieved_context, # Manteniamo anche il contesto grezzo per debug
        "error": None
    }
    
    save_interaction_log(user_question, retrieved_context, final_answer)
    
    return result_package


def answer_user_question(user_question: str) -> str:
    """
    Funzione wrapper per l'utente finale. Restituisce solo la stringa della risposta.
    """
    result = run_qa_pipeline(user_question)
    return result.get("answer", "Si è verificato un errore inaspettato.")


# --- Esempio di Utilizzo e Test ---

if __name__ == "__main__":
    import atexit
    
    # Registra la funzione di cleanup per chiudere la connessione quando lo script termina
    atexit.register(close_retriever_connection)
    
    # Test della pipeline completa
    test_question = "Cos'è il DGUE e dove si usa?"
    
    print(f"--- TEST PIPELINE COMPLETA ---")
    print(f"Domanda: {test_question}\n")
    
    # La funzione run_qa_pipeline si occuperà di inizializzare il retriever se necessario
    full_output = run_qa_pipeline(test_question)
    
    print("\n--- RISULTATO STRUTTURATO (per valutazione) ---")
    print(json.dumps(full_output, indent=2, ensure_ascii=False))
    
    print("\n--- RISPOSTA FINALE (per utente) ---")
    print(full_output.get("answer"))

    # # Test interattivo
    # print("\n--- MODALITÀ INTERATTIVA ---")
    # while True:
    #     user_input = input("\nInserisci una domanda (o 'quit' per uscire): ")
    #     if user_input.lower() in ['quit', 'esci', 'exit']:
    #         break
    #     response = answer_user_question(user_input)
    #     print(f"\nRisposta: {response}")