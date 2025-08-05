import google.generativeai as genai
import json
import os
import time
from typing import Dict, Any, Optional

# Import del knowledge retriever
try:
    from knowledge_retriever import retrieve_knowledge
    KNOWLEDGE_RETRIEVER_AVAILABLE = True
except ImportError:
    print("Avviso: knowledge_retriever non disponibile. Alcune funzionalità saranno limitate.")
    KNOWLEDGE_RETRIEVER_AVAILABLE = False

# --- Configurazione Globale ---

# Carica la chiave API dalla variabile d'ambiente.
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    print("API Key di Google Gemini per Answer Generator configurata.")
except KeyError:
    print("ERRORE: La variabile d'ambiente GOOGLE_API_KEY non è impostata.")
    print("Imposta la variabile d'ambiente: set GOOGLE_API_KEY=your_api_key")

LLM_MODEL_SYNTHESIS = "gemini-2.0-flash-exp"  # Modello più aggiornato e performante

# --- Funzioni Core per Gemini ---
def call_gemini_with_retries(prompt: str, model_name: str = LLM_MODEL_SYNTHESIS, max_retries: int = 3, delay: int = 5) -> str:
    """
    Funzione robusta per chiamare l'API Gemini con gestione dei tentativi e backoff esponenziale.
    Restituisce la risposta del modello come stringa.
    """
    generation_config = {
        "temperature": 0.2,  # Leggermente più alta di 0 per una risposta più naturale, ma ancora molto fattuale.
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,  # Limite esplicito per evitare risposte troppo lunghe
    }
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name, 
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    except Exception as e:
        print(f"Errore nella configurazione del modello Gemini: {e}")
        return ""
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            
            # Controllo se la risposta è stata bloccata dai filtri di sicurezza
            if response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
                print("Risposta bloccata dai filtri di sicurezza di Gemini.")
                return "Mi dispiace, non posso fornire una risposta a questa domanda per motivi di sicurezza."
            
            # Controllo se la risposta è completa
            if response.candidates and response.candidates[0].finish_reason.name == "MAX_TOKENS":
                print("Avviso: Risposta troncata per raggiungimento del limite di token.")
            
            return response.text.strip() if response.text else ""

        except genai.types.StopCandidateException as e:
            print(f"Contenuto bloccato dai filtri di sicurezza: {e}")
            return "Mi dispiace, non posso fornire una risposta a questa domanda per motivi di sicurezza."
            
        except genai.types.GenerationException as e:
            print(f"Errore di generazione Gemini (tentativo {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                current_delay = delay * (2 ** attempt)
                print(f"Attendo {current_delay} secondi prima di ritentare...")
                time.sleep(current_delay)
            else:
                print("Massimo numero di tentativi raggiunto per errore di generazione.")
                return ""
                
        except Exception as e:
            print(f"Errore generico durante la chiamata a Gemini (tentativo {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                current_delay = delay * (2 ** attempt)
                print(f"Attendo {current_delay} secondi prima di ritentare...")
                time.sleep(current_delay)
            else:
                print("Massimo numero di tentativi raggiunto.")
                return ""
    
    return ""

def build_answer_generation_prompt(user_question: str, graph_context: str, text_context: str) -> str:
    """
    Costruisce il prompt finale per generare la risposta, combinando tutti i contesti.
    """
    # Pulisce i contesti da eventuali placeholder non necessari
    if graph_context and "Nessun dato strutturato disponibile" in graph_context:
        graph_context = ""
    if text_context and "Nessun testo originale disponibile" in text_context:
        text_context = ""
    
    prompt = f"""
Sei un assistente esperto e amichevole per la piattaforma di e-procurement EmPULIA.
Il tuo compito è rispondere alla domanda dell'utente in modo chiaro, preciso e completo, basandoti **ESCLUSIVAMENTE** sulle informazioni fornite.

**NON INVENTARE INFORMAZIONI.** Se il contesto fornito non è sufficiente per rispondere, dichiara onestamente di non avere abbastanza dettagli.

Ecco la domanda dell'utente e le informazioni che ho recuperato per te:

---
**DOMANDA UTENTE:**
{user_question}
---

**INFORMAZIONI RECUPERATE DAL KNOWLEDGE GRAPH (Fatti Strutturati):**
{graph_context if graph_context else "Nessun dato strutturato specifico disponibile per questa domanda."}
---

**INFORMAZIONI RECUPERATE DAI MANUALI ORIGINALI (Contesto Testuale):**
{text_context if text_context else "Nessun testo originale specifico disponibile per questa domanda."}
---

**ISTRUZIONI PER LA RISPOSTA:**

1.  **Sintetizza:** Leggi e comprendi sia le informazioni strutturate dal Knowledge Graph sia il testo originale. Le informazioni del grafo sono un riassunto affidabile, mentre il testo originale fornisce dettagli e sfumature. Usali entrambi per costruire la tua risposta.
2.  **Sii Diretto:** Inizia la risposta in modo diretto, rispondendo alla domanda principale dell'utente.
3.  **Struttura la Risposta:** Se stai descrivendo una procedura o una lista di requisiti, usa elenchi puntati o numerati per rendere la risposta facile da seguire.
4.  **Cita le Fonti (Implicitamente):** Quando possibile, fai riferimento al contesto. Ad esempio, puoi dire "Secondo la guida..." o "La procedura prevede che...".
5.  **Tono:** Mantieni un tono professionale, chiaro e di supporto.
6.  **Regola Fondamentale:** Se una informazione non è presente nel contesto fornito, **NON DEVI menzionarla**. È meglio dare una risposta incompleta ma corretta che una risposta completa ma con informazioni inventate.
7.  **Lunghezza:** Fornisci una risposta completa ma concisa. Non ripetere informazioni inutilmente.

**RISPOSTA IN ITALIANO:**
"""
    return prompt

def validate_context(retrieved_context: Dict[str, Any]) -> bool:
    """
    Valida se il contesto recuperato contiene informazioni utili.
    """
    graph_context = retrieved_context.get("graph_context", "")
    text_context = retrieved_context.get("text_context", "")
    
    # Controlli per contesti vuoti o con messaggi di default
    graph_empty = (not graph_context or 
                   "Nessun dato strutturato disponibile" in graph_context or
                   "Nessuna informazione trovata" in graph_context)
    
    text_empty = (not text_context or 
                  "Nessun testo originale disponibile" in text_context or
                  "Nessuna informazione trovata" in text_context)
    
    return not (graph_empty and text_empty)

def get_knowledge_context(user_question: str) -> Dict[str, Any]:
    """
    Recupera il contesto di conoscenza usando il knowledge retriever.
    
    Args:
        user_question: La domanda dell'utente
    
    Returns:
        Dizionario contenente 'graph_context' e 'text_context'
    """
    if not KNOWLEDGE_RETRIEVER_AVAILABLE:
        print("Avviso: Knowledge retriever non disponibile, uso contesto mock.")
        return {
            "graph_context": "Knowledge retriever non disponibile.",
            "text_context": "Knowledge retriever non disponibile."
        }
    
    try:
        print(f"Recupero informazioni per la domanda: {user_question}")
        # Chiama il knowledge retriever
        retrieved_context = retrieve_knowledge(user_question)
        
        # Valida il formato del contesto recuperato
        if not isinstance(retrieved_context, dict):
            print("Errore: Il knowledge retriever ha restituito un formato non valido.")
            return {
                "graph_context": "Errore nel recupero delle informazioni dal knowledge graph.",
                "text_context": "Errore nel recupero delle informazioni testuali."
            }
        
        return retrieved_context
        
    except Exception as e:
        print(f"Errore durante il recupero della conoscenza: {e}")
        return {
            "graph_context": f"Errore nel recupero delle informazioni: {str(e)}",
            "text_context": f"Errore nel recupero delle informazioni: {str(e)}"
        }

def generate_final_answer(user_question: str, retrieved_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Funzione principale che orchestra la generazione della risposta finale.
    
    Args:
        user_question: La domanda dell'utente
        retrieved_context: Dizionario contenente 'graph_context' e 'text_context' (opzionale)
                          Se non fornito, verrà recuperato automaticamente
    
    Returns:
        La risposta finale generata dal modello
    """
    if not user_question or not user_question.strip():
        return "Mi dispiace, non ho ricevuto una domanda valida. Potresti riformularla?"
    
    # Se il contesto non è fornito, recuperalo usando il knowledge retriever
    if retrieved_context is None:
        retrieved_context = get_knowledge_context(user_question)
    
    # Valida il contesto recuperato
    if not validate_context(retrieved_context):
        return ("Mi dispiace, non sono riuscito a trovare alcuna informazione rilevante "
                "per la tua domanda nella mia base di conoscenza. Potresti provare a "
                "riformulare la domanda usando termini più specifici relativi alla piattaforma EmPULIA?")

    graph_context = retrieved_context.get("graph_context", "")
    text_context = retrieved_context.get("text_context", "")

    print("Generazione della risposta finale basata sul contesto recuperato...")
    print(f"Contesto grafo disponibile: {bool(graph_context and 'Nessun' not in graph_context and 'Errore' not in graph_context)}")
    print(f"Contesto testuale disponibile: {bool(text_context and 'Nessun' not in text_context and 'Errore' not in text_context)}")
    
    prompt = build_answer_generation_prompt(user_question, graph_context, text_context)
    
    final_answer = call_gemini_with_retries(prompt)
    
    if not final_answer:
        return ("Mi dispiace, si è verificato un problema durante la generazione della risposta. "
                "Riprova più tardi o contatta l'assistenza tecnica.")
    
    # Salva il log dell'interazione
    save_interaction_log(user_question, retrieved_context, final_answer)
        
    return final_answer

def answer_user_question(user_question: str) -> str:
    """
    Funzione principale per rispondere alle domande degli utenti.
    Questa è la funzione da chiamare dall'interfaccia utente.
    
    Args:
        user_question: La domanda dell'utente
    
    Returns:
        La risposta generata
    """
    try:
        return generate_final_answer(user_question)
    except Exception as e:
        print(f"Errore critico nella generazione della risposta: {e}")
        return ("Mi dispiace, si è verificato un errore imprevisto. "
                "Riprova più tardi o contatta l'assistenza tecnica.")

def save_interaction_log(user_question: str, context: Dict[str, Any], answer: str, log_file: str = "interaction_log.json") -> None:
    """
    Salva un log dell'interazione per debugging e miglioramento.
    """
    try:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_question": user_question,
            "graph_context_length": len(context.get("graph_context", "")),
            "text_context_length": len(context.get("text_context", "")),
            "answer_length": len(answer),
            "answer": answer,
            "context_valid": validate_context(context)
        }
        
        # Carica log esistenti o crea nuovo file
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, IOError):
                logs = []
        
        logs.append(log_entry)
        
        # Mantieni solo gli ultimi 100 log per evitare file troppo grandi
        if len(logs) > 100:
            logs = logs[-100:]
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Avviso: Impossibile salvare il log dell'interazione: {e}")

# Funzione di utilità per test
def test_answer_generation(test_question: str = "Come si registra un utente PA su EmPULIA?") -> str:
    """
    Funzione di test per verificare il funzionamento del generatore di risposte.
    Usa il knowledge retriever reale se disponibile, altrimenti un contesto mock.
    """
    if KNOWLEDGE_RETRIEVER_AVAILABLE:
        print("Test con knowledge retriever reale...")
        return generate_final_answer(test_question)
    else:
        print("Test con contesto mock (knowledge retriever non disponibile)...")
        mock_context = {
            "graph_context": "Test: Registrazione utente PA richiede SPID e selezione ente.",
            "text_context": "Test: La procedura di registrazione inizia dalla homepage della piattaforma."
        }
        return generate_final_answer(test_question, mock_context)

if __name__ == "__main__":
    # Test di base
    print("Test del generatore di risposte...")
    test_result = test_answer_generation()
    print(f"Risultato test: {test_result}")
    
    # Test interattivo se desiderato
    # while True:
    #     user_input = input("\nInserisci una domanda (o 'quit' per uscire): ")
    #     if user_input.lower() == 'quit':
    #         break
    #     response = answer_user_question(user_input)
    #     print(f"\nRisposta: {response}")