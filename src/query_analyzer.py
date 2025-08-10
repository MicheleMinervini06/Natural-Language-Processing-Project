import google.generativeai as genai
import json
import os
import time
from typing import List, Dict, Any

# --- Configurazione Globale ---
   
LLM_MODEL_ANALYSIS = "gemini-2.5-flash"

# I tipi di entità rimangono gli stessi del nostro KG
KNOWN_ENTITY_TYPES = [
    "PiattaformaModulo", "FunzionalitàPiattaforma", "RuoloUtente", "AzioneUtente",
    "OperazioneSistema", "InterfacciaUtenteElemento", "ComandoUI", "DocumentoSistema",
    "TipoDocumento", "Prerequisito", "Condizione", "ParametroConfigurazione",
    "Criterio", "StatoDocumento", "StatoProcedura", "EnteEsterno",
    "Organismo", "TermineTemporale", "Scadenza", "MessaggioSistema",
    "Notifica", "SezioneGuida"
]

# --- Funzioni Core per Gemini ---

def call_gemini_with_retries(prompt: str, model_name: str = LLM_MODEL_ANALYSIS, max_retries: int = 3, delay: int = 5) -> str:
    """
    Funzione robusta per chiamare l'API Gemini con gestione dei tentativi.
    Restituisce la risposta del modello come stringa.
    """
    # Configurazione specifica per l'output JSON e la determinazione
    generation_config = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
    }
    
    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # L'API di Gemini a volte racchiude l'output JSON in ```json ... ```
            # Puliamo questo markup se presente.
            cleaned_response = response.text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            return cleaned_response.strip()

        except Exception as e:
            print(f"Errore durante la chiamata a Gemini (tentativo {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                current_delay = delay * (2 ** attempt) # Backoff esponenziale
                print(f"Attendo {current_delay} secondi prima di ritentare...")
                time.sleep(current_delay)
            else:
                print("Massimo numero di tentativi raggiunto.")
                return ""
    return ""


def build_gemini_query_analysis_prompt(user_question: str) -> str:
    """
    Costruisce il prompt per analizzare la domanda dell'utente, ottimizzato per Gemini.
    La logica è la stessa, ma la formulazione può essere leggermente diversa per massimizzare la comprensione.
    """
    prompt = f"""
        Sei un agente esperto di Natural Language Understanding (NLU) che analizza le domande degli utenti per interrogare un Knowledge Graph sulla piattaforma di e-procurement EmPULIA.
        Il tuo unico compito è prendere la domanda dell'utente e convertirla in un oggetto JSON strutturato secondo le istruzioni seguenti. Non aggiungere commenti o spiegazioni al di fuori del JSON.

        **Domanda Utente:**
        `{user_question}`

        **Istruzioni per la Generazione del JSON:**

        1.  **Analisi dell'Intento:** Determina l'obiettivo principale della domanda. Nel campo `intento` del JSON, inserisci UNO dei seguenti valori:
            - `find_procedure`: Per domande su "come fare qualcosa".
            - `find_definition`: Per domande su "cos'è qualcosa".
            - `find_requirements`: Per domande su "cosa serve per qualcosa".
            - `find_relationship`: Per domande che collegano due o più concetti.
            - `generic_search`: Per tutte le altre domande.

        2.  **Estrazione delle Entità Chiave:** Identifica tutte le entità menzionate esplicitamente o implicitamente nella domanda. Nel campo `entita_chiave` (una lista di oggetti), per ogni entità estratta fornisci:
            - `nome`: Il nome dell'entità, normalizzato e conciso.
            - `tipo`: Il tipo più appropriato per l'entità, scelto dalla seguente lista: `{', '.join(KNOWN_ENTITY_TYPES)}`.

        3.  **Inclusione della Domanda Originale:** Nel campo `domanda_originale`, riporta la domanda esatta dell'utente.

        **Output Desiderato (solo JSON):**
        ```json
        {{
        "intento": "...",
        "entita_chiave": [
            {{
            "nome": "...",
            "tipo": "..."
            }}
        ],
        "domanda_originale": "..."
        }}

    """
    return prompt

def analyze_user_question(user_question: str) -> Dict[str, Any]:
    """
    Funzione principale che analizza una domanda utente usando l'API di Gemini.
    Prende una stringa in input e restituisce un dizionario strutturato o un dizionario vuoto in caso di errore.
    """
    if not user_question or not user_question.strip():
        print("Errore: la domanda dell'utente è vuota.")
        return {}
    
    print(f"Analisi della domanda con Gemini: '{user_question}'...")

    prompt = build_gemini_query_analysis_prompt(user_question)
    llm_output_str = call_gemini_with_retries(prompt)

    if not llm_output_str:
        print("Errore: Gemini non ha restituito una risposta valida.")
        return {}

    try:
        analysis_result = json.loads(llm_output_str)
        # Validazione della struttura dell'output
        if "intento" not in analysis_result or "entita_chiave" not in analysis_result:
            raise ValueError("L'output JSON non contiene le chiavi 'intento' o 'entita_chiave'.")
        if not isinstance(analysis_result["entita_chiave"], list):
            raise ValueError("'entita_chiave' non è una lista.")
        
        print("Analisi completata con successo.")
        return analysis_result

    except json.JSONDecodeError:
        print(f"Errore critico: Impossibile fare il parsing dell'output JSON di Gemini.")
        print(f"Output ricevuto: {llm_output_str}")
        return {}
    except ValueError as e:
        print(f"Errore di validazione: {e}")
        print(f"Output ricevuto: {llm_output_str}")
        return {}

if __name__ == "__main__":
    # Configurazione API Key per Gemini
    api_key_from_env = os.getenv("GEMINI_API_KEY")
    if api_key_from_env:
        genai.configure(api_key=api_key_from_env)
        print("API Key Gemini caricata dalla variabile d'ambiente.")
    else:
        print("ATTENZIONE: API Key Gemini non trovata. Impostala nella variabile d'ambiente GEMINI_API_KEY.")
        exit() 
    
    # Esempio 1: Domanda procedurale
    question1 = "Come posso cambiare la mia password su EmPULIA?"
    analysis1 = analyze_user_question(question1)
    print("\n--- Risultato Analisi 1 ---")
    print(json.dumps(analysis1, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50 + "\n")
    
    # Esempio 2: Domanda sui requisiti
    question2 = "Che documenti servono per l'iscrizione all'Albo Fornitori?"
    analysis2 = analyze_user_question(question2)
    print("\n--- Risultato Analisi 2 ---")
    print(json.dumps(analysis2, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50 + "\n")

    # Esempio 3: Domanda di definizione
    question3 = "Cos'è il DGUE?"
    analysis3 = analyze_user_question(question3)
    print("\n--- Risultato Analisi 3 ---")
    print(json.dumps(analysis3, indent=2, ensure_ascii=False))

    print("\n" + "="*50 + "\n")

    # Esempio 4: Come creare una commisione di gara
    question4 = "Come viene gestito il calcolo dell'anomalia dalla piattaforma?"
    analysis4 = analyze_user_question(question4)
    print("\n--- Risultato Analisi 4 ---")
    print(json.dumps(analysis4, indent=2, ensure_ascii=False))

    print("\n" + "="*50 + "\n")

    # Esempio 5: Domanda con errore (stringa vuota)
    question5 = "   "
    analysis5 = analyze_user_question(question5)
    print("\n--- Risultato Analisi 5 (Input Vuoto) ---")
    print(json.dumps(analysis5, indent=2, ensure_ascii=False))