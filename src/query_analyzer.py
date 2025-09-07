import google.generativeai as genai
import json
import os
import time
from typing import List, Dict, Any

# --- Configurazione Globale ---
   
LLM_MODEL_ANALYSIS = "gemini-2.5-pro"

# I tipi di entità rimangono gli stessi del nostro KG
KNOWN_ENTITY_TYPES = [
    "PiattaformaModulo", "FunzionalitàPiattaforma", "RuoloUtente", "AzioneUtente",
    "OperazioneSistema", "InterfacciaUtenteElemento", "ComandoUI", "DocumentoSistema",
    "TipoDocumento", "Prerequisito", "Condizione", "ParametroConfigurazione",
    "Criterio", "StatoDocumento", "StatoProcedura", "EnteEsterno",
    "Organismo", "TermineTemporale", "Scadenza", "MessaggioSistema",
    "Notifica", "SezioneGuida", "NotificaSistema"
]

# Carica la lista dei nomi canonici una sola volta all'avvio
def load_canonical_entity_names(filepath: str) -> list:
    """Carica i nomi canonici delle entità dal file del KG clusterizzato."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        return sorted([e.get("nome_entita_cluster", "") for e in entities if e.get("nome_entita_cluster")])
    except Exception as e:
        print(f"Errore nel caricamento dei nomi canonici da {filepath}: {e}")
        return []

# Esegui questa operazione una volta quando il modulo viene importato
CANONICAL_ENTITY_NAMES = load_canonical_entity_names("kg_entities_clustered_final_empulia.json")

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


def get_few_shot_examples() -> str:
    """
    Restituisce una stringa formattata con esempi few-shot di alta qualità
    per guidare l'LLM nei casi più difficili.
    """
    examples = [
        {
            "user_question": "Come si fa a creare un contratto per più lotti vinti dallo stesso operatore?",
            "analysis": {
                "intento": "find_procedure",
                "entita_chiave": [
                    {"nome": "Contratto Unico Multi-Lotto", "tipo": "FunzionalitàPiattaforma"},
                    {"nome": "Operatore Economico", "tipo": "RuoloUtente"}
                ],
                "domanda_originale": "Come si fa a creare un contratto per più lotti vinti dallo stesso operatore?"
            }
        },
        {
            "user_question": "Spiegami la differenza tra Seggio di Gara e Commissione Tecnica.",
            "analysis": {
                "intento": "find_relationship",
                "entita_chiave": [
                    {"nome": "Seggio di Gara", "tipo": "Organismo"},
                    {"nome": "Commissione Tecnica", "tipo": "Organismo"}
                ],
                "domanda_originale": "Spiegami la differenza tra Seggio di Gara e Commissione Tecnica."
            }
        },
        {
            "user_question": "Cosa succede se NON invio la scheda di conclusione del contratto?",
            "analysis": {
                "intento": "find_procedure",
                "entita_chiave": [
                    {"nome": "Scheda Conclusione Contratto", "tipo": "DocumentoSistema"},
                    {"nome": "Invio", "tipo": "AzioneUtente"}
                ],
                "domanda_originale": "Cosa succede se NON invio la scheda di conclusione del contratto?"
            }
        },
        {
            "user_question": "Cosa serve per il ruolo di RUP PDG?",
            "analysis": {
                "intento": "find_requirements",
                "entita_chiave": [
                    {"nome": "RUP PDG", "tipo": "RuoloUtente"}
                ],
                "domanda_originale": "Cosa serve per il ruolo di RUP PDG?"
            }
        }
    ]
    
    # Formatta gli esempi in una stringa leggibile per il prompt
    formatted_examples = ""
    for ex in examples:
        formatted_examples += f"Esempio Domanda: \"{ex['user_question']}\"\n"
        formatted_examples += f"Esempio JSON:\n```json\n{json.dumps(ex['analysis'], indent=2, ensure_ascii=False)}\n```\n---\n"
        
    return formatted_examples

def build_gemini_query_analysis_prompt(user_question: str) -> str:
    """
    Prompt MIGLIORATO E RINFORZATO per mappare la domanda ai nomi canonici del KG,
    costringendo l'LLM a usare le chiavi JSON corrette.
    """
    # La logica per ottenere la lista di entità rimane la stessa
    entity_list_for_prompt = CANONICAL_ENTITY_NAMES
    # if len(entity_list_for_prompt) > 200:
    #     entity_list_for_prompt = entity_list_for_prompt[:200]

    prompt = f"""
Sei un sistema di Natural Language Understanding (NLU) altamente preciso. Il tuo unico scopo è convertire una domanda utente in un oggetto JSON strutturato, seguendo RIGOROSAMENTE lo schema fornito. Non deviare mai dal formato richiesto.

**Domanda Utente da Analizzare:**
`{user_question}`

---
**VOCABOLARIO CONTROLLATO (Nomi Canonici delle Entità):**
{', '.join(entity_list_for_prompt)}
---

**ISTRUZIONI PER LA GENERAZIONE DEL JSON:**

1.  **Campo `intento`:** Analizza la domanda e assegna UNO dei seguenti valori: `find_procedure`, `find_definition`, `find_requirements`, `find_relationship`, `generic_search`.

2.  **Campo `entita_chiave`:** Questo DEVE essere una lista di oggetti. Per ogni concetto rilevante nella domanda, trova il nome canonico più simile nel VOCABOLARIO CONTROLLATO e crea un oggetto con i seguenti campi:
    *   `nome`: (stringa) Il nome canonico esatto che hai scelto dal vocabolario.
    *   `tipo`: (stringa) Il tipo di entità che ritieni più appropriato per quel nome canonico (puoi inferirlo).

3.  **Campo `domanda_originale`:** Riporta la domanda dell'utente esattamente com'è.

---
**SCHEMA DI OUTPUT JSON OBBLIGATORIO:**
La tua risposta DEVE essere un singolo oggetto JSON che inizia con `{{` e finisce con `}}`. Le chiavi di primo livello DEVONO essere ESATTAMENTE `intento`, `entita_chiave`, e `domanda_originale`.

**ESEMPIO PERFETTO DI OUTPUT:**
```json
{{
  "intento": "find_requirements",
  "entita_chiave": [
    {{
      "nome": "RUP PDG",
      "tipo": "RuoloUtente"
    }}
  ],
  "domanda_originale": "Cosa serve per il ruolo di RUP PDG?"
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