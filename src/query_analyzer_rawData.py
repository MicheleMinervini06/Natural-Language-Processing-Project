import google.generativeai as genai
import json
import os
import time
from typing import List, Dict, Any
from utils.llm_handler import call_llm_for_analysis

# Tipi di entità conosciuti dal nostro Knowledge Graph
KNOWN_ENTITY_TYPES = [
    "PiattaformaModulo", "FunzionalitàPiattaforma", "RuoloUtente", "AzioneUtente",
    "OperazioneSistema", "InterfacciaUtenteElemento", "ComandoUI", "DocumentoSistema",
    "TipoDocumento", "Prerequisito", "Condizione", "ParametroConfigurazione",
    "Criterio", "StatoDocumento", "StatoProcedura", "EnteEsterno",
    "Organismo", "TermineTemporale", "Scadenza", "MessaggioSistema",
    "Notifica", "SezioneGuida", "NotificaSistema"
]

def get_few_shot_examples() -> str:
    """Restituisce esempi few-shot per guidare l'LLM."""
    examples = [
        {
            "user_question": "Qual è il primo passo che devo compiere dopo aver fatto l'accesso?",
            "analysis": {
                "intento": "find_procedure",
                "entita_chiave": [{"nome": "Primo Accesso", "tipo": "AzioneUtente"}, {"nome": "Nuove Credenziali", "tipo": "DocumentoSistema"}],
                "termini_di_ricerca_espansi": ["cambio password", "primo accesso", "nuove credenziali", "lista attività obbligatorie", "procedura post-login"]
            }
        },
        {
            "user_question": "Spiegami la differenza tra Seggio di Gara e Commissione Tecnica.",
            "analysis": {
                "intento": "find_relationship",
                "entita_chiave": [{"nome": "Seggio di Gara", "tipo": "Organismo"}, {"nome": "Commissione Tecnica", "tipo": "Organismo"}],
                "termini_di_ricerca_espansi": ["seggio di gara", "commissione tecnica", "differenza", "ruoli", "composizione"]
            }
        },
        {
            "user_question": "Cosa serve per il ruolo di RUP PDG?",
            "analysis": {
                "intento": "find_requirements",
                "entita_chiave": [{"nome": "RUP PDG", "tipo": "RuoloUtente"}],
                "termini_di_ricerca_espansi": ["RUP PDG", "requisiti", "documenti necessari", "atto di nomina", "abilitazione"]
            }
        }
    ]
    formatted_examples = ""
    for ex in examples:
        formatted_examples += f"Esempio Domanda: \"{ex['user_question']}\"\n"
        formatted_examples += f"Esempio JSON:\n```json\n{json.dumps(ex['analysis'], indent=2, ensure_ascii=False)}\n```\n---\n"
    return formatted_examples

def build_gemini_query_analysis_prompt(user_question: str) -> str:
    """Prompt migliorato che richiede sia l'analisi che l'espansione dei termini di ricerca."""
    few_shot_examples = get_few_shot_examples()
    
    prompt = f"""
Sei un agente esperto di NLU che analizza le domande degli utenti per interrogare un Knowledge Graph sulla piattaforma EmPULIA.
Il tuo compito è convertire la domanda in un oggetto JSON strutturato che includa:
1. `intento`: L'obiettivo della domanda.
2. `entita_chiave`: Le entità principali, normalizzate e tipizzate.
3. `termini_di_ricerca_espansi`: Una lista di parole chiave e sinonimi per una ricerca robusta.

**Istruzioni Dettagliate:**

1.  **Analisi dell'Intento:** In `intento`, usa UNO dei seguenti valori:
    - `find_procedure`: Per domande su "come fare".
    - `find_definition`: Per domande su "cos'è".
    - `find_requirements`: Per domande su "cosa serve per".
    - `find_relationship`: Per confronti come "differenza tra X e Y".
    - `generic_search`: Come fallback.

2.  **Estrazione delle Entità Chiave:** In `entita_chiave`:
    - `nome`: Normalizza il nome dell'entità in modo specifico (es. "cambio psw" -> "Cambio Password").
    - `tipo`: Scegli il tipo più appropriato dalla lista: `{', '.join(KNOWN_ENTITY_TYPES)}`.

3.  **Espansione dei Termini di Ricerca:** In `termini_di_ricerca_espansi`:
    - Crea una lista di 3-5 stringhe.
    - Includi i concetti principali, sinonimi, e termini procedurali correlati. Questa lista è FONDAMENTALE per trovare le informazioni nel grafo.

---
**ESEMPI DI ANALISI CORRETTE:**

{few_shot_examples}
---

**ORA, ANALIZZA LA SEGUENTE DOMANDA:**

**Domanda Utente:**
`{user_question}`

**Output JSON:**
"""
    return prompt

def analyze_user_question(user_question: str) -> Dict[str, Any]:
    """
    Funzione principale che analizza una domanda utente e la espande.
    Restituisce un dizionario strutturato o un dizionario vuoto in caso di errore.
    """
    if not user_question or not user_question.strip():
        print("Errore: la domanda dell'utente è vuota.")
        return {}
    
    print(f"Analisi ed espansione della domanda con LLM scelto: '{user_question}'...")

    prompt = build_gemini_query_analysis_prompt(user_question)
    llm_output_str = call_llm_for_analysis(prompt)

    if not llm_output_str:
        print("Errore: Gemini non ha restituito una risposta valida.")
        return {}

    try:
        analysis_result = json.loads(llm_output_str)
        # Validazione della nuova struttura
        required_keys = ["intento", "entita_chiave", "termini_di_ricerca_espansi"]
        if not all(key in analysis_result for key in required_keys):
            raise ValueError(f"L'output JSON non contiene tutte le chiavi richieste: {required_keys}.")
        if not isinstance(analysis_result["termini_di_ricerca_espansi"], list):
            raise ValueError("'termini_di_ricerca_espansi' non è una lista.")
        
        # Aggiungi la domanda originale per completezza
        analysis_result['domanda_originale'] = user_question
        
        print("Analisi ed espansione completate con successo.")
        return analysis_result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Errore critico durante il parsing o la validazione: {e}")
        print(f"Output ricevuto: {llm_output_str}")
        return {}

if __name__ == "__main__":
    test_question = "Sono un utente e ho appena ricevuto le nuove credenziali. Qual è il primo passo che devo compiere dopo aver fatto l'accesso?"
    analysis = analyze_user_question(test_question)
    print("\n--- Risultato Analisi ---")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))