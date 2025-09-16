import os
import json
import time
from typing import Dict, Any
import google.generativeai as genai
import ollama

# --- Configurazione ---
# L'utente sceglierà quale usare tramite una variabile d'ambiente o un parametro
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini") # "gemini" o "ollama"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Modelli da usare per ogni provider
GEMINI_ANALYSIS_MODEL = "gemini-2.5-pro"
GEMINI_SYNTHESIS_MODEL = "gemini-2.5-pro"
#OLLAMA_MODEL_NAME = "llama3"
#OLLAMA_MODEL_NAME = "qwen3:8b"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#OLLAMA_MODEL_NAME = "qwen:14b-chat"
OLLAMA_MODEL_NAME = "gpt-oss:20b"

# --- Funzioni per GEMINI ---
def call_gemini_api(prompt: str, model_name: str, expect_json: bool) -> str:
    """Funzione centralizzata per chiamare l'API Gemini."""
    try:
        config = {"temperature": 0.0}
        if expect_json:
            config["response_mime_type"] = "application/json"
        
        model = genai.GenerativeModel(model_name=model_name, generation_config=config)
        response = model.generate_content(prompt)
        
        cleaned_response = response.text.strip()
        if expect_json and cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        return cleaned_response
    except Exception as e:
        print(f"Errore durante la chiamata a Gemini: {e}")
        return ""


# --- Funzioni per OLLAMA ---
def call_ollama_api(prompt: str, expect_json: bool) -> str:
    """Funzione centralizzata per chiamare l'API di Ollama."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            # format='json' if expect_json else '',
            options={'temperature': 0}
        )
        cleaned_response = response['message']['content'].strip()
        if expect_json and cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        return cleaned_response
    except Exception as e:
        print(f"Errore durante la chiamata a Ollama: {e}")
        return ""

# --- Funzioni di Interfaccia con Retry Logic ---
def call_llm_for_analysis(prompt: str, max_retries: int = 3, delay: int = 5) -> str:
    """Chiama l'LLM configurato per l'analisi, aspettandosi un JSON."""
    for attempt in range(max_retries):
        print(f"Tentativo {attempt + 1}/{max_retries} di analisi con {LLM_PROVIDER.upper()}...")
        if LLM_PROVIDER == "ollama":
            result = call_ollama_api(prompt, expect_json=True)
        else: # Default a Gemini
            result = call_gemini_api(prompt, model_name=GEMINI_ANALYSIS_MODEL, expect_json=True)
        
        if result:
            return result
        time.sleep(delay * (2 ** attempt))
    
    print(f"Massimo numero di tentativi raggiunto con {LLM_PROVIDER.upper()}.")
    return ""

def call_llm_for_synthesis(prompt: str, max_retries: int = 3, delay: int = 5) -> str:
    """Chiama l'LLM configurato per la sintesi, aspettandosi testo libero."""
    for attempt in range(max_retries):
        print(f"Tentativo {attempt + 1}/{max_retries} di sintesi con {LLM_PROVIDER.upper()}...")
        if LLM_PROVIDER == "ollama":
            result = call_ollama_api(prompt, expect_json=False)
        else: # Default a Gemini
            result = call_gemini_api(prompt, model_name=GEMINI_SYNTHESIS_MODEL, expect_json=False)

        if result:
            return result
        time.sleep(delay * (2 ** attempt))

        

    print(f"Massimo numero di tentativi raggiunto con {LLM_PROVIDER.upper()}.")
    return "Si è verificato un errore durante la generazione della risposta."

if __name__ == "__main__":
    # Test rapido
    test_prompt = "Ciao, come stai oggi?"
    print("Test Sintesi LLM:")
    print(call_llm_for_synthesis(test_prompt))
    print("\nTest Analisi LLM:")
    analysis_prompt = """
Rispondi SOLO con un oggetto JSON valido che abbia ESATTAMENTE queste chiavi: 'intento', 'entita_chiave', 'termini_di_ricerca_espansi'.
Esempio:
{
  "intento": "find_procedure",
  "entita_chiave": [{"nome": "Password", "tipo": "ParametroConfigurazione"}],
  "termini_di_ricerca_espansi": ["reset password", "cambio password", "recupero password"]
}
Domanda: "Quali sono i passaggi per resettare la mia password?"
"""
    print(call_llm_for_analysis(analysis_prompt))