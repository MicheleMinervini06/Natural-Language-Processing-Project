# Configurazione API Gemini

## Come ottenere la chiave API di Gemini

1. Vai su [Google AI Studio](https://aistudio.google.com/)
2. Accedi con il tuo account Google
3. Vai su "Get API Key" nel menu laterale
4. Crea una nuova API key o usa una esistente
5. Copia la chiave API

## Come configurare la chiave API

### Metodo 1: Variabile d'ambiente (consigliato)

#### Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="la_tua_chiave_api_qui"
```

#### Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=la_tua_chiave_api_qui
```

#### Linux/Mac:
```bash
export GEMINI_API_KEY="la_tua_chiave_api_qui"
```

Per rendere permanente la configurazione, aggiungi la riga export al file `.bashrc` o `.zshrc`.

### Metodo 2: Direttamente nel codice (sconsigliato)

Puoi anche inserire la chiave direttamente nel codice decommentando e modificando questa riga in `build_KG.py`:

```python
genai.configure(api_key="TUA_CHIAVE_API_GEMINI")
```

**ATTENZIONE:** Non committare mai la chiave API nel codice sorgente!

## Installazione dipendenze

Assicurati di aver installato la libreria necessaria:

```bash
pip install google-generativeai
```

## Modelli disponibili

- `gemini-1.5-pro`: Modello più avanzato, migliore qualità
- `gemini-1.5-flash`: Modello più veloce, costi inferiori
- `gemini-1.0-pro`: Modello base

Il codice è configurato per usare `gemini-1.5-pro` per default.
