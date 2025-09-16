import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os
import sys

src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from answer_generator import run_qa_pipeline, close_retriever_connection
except ImportError as e:
    print(f"Errore: Impossibile importare i moduli dalla cartella 'src'. Assicurati che la struttura sia corretta.")
    print(e)
    sys.exit(1)

# --- App FastAPI ---

# Inizializza l'app FastAPI
app = FastAPI(
    title="EmPULIA Q&A API",
    description="API per interrogare la documentazione di EmPULIA usando un sistema RAG potenziato da KG.",
    version="1.0.0",
)

pdf_directory = os.path.join(os.path.dirname(__file__), 'data', 'pdfs')
if os.path.exists(pdf_directory):
    app.mount("/docs", StaticFiles(directory=pdf_directory), name="docs")
    print(f"Documenti PDF serviti da /docs")
else:
    print(f"Attenzione: La directory {pdf_directory} non esiste. I documenti PDF non saranno accessibili.")


# Definisci il modello di dati per la richiesta in input (Payload dell'API)
class QueryRequest(BaseModel):
    question: str
    # Aggiungiamo un flag per selezionare il tipo di dati, anche se useremo sempre 'raw' per ora
    use_raw_data: bool = True 

# Definisci il modello di dati per la risposta (Output dell'API)
class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    sources: dict | None = None # Le fonti potrebbero non essere sempre presenti

# Registra un evento che viene eseguito quando il server si spegne
@app.on_event("shutdown")
def shutdown_event():
    """Chiude le connessioni attive (es. Neo4j) in modo pulito."""
    print("Server in spegnimento, chiusura connessioni...")
    close_retriever_connection()

# Definisci l'endpoint principale dell'API
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Riceve una domanda e restituisce la risposta generata dalla pipeline RAG.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota.")

    try:
        # Chiama la tua pipeline esistente, che ora è importata correttamente
        print(f"Ricevuta domanda per l'API: '{request.question}'")
        result = run_qa_pipeline(request.question, use_raw_data=request.use_raw_data)
        
        # Restituisci il risultato completo in formato JSON, conforme al modello di risposta
        return result

    except Exception as e:
        # Gestisci errori imprevisti che potrebbero verificarsi nella pipeline
        print(f"Errore critico durante l'esecuzione della pipeline: {e}")
        # Restituisci un errore 500 generico per non esporre dettagli interni
        raise HTTPException(status_code=500, detail="Si è verificato un errore interno durante l'elaborazione della domanda.")

# Permette di avviare il server direttamente con "python main.py" dalla root del progetto
if __name__ == "__main__":
    print("Avvio del server FastAPI su http://127.0.0.1:8000")
    print("Apri http://127.0.0.1:8000/docs per la documentazione interattiva dell'API.")
    # `reload=True` è ottimo per lo sviluppo, riavvia il server a ogni modifica del codice
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)