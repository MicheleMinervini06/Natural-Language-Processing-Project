import json
import os
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import pandas as pd
import sys
from dotenv import load_dotenv

# --- RAGAs Configuration with Gemini ---
# Updated imports for newer RAGAs versions
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# --- Setup Paths and Imports ---
# Aggiungi 'src' al path per permettere import corretti
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from answer_generator import run_qa_pipeline 
from knowledge_retriever import NEO4J_URI, NEO4J_USER, NEO4J_DATABASE, NEO4J_PASSWORD
from knowledge_retriever import KnowledgeRetriever # Importa la classe

# --- Helper Functions ---
def load_dataset(filepath: str) -> list:
    """Carica il golden dataset da un file JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore nel caricamento del dataset da '{filepath}': {e}")
        return []

def generate_evaluation_data_from_pipeline(golden_dataset: list, retriever: KnowledgeRetriever) -> list:
    """
    Esegue la pipeline di Q&A su ogni domanda del golden dataset per generare i risultati da valutare.
    """
    results = []
    total_questions = len(golden_dataset)
    print(f"Generazione dei risultati per le {total_questions} domande nel golden set...")
    
    for i, item in enumerate(golden_dataset):
        question = item.get("question")
        if not question:
            print(f"Avviso: saltata voce {i+1} del dataset per mancanza di 'question'.")
            continue
            
        print(f"  Processando domanda {i+1}/{total_questions}: '{question[:60]}...'")
        
        # Esecuzione della pipeline completa passando l'istanza del retriever
        pipeline_output = run_qa_pipeline(question)
        
        result_item = {
            "question": question,
            "answer": pipeline_output["answer"],
            "contexts": pipeline_output["contexts"],
            "ground_truth": item.get("ideal_answer", "Nessuna risposta ideale fornita.")
        }
        results.append(result_item)
    
    print("Generazione dei risultati completata.")
    return results

def main_evaluation():
    # --- Setup ---
    GOLDEN_DATASET_PATH = "data/golden_dataset.json"
    ALL_CHUNKS_PATH = "data/processed/processed_chunks_toc_enhanced.json"
    EVALUATION_RUN_NAME = f"evaluation_run_gemini_{time.strftime('%Y%m%d_%H%M%S')}"

    # --- RAGAs Gemini Model Configuration ---
    print("Configurazione dei modelli Gemini per la valutazione RAGAs...")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("ERRORE: La variabile d'ambiente GEMINI_API_KEY non è impostata. Valutazione interrotta.")
        return

    # Configure models with API key explicitly
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",  # Changed to available model
        temperature=0.0,
        google_api_key=gemini_api_key
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )
    
    # --- Inizializzazione Retriever ---
    print("Inizializzazione del Knowledge Retriever...")
    retriever = KnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, ALL_CHUNKS_PATH)
    if not retriever.driver:
        print("Valutazione interrotta: impossibile connettersi a Neo4j.")
        return

    # --- Esecuzione Pipeline ---
    golden_dataset = load_dataset(GOLDEN_DATASET_PATH)
    if not golden_dataset:
        print("Valutazione interrotta: golden dataset non caricato.")
        retriever.close()
        return

    evaluation_data = generate_evaluation_data_from_pipeline(golden_dataset, retriever)
    
    # Chiudi la connessione a Neo4j, non serve più per la fase di valutazione
    retriever.close()

    # --- Esecuzione Valutazione RAGAs ---
    # Converti i dati in formato Dataset di Hugging Face
    dataset_dict = {
        "question": [item["question"] for item in evaluation_data],
        "answer": [item["answer"] for item in evaluation_data],
        "contexts": [item["contexts"] for item in evaluation_data],
        "ground_truth": [item["ground_truth"] for item in evaluation_data]
    }
    hf_dataset = Dataset.from_dict(dataset_dict)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("\nEsecuzione della valutazione con RAGAs usando Gemini come giudice...")
    
    # Use the models directly in the evaluate function
    result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    print("\n--- Risultati della Valutazione RAGAs ---")
    print(result)
    
    output_dir = "data/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{EVALUATION_RUN_NAME}.csv")
    result.to_pandas().to_csv(output_path, index=False)
    print(f"\nRisultati della valutazione salvati in: {output_path}")

if __name__ == "__main__":
    main_evaluation()