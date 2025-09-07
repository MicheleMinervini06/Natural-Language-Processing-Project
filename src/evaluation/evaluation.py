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

# --- Helper Functions ---
def load_dataset(filepath: str) -> list:
    """Carica il golden dataset da un file JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore nel caricamento del dataset da '{filepath}': {e}")
        return []

def get_user_choice():
    """Chiede all'utente quale tipo di dati utilizzare per la valutazione."""
    print("\n=== SELEZIONE TIPO DATI PER VALUTAZIONE ===")
    print("1. Dati Aggregati/Clusterizzati (knowledge_retriever)")
    print("2. Dati Grezzi (knowledge_retriever_rawData)")
    print("3. Dati generati con KgGen")
    
    while True:
        choice = input("Inserisci 1, 2 o 3: ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Scelta non valida. Inserisci 1, 2 o 3.")

def import_retriever_class(use_raw_data: bool):
    """Importa dinamicamente la classe KnowledgeRetriever appropriata."""
    if use_raw_data:
        print("Importazione knowledge_retriever_rawData...")
        try:
            from knowledge_retriever_rawData import KnowledgeRetriever, NEO4J_URI, NEO4J_USER, NEO4J_DATABASE, NEO4J_PASSWORD
            return KnowledgeRetriever, NEO4J_URI, NEO4J_USER, NEO4J_DATABASE, NEO4J_PASSWORD
        except ImportError as e:
            print(f"Errore nell'importazione di knowledge_retriever_rawData: {e}")
            print("Assicurati che il file knowledge_retriever_rawData.py esista nella cartella src/")
            return None, None, None, None, None
    else:
        print("Importazione knowledge_retriever...")
        try:
            from knowledge_retriever import KnowledgeRetriever, NEO4J_URI, NEO4J_USER, NEO4J_DATABASE, NEO4J_PASSWORD
            return KnowledgeRetriever, NEO4J_URI, NEO4J_USER, NEO4J_DATABASE, NEO4J_PASSWORD
        except ImportError as e:
            print(f"Errore nell'importazione di knowledge_retriever: {e}")
            return None, None, None, None, None

def generate_evaluation_data_from_pipeline(golden_dataset: list, use_raw_data: bool) -> list:
    """
    Esegue la pipeline di Q&A su ogni domanda del golden dataset per generare i risultati da valutare.
    
    Args:
        golden_dataset: Dataset delle domande di riferimento
        use_raw_data: True per dati raw, False per dati aggregati
    """
    results = []
    total_questions = len(golden_dataset)
    data_type_str = "dati grezzi" if use_raw_data else "dati aggregati"
    print(f"Generazione dei risultati per le {total_questions} domande nel golden set usando {data_type_str}...")
    
    for i, item in enumerate(golden_dataset):
        question = item.get("question")
        if not question:
            print(f"Avviso: saltata voce {i+1} del dataset per mancanza di 'question'.")
            continue
            
        print(f"  Processando domanda {i+1}/{total_questions}: '{question[:60]}...'")
        
        # Esecuzione della pipeline completa con il tipo di dati specificato
        pipeline_output = run_qa_pipeline(question, use_raw_data=use_raw_data)
        
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
    # --- User Choice ---
    choice = get_user_choice()
    use_raw_data = (choice == "2")
    
    # --- Setup ---
    GOLDEN_DATASET_PATH = "data/golden_dataset.json"
    data_type = "rawData" if use_raw_data else "aggregated"
    EVALUATION_RUN_NAME = f"evaluation_run_{data_type}_gemini_{time.strftime('%Y%m%d_%H%M%S')}"

    # --- RAGAs Gemini Model Configuration ---
    print("Configurazione dei modelli Gemini per la valutazione RAGAs...")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("ERRORE: La variabile d'ambiente GEMINI_API_KEY non è impostata. Valutazione interrotta.")
        return

    # Configure models with API key explicitly
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.0,
        google_api_key=gemini_api_key
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )

    # --- Esecuzione Pipeline ---
    golden_dataset = load_dataset(GOLDEN_DATASET_PATH)
    if not golden_dataset:
        print("Valutazione interrotta: golden dataset non caricato.")
        return

    # Genera i dati di valutazione usando il tipo di dati specificato
    evaluation_data = generate_evaluation_data_from_pipeline(golden_dataset, use_raw_data)

    # --- Esecuzione Valutazione RAGAs ---
    dataset_dict = {
        "question": [item["question"] for item in evaluation_data],
        "answer": [item["answer"] for item in evaluation_data],
        "contexts": [item["contexts"] for item in evaluation_data],
        "ground_truth": [item["ground_truth"] for item in evaluation_data]
    }
    hf_dataset = Dataset.from_dict(dataset_dict)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print(f"\nEsecuzione della valutazione con RAGAs usando Gemini come giudice ({data_type})...")
    
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