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

# Aggiungi la cartella 'src' al path per permettere import corretti
# Questo è necessario se esegui lo script direttamente
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

# Importa l'orchestratore e il retriever dal tuo modulo
from answer_generator import KnowledgeRetrieverSingleton, get_knowledge_context, generate_final_answer
from knowledge_retriever import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def load_golden_dataset(filepath: str) -> list:
    """Carica il golden dataset da un file JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore nel caricamento del golden dataset da '{filepath}': {e}")
        return []

def generate_evaluation_data_from_pipeline(golden_dataset: list) -> list:
    """
    Esegue la pipeline di Q&A su ogni domanda del golden dataset per generare i risultati da valutare.
    Riutilizza la logica di `answer_generator`.
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
        
        # Esegui la tua pipeline completa passo dopo passo per raccogliere gli artefatti
        
        # 1. Recupera il contesto
        retrieved_context = get_knowledge_context(question)
        
        # 2. Genera la risposta
        # Passiamo il contesto già recuperato per evitare che la funzione lo recuperi di nuovo
        generated_answer = generate_final_answer(question, retrieved_context)
        
        # 3. Assembla l'output per RAGAs
        # Il contesto per RAGAs è una lista di stringhe. Uniamo il contesto del grafo e del testo.
        contexts_list = []
        graph_ctx = retrieved_context.get("graph_context", "")
        text_ctx = retrieved_context.get("text_context", "")
        
        if graph_ctx and "Nessuna" not in graph_ctx:
            contexts_list.append(graph_ctx)
        if text_ctx and "Nessun" not in text_ctx:
            contexts_list.append(text_ctx)

        result_item = {
            "question": question,
            "answer": generated_answer,
            "contexts": contexts_list,
            "ground_truth": item.get("ideal_answer", "Nessuna risposta ideale fornita.")
        }
        results.append(result_item)
    
    print("Generazione dei risultati per la valutazione completata.")
    return results

def run_ragas_evaluation(evaluation_data: list, run_name: str):
    """
    Prende i dati generati, li converte nel formato Dataset e lancia la valutazione RAGAs.
    """
    # Converti i dati in un formato compatibile con RAGAs
    try:
        dataset_dict = {
            "question": [item["question"] for item in evaluation_data],
            "answer": [item["answer"] for item in evaluation_data],
            "contexts": [item["contexts"] for item in evaluation_data],
            "ground_truth": [item["ground_truth"] for item in evaluation_data]
        }
        hf_dataset = Dataset.from_dict(dataset_dict)
    except KeyError as e:
        print(f"Errore: Chiave mancante nel dataset di valutazione: {e}. Controlla la funzione 'generate_evaluation_data_from_pipeline'.")
        return

    # Definisci le metriche
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Esegui la valutazione
    print("\nEsecuzione della valutazione con RAGAs... (richiede chiamate API a OpenAI/altro LLM per il giudizio)")
    
    # Assicurati che RAGAs usi il modello desiderato (es. OpenAI per il giudice)
    # RAGAs di default cerca le chiavi OpenAI, quindi assicurati che OPENAI_API_KEY sia impostata.
    if "OPENAI_API_KEY" not in os.environ:
         print("ATTENZIONE: La variabile d'ambiente OPENAI_API_KEY non è impostata. RAGAs potrebbe fallire.")
         # return # Potresti voler uscire se la chiave non è disponibile

    result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
    )

    print("\n--- Risultati della Valutazione RAGAs ---")
    print(result)

    # Salva i risultati
    output_dir = "data/evaluation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"{run_name}.csv")
    results_df = result.to_pandas()
    results_df.to_csv(output_path, index=False)
    print(f"\nRisultati della valutazione salvati in: {output_path}")


if __name__ == "__main__":
    # --- Setup ---
    # Path al tuo golden dataset
    GOLDEN_DATASET_PATH = "data/golden_dataset.json"
    
    # Nome per questa esecuzione di valutazione
    EVALUATION_RUN_NAME = f"evaluation_run_{time.strftime('%Y%m%d_%H%M%S')}"

    # --- Esecuzione ---
    
    # Carica il dataset di riferimento
    golden_dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    
    if golden_dataset:
        # Genera i risultati usando la tua pipeline
        evaluation_results = generate_evaluation_data_from_pipeline(golden_dataset)
        
        # Lancia la valutazione con RAGAs
        run_ragas_evaluation(evaluation_results, EVALUATION_RUN_NAME)
    
    # Chiudi la connessione del retriever alla fine di tutto
    KnowledgeRetrieverSingleton.close()
    print("Valutazione completata.")