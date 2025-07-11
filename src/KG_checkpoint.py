"""
Sistema di checkpoint per la costruzione del Knowledge Graph.
Gestisce il salvataggio e ripristino del progresso durante l'elaborazione.
"""

import json
import os
import pickle
import time
import glob
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

def create_checkpoint_filename(base_name: str, total_chunks: int) -> str:
    """Crea un nome file per il checkpoint basato sui parametri."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_checkpoint_{total_chunks}chunks_{timestamp}.pkl"

def save_checkpoint(data: dict, filepath: str) -> None:
    """Salva un checkpoint con i dati di progresso."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint salvato: {filepath}")
    except Exception as e:
        print(f"Errore nel salvataggio checkpoint: {e}")

def load_checkpoint(filepath: str) -> Optional[dict]:
    """Carica un checkpoint esistente."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Checkpoint caricato: {filepath}")
        return data
    except FileNotFoundError:
        print(f"Nessun checkpoint trovato: {filepath}")
        return None
    except Exception as e:
        print(f"Errore nel caricamento checkpoint: {e}")
        return None

def find_latest_checkpoint(pattern: str = "*checkpoint*.pkl") -> Optional[str]:
    """Trova l'ultimo checkpoint disponibile."""
    checkpoints = glob.glob(pattern)
    if checkpoints:
        # Ordina per data di modifica (più recente prima)
        latest = max(checkpoints, key=os.path.getmtime)
        return latest
    return None

def ask_user_confirmation(message: str) -> bool:
    """Chiede conferma all'utente."""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Risposta non valida. Inserisci 'y' per sì o 'n' per no.")

def extract_knowledge_from_chunks_with_checkpoint(
    chunks: List[Dict[str, Any]], 
    output_dir: str = "llm_outputs", 
    checkpoint_every: int = 10
) -> Tuple[List[Dict], List[Dict]]:
    """
    Versione con checkpoint dell'estrazione della conoscenza.
    Salva il progresso ogni N chunk processati.
    """
    # Import locale per evitare import circolare
    from build_KG import build_extraction_prompt, call_llm_api, parse_llm_extraction_output
    
    checkpoint_file = f"extraction_checkpoint_{len(chunks)}chunks.pkl"
    
    # Prova a caricare un checkpoint esistente
    checkpoint_data = load_checkpoint(checkpoint_file)
    
    if checkpoint_data:
        print(f"Ripresa dall'ultimo checkpoint:")
        print(f"  Chunk processati: {checkpoint_data['processed_count']}/{len(chunks)}")
        print(f"  Entità accumulate: {len(checkpoint_data['all_entities'])}")
        print(f"  Relazioni accumulate: {len(checkpoint_data['all_relations'])}")
        
        # Chiedi conferma per riprendere
        if ask_user_confirmation("Vuoi riprendere dall'ultimo checkpoint?"):
            all_entities = checkpoint_data['all_entities']
            all_relations = checkpoint_data['all_relations']
            start_index = checkpoint_data['processed_count']
            processed_chunks_count = checkpoint_data['processed_count']
        else:
            print("Inizio da zero...")
            all_entities = []
            all_relations = []
            start_index = 0
            processed_chunks_count = 0
    else:
        all_entities = []
        all_relations = []
        start_index = 0
        processed_chunks_count = 0
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Processa i chunk rimanenti
    for i in range(start_index, len(chunks)):
        chunk = chunks[i]
        chunk_id = chunk.get('chunk_id', f"chunk_{i}")
        section_title = chunk.get('section_title', "Nessun Titolo Assegnato")
        chunk_text = chunk.get('text', "")

        # Salva il singolo chunk in un file di testo
        chunk_filename = os.path.join(output_dir, f"{chunk_id}_input.txt")
        try:
            with open(chunk_filename, 'w', encoding='utf-8') as f_out:
                f_out.write(f"CHUNK_ID: {chunk_id}\nPAGE_NUMBER: {chunk.get('page_number')}\nSECTION_TITLE: {section_title}\n\n---\n{chunk_text}")
        except Exception as e:
            print(f"Errore durante il salvataggio del chunk input {chunk_id}: {e}")

        print(f"Processo il chunk {i+1}/{len(chunks)}: ID='{chunk_id}' - Sezione='{section_title}'")
        
        if not chunk_text.strip():
            print(f"Avviso: Chunk {chunk_id} saltato per mancanza di testo significativo.")
            continue

        try:
            prompt = build_extraction_prompt(chunk_text, section_title, chunk_id)
            llm_output_str = call_llm_api(prompt)

            # Salva l'output LLM
            llm_output_filename = os.path.join(output_dir, f"{chunk_id}_llm_output.json")
            try:
                with open(llm_output_filename, 'w', encoding='utf-8') as f_out:
                    try:
                        parsed_json = json.loads(llm_output_str)
                        json.dump(parsed_json, f_out, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        f_out.write(llm_output_str if llm_output_str else "{}")
            except Exception as e:
                print(f"Errore durante il salvataggio dell'output LLM per {chunk_id}: {e}")

            if llm_output_str:
                entities, relations = parse_llm_extraction_output(llm_output_str)
                
                # Aggiungi provenienza
                for entity in entities:
                    entity['source_chunk_id'] = chunk_id
                    entity['source_page_number'] = chunk.get('page_number')
                    entity['source_section_title'] = section_title
                for relation in relations:
                    relation['source_chunk_id'] = chunk_id
                    relation['source_page_number'] = chunk.get('page_number')
                    relation['source_section_title'] = section_title

                all_entities.extend(entities)
                all_relations.extend(relations)
                processed_chunks_count += 1
                print(f"  Estratte {len(entities)} entità e {len(relations)} relazioni.")
            else:
                print(f"  Nessun output valido dall'LLM per il chunk {chunk_id}.")

        except Exception as e:
            print(f"ERRORE nel processamento del chunk {chunk_id}: {e}")
            print("Continuo con il prossimo chunk...")

        # Salva checkpoint ogni N chunk
        if (i + 1) % checkpoint_every == 0:
            checkpoint_data = {
                'all_entities': all_entities,
                'all_relations': all_relations,
                'processed_count': i + 1,
                'total_chunks': len(chunks),
                'last_processed_chunk_id': chunk_id,
                'timestamp': datetime.now().isoformat()
            }
            save_checkpoint(checkpoint_data, checkpoint_file)

        # Pausa tra chunk
        if i < len(chunks) - 1:
            time.sleep(1.5)

    # Salva checkpoint finale
    final_checkpoint_data = {
        'all_entities': all_entities,
        'all_relations': all_relations,
        'processed_count': len(chunks),
        'total_chunks': len(chunks),
        'last_processed_chunk_id': chunk_id if 'chunk_id' in locals() else 'unknown',
        'timestamp': datetime.now().isoformat(),
        'completed': True
    }
    save_checkpoint(final_checkpoint_data, checkpoint_file)

    print(f"\nElaborazione chunk completata. Processati {processed_chunks_count}/{len(chunks)} chunk con output valido.")
    print(f"Totale entità estratte: {len(all_entities)}")
    print(f"Totale relazioni estratte: {len(all_relations)}")
    
    return all_entities, all_relations

def check_existing_files(filepath: str, description: str) -> bool:
    """Controlla se esiste un file e chiede all'utente se usarlo."""
    if os.path.exists(filepath):
        print(f"File {description} trovato: {filepath}")
        return ask_user_confirmation(f"Vuoi usare il file esistente per {description}?")
    return False

def load_existing_json_files(entities_path: str, relations_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Carica file JSON esistenti."""
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    with open(relations_path, 'r', encoding='utf-8') as f:
        relations = json.load(f)
    return entities, relations

def process_with_full_checkpoint_system(
    input_json_path: str, 
    output_dir_llm: str = "llm_extraction_outputs"
) -> None:
    """Sistema completo con checkpoint per tutti i passaggi."""
    
    # Import locale per evitare import circolare
    from build_KG import (
        load_chunks_from_json, save_kg_to_json, 
        aggregate_knowledge_improved, llm_cluster_knowledge
    )
    
    # Definisci i percorsi di output
    output_entities_raw_path = "kg_entities_raw_empulia.json"
    output_relations_raw_path = "kg_relations_raw_empulia.json"
    output_entities_aggregated_improved_path = "kg_entities_aggregated_improved_empulia.json"
    output_relations_aggregated_improved_path = "kg_relations_aggregated_improved_empulia.json"
    output_entities_clustered_path = "kg_entities_clustered_final_empulia.json"
    output_relations_clustered_path = "kg_relations_clustered_final_empulia.json"
    
    # Carica i chunk
    print("=== CARICAMENTO CHUNK ===")
    document_chunks = load_chunks_from_json(input_json_path)
    if not document_chunks:
        print(f"Nessun chunk caricato da {input_json_path}. Verifica il file.")
        return
    
    print(f"Caricati {len(document_chunks)} chunk dal dataset.")
    
    # FASE 1: Estrazione con checkpoint
    print("\n=== FASE 1: ESTRAZIONE ENTITÀ E RELAZIONI ===")
    
    # Controlla se esistono già i file di output grezzi
    if (check_existing_files(output_entities_raw_path, "estrazione grezza entità") and 
        check_existing_files(output_relations_raw_path, "estrazione grezza relazioni")):
        
        print("Caricamento dei file esistenti...")
        raw_entities, raw_relations = load_existing_json_files(
            output_entities_raw_path, output_relations_raw_path
        )
        print(f"Caricati {len(raw_entities)} entità e {len(raw_relations)} relazioni.")
    else:
        raw_entities, raw_relations = extract_knowledge_from_chunks_with_checkpoint(
            document_chunks, output_dir_llm, checkpoint_every=5
        )
        save_kg_to_json(raw_entities, output_entities_raw_path, "Entità grezze")
        save_kg_to_json(raw_relations, output_relations_raw_path, "Relazioni grezze")
    
    # FASE 2: Aggregazione migliorata
    print("\n=== FASE 2: AGGREGAZIONE MIGLIORATA ===")
    
    if (check_existing_files(output_entities_aggregated_improved_path, "aggregazione migliorata entità") and 
        check_existing_files(output_relations_aggregated_improved_path, "aggregazione migliorata relazioni")):
        
        print("Caricamento dei file esistenti...")
        aggregated_entities_improved, aggregated_relations_improved = load_existing_json_files(
            output_entities_aggregated_improved_path, output_relations_aggregated_improved_path
        )
        print(f"Caricati {len(aggregated_entities_improved)} entità e {len(aggregated_relations_improved)} relazioni aggregate.")
    else:
        aggregated_entities_improved, aggregated_relations_improved = aggregate_knowledge_improved(
            raw_entities, raw_relations
        )
        save_kg_to_json(aggregated_entities_improved, output_entities_aggregated_improved_path, 
                       "Entità aggregate (versione migliorata)")
        save_kg_to_json(aggregated_relations_improved, output_relations_aggregated_improved_path, 
                       "Relazioni aggregate (versione migliorata)")
    
    # FASE 3: Clustering con LLM
    print("\n=== FASE 3: CLUSTERING CON LLM ===")
    
    if (check_existing_files(output_entities_clustered_path, "clustering finale entità") and 
        check_existing_files(output_relations_clustered_path, "clustering finale relazioni")):
        
        print("Caricamento dei file esistenti...")
        final_clustered_entities, final_clustered_relations = load_existing_json_files(
            output_entities_clustered_path, output_relations_clustered_path
        )
        print(f"Caricati {len(final_clustered_entities)} entità e {len(final_clustered_relations)} relazioni clusterizzate.")
    else:
        final_clustered_entities, final_clustered_relations = llm_cluster_knowledge(
            aggregated_entities_improved, 
            aggregated_relations_improved
        )
        save_kg_to_json(final_clustered_entities, output_entities_clustered_path, 
                       "Entità clusterizzate finali (LLM)")
        save_kg_to_json(final_clustered_relations, output_relations_clustered_path, 
                       "Relazioni clusterizzate finali (LLM)")
    
    # Statistiche finali
    print("\n=== COMPLETAMENTO PROCESSAMENTO ===")
    print(f"Entità finali: {len(final_clustered_entities)}")
    print(f"Relazioni finali: {len(final_clustered_relations)}")
    print(f"Riduzione entità: {len(raw_entities)} → {len(aggregated_entities_improved)} → {len(final_clustered_entities)}")
    print(f"Riduzione relazioni: {len(raw_relations)} → {len(aggregated_relations_improved)} → {len(final_clustered_relations)}")
    print(f"Output finali salvati in:")
    print(f"  - {output_entities_clustered_path}")
    print(f"  - {output_relations_clustered_path}")

def cleanup_checkpoints(pattern: str = "*checkpoint*.pkl") -> None:
    """Pulisce i file di checkpoint vecchi."""
    checkpoints = glob.glob(pattern)
    if checkpoints:
        print(f"Trovati {len(checkpoints)} file di checkpoint:")
        for checkpoint in checkpoints:
            print(f"  - {checkpoint}")
        
        if ask_user_confirmation("Vuoi eliminare tutti i checkpoint?"):
            for checkpoint in checkpoints:
                try:
                    os.remove(checkpoint)
                    print(f"Eliminato: {checkpoint}")
                except Exception as e:
                    print(f"Errore nell'eliminazione di {checkpoint}: {e}")
        else:
            print("Checkpoint mantenuti.")
    else:
        print("Nessun checkpoint trovato.")

if __name__ == "__main__":
    """Funzione principale per l'esecuzione standalone."""
    import sys
    
    # Configura API Gemini
    import google.generativeai as genai
    api_key_from_env = os.getenv("GEMINI_API_KEY")
    if api_key_from_env:
        genai.configure(api_key=api_key_from_env)
        print("API Key Gemini caricata dalla variabile d'ambiente.")
    else:
        print("ATTENZIONE: API Key Gemini non trovata. Impostala nella variabile d'ambiente GEMINI_API_KEY.")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--cleanup":
            cleanup_checkpoints()
            sys.exit(0)
    
    # Esempio di utilizzo
    input_json_path = "data\\processed\\test.json"
    output_dir_llm = "llm_extraction_outputs"
    
    process_with_full_checkpoint_system(input_json_path, output_dir_llm)