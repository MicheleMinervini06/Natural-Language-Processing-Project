import json
import os
import time
import logging
from typing import List, Dict, Any, Tuple
from kg_gen import KGGen
from utils.entity_normalizer import EntityNormalizer

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configurazione ---
LLM_MODEL = os.getenv("KGGEN_MODEL", "gemini/gemini-2.0-flash")

# Importa le costanti dei tipi dal modulo principale (se disponibili)
try:
    from build_KG import ENTITY_TYPES, RELATION_TYPES
except ImportError:
    logger.warning("Non è possibile importare ENTITY_TYPES e RELATION_TYPES da build_KG.py")
    ENTITY_TYPES = ["AzioneUtente", "DocumentoSistema", "FunzionalitaPiattaforma", "RuoloUtente", "ProceduraAmministrativa"]
    RELATION_TYPES = ["richiede", "genera", "utilizza", "gestisce", "accede_a"]

def validate_configuration() -> bool:
    """Valida la configurazione necessaria per l'esecuzione."""
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("API Key Gemini non trovata. Imposta la variabile d'ambiente GEMINI_API_KEY.")
        return False
    return True

def load_and_combine_chunks(filepath: str) -> Dict[str, str]:
    """
    Carica i chunk da un file JSON e li combina in un unico blocco di testo per ogni file PDF di origine.
    
    Returns:
        Un dizionario dove la chiave è il nome del file sorgente e il valore è il testo completo.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        docs_text = {}
        for chunk in chunks:
            source_file = chunk.get("source_file")
            text = chunk.get("text", "")
            if source_file and text:
                if source_file not in docs_text:
                    docs_text[source_file] = ""
                docs_text[source_file] += text + "\n\n"
        
        logger.info(f"Caricati e combinati {len(chunks)} chunk in {len(docs_text)} documenti.")
        return docs_text
        
    except Exception as e:
        logger.error(f"Errore nel caricamento e combinazione dei chunk: {e}")
        return {}

def infer_entity_type(entity_name: str) -> str:
    """Inferisce il tipo di entità basandosi su parole chiave."""
    name_lower = entity_name.lower()
    
    # Definizioni di pattern per ciascun tipo
    patterns = {
        "AzioneUtente": ["password", "accesso", "login", "autenticazione", "registrazione", "download", "upload", "invio"],
        "DocumentoSistema": ["documento", "pdf", "dgue", "allegato", "file", "modulo", "certificato", "fattura"],
        "FunzionalitaPiattaforma": ["piattaforma", "sistema", "funzione", "servizio", "modulo", "sezione"],
        "RuoloUtente": ["operatore", "utente", "amministratore", "responsabile", "rup", "direttore"],
        "ProceduraAmministrativa": ["procedura", "gara", "appalto", "bando", "contratto", "offerta", "manifestazione"]
    }
    
    for entity_type, keywords in patterns.items():
        if any(keyword in name_lower for keyword in keywords):
            return entity_type
    
    return "Unknown"

def infer_relation_type(predicate: str) -> str:
    """Inferisce il tipo di relazione basandosi sul predicato."""
    predicate_lower = predicate.lower()
    
    relation_mapping = {
        "richiede": ["richiede", "necessita", "ha_bisogno_di", "dipende_da"],
        "genera": ["genera", "crea", "produce", "emette"],
        "utilizza": ["utilizza", "usa", "impiega", "si_serve_di"],
        "gestisce": ["gestisce", "amministra", "controlla", "supervisiona"],
        "accede_a": ["accede", "accede_a", "consulta", "visualizza"]
    }
    
    for relation_type, keywords in relation_mapping.items():
        if any(keyword in predicate_lower for keyword in keywords):
            return relation_type
    
    return predicate  # Mantieni il predicato originale se non trovato

def normalize_entity_name(entity_name: str) -> str:
    """Normalizza il nome dell'entità usando EntityNormalizer."""
    try:
        return EntityNormalizer.normalize_entity_name(entity_name)
    except Exception as e:
        logger.warning(f"Impossibile normalizzare '{entity_name}': {e}")
        return entity_name

def adapt_kggen_output(graph_result, source_file: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Converte l'output della libreria kg-gen nel formato JSON "raw" atteso dalla nostra pipeline.
    
    Args:
        graph_result: L'oggetto Graph restituito dal metodo kg.generate().
        source_file (str): Il nome del file di origine da cui il grafo è stato estratto.

    Returns:
        Una tupla contenente (lista_entita_raw, lista_relazioni_raw).
    """
    logger.info(f"Tipo di graph_result: {type(graph_result)}")
    logger.info(f"Attributi disponibili: {dir(graph_result)}")
    
    # --- Adattamento Entità in formato RAW ---
    final_entities = []
    
    try:
        # Accesso agli attributi dell'oggetto Graph
        entity_cluster_map = getattr(graph_result, 'entity_clusters', {})
        all_unique_entities = getattr(graph_result, 'entities', set())
        raw_relations = getattr(graph_result, 'relations', set())
        
        logger.info(f"Entity clusters trovati: {len(entity_cluster_map)}")
        logger.info(f"Entità uniche trovate: {len(all_unique_entities)}")
        logger.info(f"Relazioni trovate: {len(raw_relations)}")
        
        # Se gli attributi sono vuoti, proviamo metodi alternativi
        if not all_unique_entities and hasattr(graph_result, 'nodes'):
            all_unique_entities = graph_result.nodes
            logger.info(f"Usando nodes: {len(all_unique_entities)}")
            
        if not raw_relations and hasattr(graph_result, 'edges'):
            raw_relations = graph_result.edges
            logger.info(f"Usando edges: {len(raw_relations)}")
            
    except Exception as e:
        logger.error(f"Errore nell'accesso agli attributi del grafo: {e}")
        return [], []
    
    # Processa tutte le entità uniche come entità separate (formato RAW)
    entity_counter = 0
    for entity_name in all_unique_entities:
        entity_str = str(entity_name).strip()
        if not entity_str:
            continue
            
        # Normalizza il nome dell'entità
        normalized_name = normalize_entity_name(entity_str)
        
        # Inferisci il tipo dell'entità
        entity_type = infer_entity_type(normalized_name)
        
        # Crea il chunk_id nel formato atteso
        chunk_id = f"{source_file}_section_{entity_counter}"
        
        # Crea l'entità in formato RAW
        entity_dict = {
            "nome_entita": normalized_name,
            "tipo_entita": entity_type,
            "descrizione_entita": f"Entità di tipo '{entity_type}' estratta dal documento EmPULIA tramite kg-gen.",
            "source_chunk_id": chunk_id,
            "source_page_number": None,  # kg-gen non fornisce info di pagina
            "source_section_title": None  # kg-gen non fornisce info di sezione
        }
        final_entities.append(entity_dict)
        entity_counter += 1
        
    # --- Adattamento Relazioni in formato RAW ---
    final_relations = []
    
    relation_counter = 0
    for rel in raw_relations:
        try:
            # Le relazioni potrebbero essere tuple (soggetto, predicato, oggetto) 
            # o oggetti con attributi
            if isinstance(rel, (tuple, list)) and len(rel) >= 3:
                soggetto, predicato, oggetto = str(rel[0]), str(rel[1]), str(rel[2])
            elif hasattr(rel, 'source') and hasattr(rel, 'target') and hasattr(rel, 'relation'):
                soggetto = str(rel.source)
                predicato = str(rel.relation)
                oggetto = str(rel.target)
            elif hasattr(rel, 'subject') and hasattr(rel, 'predicate') and hasattr(rel, 'object'):
                soggetto = str(rel.subject)
                predicato = str(rel.predicate)
                oggetto = str(rel.object)
            else:
                logger.warning(f"Formato relazione non riconosciuto: {rel} (tipo: {type(rel)})")
                continue
            
            # Normalizza soggetto e oggetto
            soggetto_normalizzato = normalize_entity_name(soggetto.strip())
            oggetto_normalizzato = normalize_entity_name(oggetto.strip())
            
            # Inferisci il tipo di relazione
            predicato_normalizzato = infer_relation_type(predicato.strip())
            
            # Crea il chunk_id nel formato atteso
            chunk_id = f"{source_file}_section_{relation_counter}"
            
            # Crea la relazione in formato RAW
            relation_dict = {
                "soggetto": soggetto_normalizzato,
                "predicato": predicato_normalizzato,
                "oggetto": oggetto_normalizzato,
                "contesto_relazione": f"Relazione '{predicato}' estratta tramite kg-gen dal documento EmPULIA.",
                "source_chunk_id": chunk_id,
                "source_page_number": None,  # kg-gen non fornisce info di pagina
                "source_section_title": None  # kg-gen non fornisce info di sezione
            }
            final_relations.append(relation_dict)
            relation_counter += 1
            
        except Exception as e:
            logger.warning(f"Errore nel processare la relazione {rel}: {e}")
            continue

    logger.info(f"Entità RAW processate: {len(final_entities)}, Relazioni RAW processate: {len(final_relations)}")
    return final_entities, final_relations

def merge_cross_document_entities_raw(all_entities: List[Dict]) -> List[Dict]:
    """Unisce entità identiche da documenti diversi mantenendo il formato RAW."""
    # Per il formato RAW, ogni entità rimane separata anche se identica
    # Questo mantiene la tracciabilità delle occorrenze multiple
    logger.info(f"Mantenimento formato RAW: {len(all_entities)} entità conservate")
    return all_entities

def merge_cross_document_relations_raw(all_relations: List[Dict]) -> List[Dict]:
    """Unisce relazioni identiche da documenti diversi mantenendo il formato RAW."""
    # Per il formato RAW, ogni relazione rimane separata anche se identica
    logger.info(f"Mantenimento formato RAW: {len(all_relations)} relazioni conservate")
    return all_relations

def save_output_json(data: List[Dict], filepath: str, description: str):
    """Salva i dati in un file JSON."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"{description} salvate con successo in '{filepath}' ({len(data)} elementi)")
    except Exception as e:
        logger.error(f"Errore nel salvataggio del file {description}: {e}")

def main():
    """Funzione principale."""
    if not validate_configuration():
        return 1

    input_json_path = "data/processed/processed_chunks_toc_enhanced.json"
    output_entities_raw_path = "kg_entities_raw_empulia_kggen.json"
    output_relations_raw_path = "kg_relations_raw_empulia_kggen.json"

    # 1. Carica e combina i chunk in documenti completi
    documents_to_process = load_and_combine_chunks(input_json_path)

    if not documents_to_process:
        logger.error("Nessun documento da processare. Esecuzione terminata.")
        return 1

    # 2. Inizializza KGGen
    logger.info(f"Inizializzazione di KGGen con il modello: {LLM_MODEL}")
    try:
        kg_generator = KGGen(
            model=LLM_MODEL,
            temperature=0.1
        )
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione di KGGen: {e}")
        return 1

    all_final_entities = []
    all_final_relations = []
    
    # 3. Processa ogni documento separatamente
    for source_file, full_text in documents_to_process.items():
        logger.info(f"Inizio estrazione per il documento: {source_file}")
        
        try:
            # Context più specifico per EmPULIA
            context_prompt = (
                "Questo testo è un manuale utente per la piattaforma di e-procurement EmPULIA. "
                "Estrai entità e relazioni relative a: procedure amministrative, ruoli utente, "
                "documenti di sistema, funzionalità della piattaforma, e azioni utente. "
                "Focalizzati su elementi concreti e operativi del sistema."
            )
            
            graph = kg_generator.generate(
                input_data=full_text,
                context=context_prompt,
                cluster=True,
                chunk_size=8000
            )
            
            # 4. Adatta l'output al formato RAW richiesto
            logger.info(f"Adattamento dell'output di kg-gen per {source_file}...")
            final_entities, final_relations = adapt_kggen_output(graph, source_file)
            
            all_final_entities.extend(final_entities)
            all_final_relations.extend(final_relations)

            logger.info(f"Completata estrazione per {source_file}: {len(final_entities)} entità, {len(final_relations)} relazioni")

        except Exception as e:
            logger.error(f"ERRORE durante l'elaborazione di {source_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        time.sleep(5)  # Pausa tra documenti

    # 5. Unione finale mantenendo formato RAW
    logger.info("Unione finale mantenendo formato RAW...")
    merged_entities = merge_cross_document_entities_raw(all_final_entities)
    merged_relations = merge_cross_document_relations_raw(all_final_relations)

    # 6. Salvataggio risultati
    logger.info("Salvataggio dei risultati finali...")
    save_output_json(merged_entities, output_entities_raw_path, "Entità RAW finali")
    save_output_json(merged_relations, output_relations_raw_path, "Relazioni RAW finali")

    logger.info("=== Generazione Knowledge Graph RAW con KG-Gen Completata ===")
    logger.info(f"Totale entità RAW finali: {len(merged_entities)}")
    logger.info(f"Totale relazioni RAW finali: {len(merged_relations)}")
    
    return 0

if __name__ == "__main__":
    exit(main())