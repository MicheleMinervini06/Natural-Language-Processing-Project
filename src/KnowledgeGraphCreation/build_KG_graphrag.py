import asyncio
import json
import os
import time
import logging
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

# Importazioni per Gemini
import google.generativeai as genai
from dotenv import load_dotenv

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente
load_dotenv()

# --- Configurazione ---
LLM_MODEL_EXTRACTION = os.getenv("KGGEN_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configurazione parallelizzazione
MAX_CONCURRENT_REQUESTS = 5  # Limite di richieste simultanee
BATCH_SIZE = 10  # Dimensione del batch per processamento
RATE_LIMIT_DELAY = 0.5  # Delay tra richieste (secondi)

# --- Placeholder per EntityNormalizer ---
class EntityNormalizer:
    @staticmethod
    def normalize_entity_name(entity_name: str) -> str:
        return ' '.join(entity_name.strip().lower().split())

# --- Schema e Prompt (identici a prima) ---
NODE_LABELS = [
    "AzioneUtente", "DocumentoSistema", "FunzionalitaPiattaforma", "RuoloUtente", "ProceduraAmministrativa",
    "PiattaformaModulo", "InterfacciaUtenteElemento", "StatoProcedura", "NotificaSistema", 
    "CriterioDiValutazione", "TermineTemporale"
]
RELATIONSHIP_TYPES = [
    "richiede", "genera", "utilizza", "gestisce", "accede_a", "ESEGUITA_DA", "INCLUDE_FUNZIONALITA",
    "HA_STATO", "CONTIENE_ELEMENTO", "INVIA_NOTIFICA", "SI_APPLICA_A"
]
PROMPT_TEMPLATE = """
Sei un sistema esperto nell'estrazione di informazioni per creare knowledge graph da manuali tecnici della piattaforma EmPULIA.

REGOLE FONDAMENTALI:
1.  La tua risposta DEVE essere un singolo oggetto JSON valido. Non includere testo, spiegazioni o markdown (```json) prima o dopo il JSON.
2.  Estrai entità (nodi) e relazioni dal testo fornito.
3.  Assegna a ogni nodo un `id` univoco (es: "nodo_1", "nodo_2").
4.  Usa ESCLUSIVAMENTE gli `id` dei nodi per definire `start_node_id` e `end_node_id` di ogni relazione.
5.  Usa ESCLUSIVAMENTE i tipi di nodi (`label`) e relazioni (`type`) forniti nello schema.

SCHEMA:
- Tipi di Nodi (label): {node_labels}
- Tipi di Relazioni (type): {relationship_types}

FORMATO JSON RICHIESTO:
{{
  "nodes": [
    {{"id": "id_stringa_univoco", "label": "TIPO_DI_NODO_DALLO_SCHEMA", "properties": {{"name": "Nome Entità", "description": "Descrizione contestuale."}}}}
  ],
  "relationships": [
    {{"type": "TIPO_DI_RELAZIONE_DALLO_SCHEMA", "start_node_id": "id_nodo_partenza", "end_node_id": "id_nodo_arrivo", "properties": {{"context": "Frase che giustifica la relazione."}}}}
  ]
}}

TESTO DA ANALIZZARE:
---
{text}
---

JSON DI OUTPUT:
"""

class GeminiExtractor:
    """Classe per gestire l'estrazione asincrona con Gemini"""
    
    def __init__(self, api_key: str, model_name: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Configura Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"}
        )
    
    async def extract_async(self, prompt: str, chunk_id: str) -> Dict:
        """Estrazione asincrona con limite di concorrenza"""
        async with self.semaphore:
            try:
                # Usa ThreadPoolExecutor per non bloccare l'event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self._sync_extract, 
                    prompt
                )
                
                logger.info(f"✓ Chunk {chunk_id} processato con successo")
                
                # Rate limiting asincrono
                await asyncio.sleep(RATE_LIMIT_DELAY)
                return result
                
            except Exception as e:
                logger.error(f"✗ Errore nel chunk {chunk_id}: {e}")
                return {"nodes": [], "relationships": []}
    
    def _sync_extract(self, prompt: str) -> Dict:
        """Estrazione sincrona (chiamata dal ThreadPoolExecutor)"""
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
                return json.loads(cleaned_text)
            except Exception as e:
                if attempt == 2:  # Ultimo tentativo
                    logger.warning(f"Tutti i tentativi falliti: {e}")
                    raise e
                time.sleep(2 ** attempt)  # Backoff esponenziale
                logger.warning(f"Tentativo {attempt + 1} fallito, riprovo: {e}")
        
        return {"nodes": [], "relationships": []}

def validate_configuration() -> bool:
    """Validazione configurazione (identica a prima)"""
    if not GEMINI_API_KEY:
        logger.error("API Key Gemini non trovata. Imposta la variabile d'ambiente GEMINI_API_KEY.")
        return False
    return True

def load_source_chunks(filepath: str) -> List[Dict[str, Any]]:
    """Carica i chunk dal file JSON (identica a prima)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        logger.info(f"Caricati {len(chunks_data)} chunk sorgente da '{filepath}'.")
        if not isinstance(chunks_data, list):
            logger.error("Il file JSON non contiene una lista di chunk.")
            return []
        return chunks_data
    except Exception as e:
        logger.error(f"Errore nel caricamento dei chunk sorgente: {e}")
        return []

def normalize_entity_name(entity_name: str) -> str:
    """Normalizzazione nomi entità (identica a prima)"""
    return EntityNormalizer.normalize_entity_name(entity_name)

def adapt_gemini_output(gemini_nodes: List[Dict], gemini_relations: List[Dict], original_chunk: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Adatta output Gemini (identica a prima)"""
    final_entities, final_relations = [], []
    
    # Estrai i metadati dal chunk originale
    original_chunk_id = original_chunk.get("chunk_id", "ID_non_trovato")
    source_file = original_chunk.get("source_file", "File_sconosciuto")
    page_number = original_chunk.get("page_number")
    
    node_id_to_name = {node.get('id'): node.get('properties', {}).get('name') for node in gemini_nodes if node.get('id') and node.get('properties', {}).get('name')}
    
    for node in gemini_nodes:
        properties = node.get('properties', {})
        entity_name = properties.get('name')
        if not entity_name: continue
        final_entities.append({
            "nome_entita": normalize_entity_name(entity_name),
            "tipo_entita": node.get('label', 'Unknown'),
            "descrizione_entita": properties.get('description', f"Entità estratta da {source_file}."),
            "source_chunk_id": original_chunk_id,
            "source_page_number": page_number,
            "source_section_title": None
        })
        
    for rel in gemini_relations:
        soggetto_name = node_id_to_name.get(rel.get('start_node_id'))
        oggetto_name = node_id_to_name.get(rel.get('end_node_id'))
        predicato = rel.get('type')
        if not (soggetto_name and oggetto_name and predicato):
            logger.warning(f"Relazione incompleta saltata nel chunk {original_chunk_id}: {rel}")
            continue
        final_relations.append({
            "soggetto": normalize_entity_name(soggetto_name), 
            "predicato": predicato, 
            "oggetto": normalize_entity_name(oggetto_name),
            "contesto_relazione": rel.get('properties', {}).get('context', f"Relazione estratta da {source_file}."),
            "source_chunk_id": original_chunk_id,
            "source_page_number": page_number,
            "source_section_title": None
        })
    return final_entities, final_relations

def save_output_json(data: List[Dict], filepath: str, description: str):
    """Salvataggio JSON (identica a prima)"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"{description} salvate con successo in '{filepath}' ({len(data)} elementi)")
    except Exception as e:
        logger.error(f"Errore nel salvataggio del file {description}: {e}")

async def process_chunk_batch(extractor: GeminiExtractor, chunk_batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Processa un batch di chunk in parallelo"""
    tasks = []
    
    # Crea le task per tutti i chunk del batch
    for chunk in chunk_batch:
        chunk_id = chunk.get("chunk_id", "unknown")
        chunk_text = chunk.get("text", "").strip()
        
        if not chunk_text:
            logger.warning(f"Chunk {chunk_id} vuoto, saltato")
            continue
        
        # Prepara il prompt
        prompt = PROMPT_TEMPLATE.format(
            node_labels=json.dumps(NODE_LABELS),
            relationship_types=json.dumps(RELATIONSHIP_TYPES),
            text=chunk_text
        )
        
        # Crea la task asincrona
        task = extractor.extract_async(prompt, chunk_id)
        tasks.append((task, chunk))
    
    if not tasks:
        return [], []
    
    # Esegui tutte le task in parallelo
    logger.info(f"Processando {len(tasks)} chunk in parallelo...")
    results = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)
    
    # Aggrega i risultati
    all_entities = []
    all_relations = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Errore nel processamento: {result}")
            continue
        
        _, chunk = tasks[i]
        nodes = result.get("nodes", [])
        relations = result.get("relationships", [])
        
        if nodes or relations:
            entities, relations_adapted = adapt_gemini_output(nodes, relations, chunk)
            all_entities.extend(entities)
            all_relations.extend(relations_adapted)
            logger.debug(f"Chunk {chunk.get('chunk_id')}: +{len(entities)} entità, +{len(relations_adapted)} relazioni")
    
    return all_entities, all_relations

async def main_async() -> int:
    """Versione asincrona della funzione main"""
    if not validate_configuration():
        return 1

    input_json_path = "data/processed/processed_chunks_toc_enhanced.json"
    output_entities_raw_path = "kg_entities_raw_empulia_gemini_v2_async.json"
    output_relations_raw_path = "kg_relations_raw_empulia_gemini_v2_async.json"

    # 1. Carica i chunk sorgente
    source_chunks = load_source_chunks(input_json_path)
    if not source_chunks:
        logger.error("Nessun chunk da processare. Esecuzione terminata.")
        return 1

    logger.info(f"🚀 AVVIO PROCESSAMENTO ASINCRONO")
    logger.info(f"📊 Chunk totali: {len(source_chunks)}")
    logger.info(f"⚡ Richieste simultanee: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"📦 Dimensione batch: {BATCH_SIZE}")
    
    # 2. Inizializza l'estrattore
    extractor = GeminiExtractor(GEMINI_API_KEY, LLM_MODEL_EXTRACTION, MAX_CONCURRENT_REQUESTS)
    
    # 3. Processa in batch
    all_final_entities = []
    all_final_relations = []
    
    start_time = time.time()
    
    # Dividi i chunk in batch
    for i in range(0, len(source_chunks), BATCH_SIZE):
        batch = source_chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(source_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"--- 📦 BATCH {batch_num}/{total_batches} ({len(batch)} chunk) ---")
        batch_start_time = time.time()
        
        # Processa il batch in parallelo
        batch_entities, batch_relations = await process_chunk_batch(extractor, batch)
        
        # Aggrega i risultati
        all_final_entities.extend(batch_entities)
        all_final_relations.extend(batch_relations)
        
        batch_time = time.time() - batch_start_time
        logger.info(f"✅ Batch {batch_num} completato in {batch_time:.2f}s")
        logger.info(f"   📈 +{len(batch_entities)} entità, +{len(batch_relations)} relazioni")
        logger.info(f"   🎯 Totale: {len(all_final_entities)} entità, {len(all_final_relations)} relazioni")
    
    total_time = time.time() - start_time
    
    # 4. Salva i risultati
    save_output_json(all_final_entities, output_entities_raw_path, "Entità RAW finali (ASYNC)")
    save_output_json(all_final_relations, output_relations_raw_path, "Relazioni RAW finali (ASYNC)")

    # 5. Statistiche finali
    logger.info("=" * 60)
    logger.info("🎉 PROCESSAMENTO ASINCRONO COMPLETATO")
    logger.info(f"⏱️  Tempo totale: {total_time:.2f} secondi")
    logger.info(f"⚡ Velocità: {len(source_chunks)/total_time:.2f} chunk/secondo")
    logger.info(f"📊 Totale entità: {len(all_final_entities)}")
    logger.info(f"📊 Totale relazioni: {len(all_final_relations)}")
    
    # Confronto con versione sequenziale stimata
    estimated_sequential_time = len(source_chunks) * 2.5  # ~2.5s per chunk
    speedup = estimated_sequential_time / total_time
    logger.info(f"🚀 Speedup stimato: {speedup:.1f}x più veloce della versione sequenziale")
    logger.info("=" * 60)
    
    return 0

def main() -> int:
    """Wrapper per eseguire la versione asincrona"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    exit(main())