import json
import logging
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import sqlite3
import os
import sys

# Aggiungi 'src' al path per permettere l'import da altri moduli nella stessa cartella
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from query_analyzer import analyze_user_question

# --- Configurazione ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password"
NEO4J_DATABASE = "testaggregated"    

class KnowledgeRetriever:
    """
    Classe per recuperare conoscenza dal Knowledge Graph in Neo4j
    e opzionalmente dai chunk di testo originali.
    PROGETTATA PER UN GRAFO CON:
    - Nodi: etichetta :Entity e proprietà .type
    - Relazioni: etichetta :RELATED e proprietà .type
    """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_database, processed_chunks_filepath):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))
            self.database = neo4j_database
            self.driver.verify_connectivity()
            self.logger.info(f"Connessione a Neo4j ({neo4j_uri}, DB: '{neo4j_database}') stabilita.")
        except Exception as e:
            self.logger.error(f"Errore di connessione a Neo4j: {e}")
            self.driver = None
            raise ConnectionError(f"Impossibile connettersi a Neo4j: {e}")
        
        self.chunks_db_path = self._create_chunks_db(processed_chunks_filepath)

    def _create_chunks_db(self, filepath: str) -> str:
        db_path = filepath.replace('.json', '.db')
        if os.path.exists(db_path):
            self.logger.info(f"Database SQLite dei chunk già esistente in '{db_path}'.")
            return db_path
        
        self.logger.info(f"Creazione del database SQLite dei chunk da '{filepath}'...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks_list = json.load(f)
            
            conn = sqlite3.connect(db_path)
            conn.execute('''CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY, 
                text TEXT, 
                source_file TEXT, 
                page_number INTEGER, 
                section_title TEXT
            )''')
            
            for chunk in chunks_list:
                conn.execute('INSERT INTO chunks VALUES (?, ?, ?, ?, ?)', (
                    chunk['chunk_id'], chunk.get('text', ''), chunk.get('source_file', ''), 
                    chunk.get('page_number', 0), chunk.get('section_title', '')
                ))
            
            conn.commit()
            conn.close()
            self.logger.info(f"Database SQLite dei chunk creato con successo in '{db_path}'.")
            return db_path
        except Exception as e:
            self.logger.error(f"Errore nella creazione del database chunks: {e}")
            return ""

    @lru_cache(maxsize=256)
    def _get_chunk_by_id(self, chunk_id: str) -> Dict:
        if not self.chunks_db_path: return {}
        try:
            conn = sqlite3.connect(self.chunks_db_path, uri=True) # uri=True per la thread safety
            cursor = conn.execute('SELECT chunk_id, text, source_file, page_number, section_title FROM chunks WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {'chunk_id': row[0], 'text': row[1], 'source_file': row[2], 'page_number': row[3], 'section_title': row[4]}
            return {}
        except Exception as e:
            self.logger.error(f"Errore nel recupero del chunk {chunk_id}: {e}")
            return {}

    def close(self):
        if self.driver:
            self.driver.close()
            self.logger.info("Connessione a Neo4j chiusa.")

    def _run_cypher_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        if not self.driver:
            self.logger.error("Driver Neo4j non disponibile")
            return []
        try:
            records, _, _ = self.driver.execute_query(
                query,
                parameters or {},
                database_=self.database,
            )
            return [r.data() for r in records]
        except Exception as e:
            self.logger.error(f"Errore Cypher: {e}\nQuery: {query}\nParams: {parameters}")
            return []

    def _generate_cypher_from_analysis(self, analysis: Dict[str, Any]) -> Tuple[str, Dict]:
        intento = analysis.get("intento")
        entita = analysis.get("entita_chiave", [])
        
        if not intento or not entita:
            self.logger.warning("Analisi vuota, impossibile generare query.")
            return "", {}

        entity_names = [e.get("nome") for e in entita if e.get("nome")]
        if not entity_names:
            return "", {}

        query = ""
        params = {"entity_names": entity_names}

        if intento == "find_procedure":
            query = """
            MATCH p=(start_node:Entity)-[rels:RELATED*1..5]->(passo:Entity)
            WHERE start_node.name IN $entity_names
              AND start_node.type IN ['AzioneUtente', 'FunzionalitàPiattaforma', 'PiattaformaModulo']
              AND ALL(r IN rels WHERE r.type IN ['hapassosuccessivo', 'includeoperazione'])
            RETURN p ORDER BY length(p) DESC LIMIT 3
            """
        elif intento == "find_requirements":
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $entity_names
            OPTIONAL MATCH p1=(e)-[r1:RELATED]->(prereq) WHERE r1.type = 'haprequisito'
            OPTIONAL MATCH p2=(e)-[r2:RELATED]->(doc) WHERE r2.type = 'richiededocumento'
            RETURN p1, p2
            """
        elif intento == "find_definition":
            params = {"entity_name": entity_names[0]}
            query = "MATCH p=(start_node:Entity {name: $entity_name})-[r:RELATED]-(neighbor:Entity) RETURN p LIMIT 15"
        elif intento == "find_relationship" and len(entity_names) > 1:
            params = {"start_name": entity_names[0], "end_name": entity_names[1]}
            query = "MATCH (start_node:Entity {name: $start_name}), (end_node:Entity {name: $end_name}) MATCH p = allShortestPaths((start_node)-[:RELATED*..10]-(end_node)) RETURN p LIMIT 1"
        else: # Fallback
            params = {"entity_name": entity_names[0]}
            query = "MATCH p=(start_node:Entity {name: $entity_name})-[r:RELATED]-(neighbor:Entity) RETURN p LIMIT 5"

        self.logger.info(f"Query Cypher generata (modello clusterizzato):\n{query.strip()}")
        return query, params

    def _format_context_from_results(self, records: List[Dict]) -> Tuple[str, set]:
        if not records:
            return "Nessuna informazione trovata nel Knowledge Graph.", set()

        context_str = "Informazioni rilevanti trovate nel Knowledge Graph:\n"
        source_chunk_ids = set()
        nodes_seen = set()

        for record in records:
            for key, value in record.items():
                if value is None: continue

                path_like_object = None
                if isinstance(value, dict) and 'nodes' in value and 'relationships' in value:
                    path_like_object = value
                elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                    path_like_object = {'nodes': [dict(n) for n in value.nodes], 'relationships': [dict(r) for r in value.relationships]}

                if path_like_object:
                    for node_props in path_like_object.get('nodes', []):
                        node_name = node_props.get('name')
                        if node_name and node_name not in nodes_seen:
                            nodes_seen.add(node_name)
                            # --- MODIFICA CHIAVE QUI ---
                            chunk_sources = node_props.get('source_chunk_ids', [])
                            if isinstance(chunk_sources, list):
                                source_chunk_ids.update(chunk_sources)
                            
                            context_str += f"\n- Entità: {node_name} (Tipo: {node_props.get('type', 'N/A')})"
                            descriptions = node_props.get('descriptions', [])
                            if descriptions:
                                context_str += f"\n  Descrizione: {descriptions[0]}"
                
                # Aggiungi qui anche la logica per i singoli nodi nel caso di fallback
                elif isinstance(value, dict) and 'name' in value and 'type' in value:
                     node_props = value
                     node_name = node_props.get('name')
                     if node_name and node_name not in nodes_seen:
                        nodes_seen.add(node_name)
                        chunk_sources = node_props.get('source_chunk_ids', [])
                        if isinstance(chunk_sources, list):
                            source_chunk_ids.update(chunk_sources)
                        
                        context_str += f"\n- Entità: {node_name} (Tipo: {node_props.get('type', 'N/A')})"
                        descriptions = node_props.get('descriptions', [])
                        if descriptions:
                            context_str += f"\n  Descrizione: {descriptions[0]}"
        
        self.logger.info(f"Estratti {len(source_chunk_ids)} chunk IDs dal grafo: {list(source_chunk_ids)[:5]}...")
        
        if not nodes_seen: # Se non ha processato nulla di significativo
            return "Nessuna informazione trovata nel Knowledge Graph.", set()

        return context_str.strip(), source_chunk_ids

    def retrieve_knowledge(self, user_question: str, retrieve_text: bool = True) -> Dict[str, Any]:
        self.logger.info(f"Avvio recupero conoscenza per: '{user_question}'")
        analysis = analyze_user_question(user_question)
        self.logger.debug(f"Analisi completata: {analysis}")
        
        if not isinstance(analysis, dict) or not analysis.get("entita_chiave"):
            self.logger.warning("Formato di analisi non valido o nessuna entità estratta.")
            return {"graph_context": "Impossibile analizzare la domanda.", "text_context": "", "source_chunk_ids": set()}
        
        cypher_query, params = self._generate_cypher_from_analysis(analysis)
        query_results = self._run_cypher_query(cypher_query, parameters=params) if cypher_query else []
        
        if not query_results:
            self.logger.info("Query principale non ha prodotto risultati. Tentativo di fallback...")
            entity_name = analysis["entita_chiave"][0].get("nome", "")
            
            fallback_query = """
            MATCH (n:Entity) 
            WHERE toLower(n.name) CONTAINS toLower($entity_name)
            OPTIONAL MATCH p=(n)-[r:RELATED]-(neighbor)
            RETURN p, n LIMIT 10
            """
            fallback_params = {"entity_name": entity_name}
            query_results = self._run_cypher_query(fallback_query, parameters=fallback_params)
            self.logger.info(f"Fallback ha trovato {len(query_results)} risultati.")

        graph_context, source_chunk_ids = self._format_context_from_results(query_results)
        
        text_context = ""
        if retrieve_text and source_chunk_ids:
            self.logger.info(f"Recupero del testo originale da {len(source_chunk_ids)} chunk(s)")
            relevant_chunks = [self._get_chunk_by_id(chunk_id) for chunk_id in source_chunk_ids]
            relevant_chunks = [chunk for chunk in relevant_chunks if chunk]
            relevant_chunks.sort(key=lambda c: (c.get('page_number', 0), c.get('chunk_id', '')))
            
            if relevant_chunks:
                text_context += "\n\n--- Testo Originale dalle Guide per Contesto Aggiuntivo ---\n\n"
                for chunk in relevant_chunks:
                    text_context += f"Fonte: {chunk.get('source_file')} - Pagina {chunk.get('page_number')} - Sezione '{chunk.get('section_title', 'N/A')}'\n"
                    text_context += "```\n"
                    text_context += chunk.get('text', '')
                    text_context += "\n```\n\n"

        self.logger.info("Recupero conoscenza completato.")
        return {
            "graph_context": graph_context,
            "text_context": text_context.strip(),
            "source_chunk_ids": source_chunk_ids
        }

if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    
    processed_chunks_file_path = 'data/processed/processed_chunks_toc_enhanced.json' 
    
    try:
        retriever = KnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, processed_chunks_file_path)
        
        test_question = "Come posso creare una commissione di gara su EmPULIA?"
        retrieved_context = retriever.retrieve_knowledge(test_question, retrieve_text=True)

        print("\n" + "="*50)
        print("RISULTATO RECUPERO CONOSCENZA")
        print("="*50 + "\n")
        print("--- Contesto dal Grafo ---")
        print(retrieved_context["graph_context"])
        print("\n--- Testo Originale ---")
        print(retrieved_context["text_context"][:1000] + "..." if len(retrieved_context["text_context"]) > 1000 else retrieved_context["text_context"])
        
        retriever.close()
    except ConnectionError:
        print("Esecuzione terminata a causa di errore di connessione a Neo4j.")