import json
import logging
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import sqlite3
import os
from utils.entity_normalizer import EntityNormalizer, create_search_patterns
from query_analyzer import analyze_user_question

# --- Configurazione Neo4j ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password"
NEO4J_DATABASE = "test"

# --- Configurazione Debug ---
DEBUG_LEVEL = "INFO"  # Cambia a: "DEBUG", "INFO", "WARNING", "ERROR"
DEBUG_TO_FILE = False  # True per salvare i log in file

class KnowledgeRetriever:
    """
    Classe per recuperare conoscenza dal Knowledge Graph in Neo4j
    e opzionalmente dai chunk di testo originali.
    """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_database, all_chunks_filepath, debug_level="INFO"):
        # Configura il logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, debug_level.upper()))
        
        # Configura handler se non già configurato
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password), database=neo4j_database)
            self.driver.verify_connectivity()
            self.logger.info("Connessione a Neo4j per il recupero stabilita.")
        except Exception as e:
            self.logger.error(f"Errore di connessione a Neo4j: {e}")
            self.driver = None

            raise ConnectionError(f"Impossibile connettersi a Neo4j: {e}")
        
        self.chunks_db_path = self._create_chunks_db(all_chunks_filepath)
    
    def _create_chunks_db(self, filepath: str) -> str:
        """Crea un database SQLite per i chunk per un accesso più efficiente."""
        db_path = filepath.replace('.json', '.db')
        
        if os.path.exists(db_path):
            return db_path
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks_list = json.load(f)
            
            conn = sqlite3.connect(db_path)
            conn.execute('''
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT,
                    source_file TEXT,
                    page_number INTEGER,
                    section_title TEXT
                )
            ''')
            
            for chunk in chunks_list:
                conn.execute('''
                    INSERT INTO chunks VALUES (?, ?, ?, ?, ?)
                ''', (
                    chunk['chunk_id'],
                    chunk.get('text', ''),
                    chunk.get('source_file', ''),
                    chunk.get('page_number', 0),
                    chunk.get('section_title', '')
                ))
            
            conn.commit()
            conn.close()
            return db_path
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione del database chunks: {e}")
            return ""
    
    @lru_cache(maxsize=128)
    def _get_chunk_by_id(self, chunk_id: str) -> Dict:
        """Recupera un chunk dal database con caching."""
        if not self.chunks_db_path:
            return {}
            
        try:
            conn = sqlite3.connect(self.chunks_db_path)
            cursor = conn.execute(
                'SELECT * FROM chunks WHERE chunk_id = ?', 
                (chunk_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'chunk_id': row[0],
                    'text': row[1],
                    'source_file': row[2],
                    'page_number': row[3],
                    'section_title': row[4]
                }
            return {}
        except Exception as e:
            self.logger.error(f"Errore nel recupero del chunk {chunk_id}: {e}")
            return {}

    def close(self):
        if self.driver:
            self.driver.close()

    def _run_cypher_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Esegue una query Cypher con gestione robusta degli errori."""
        if not self.driver:
            self.logger.error("Driver Neo4j non disponibile")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione della query Cypher: {e}")
            return []

    @lru_cache(maxsize=1)
    def _get_available_relationships(self) -> List[str]:
        """Recupera tutte le relazioni disponibili nel grafo."""
        query = "MATCH ()-[r]->() RETURN DISTINCT type(r) as relationship_type"
        results = self._run_cypher_query(query)
        return [r['relationship_type'] for r in results]

    def _generate_cypher_from_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Genera una query Cypher usando la normalizzazione condivisa.
        """
        intento = analysis.get("intento")
        entita = analysis.get("entita_chiave", [])
        
        if not intento or not entita:
            return ""

        # Usa il normalizzatore condiviso
        normalized_entities = EntityNormalizer.normalize_entity_list(entita)
        
        if not normalized_entities:
            return ""
        
        self.logger.debug(f"Entità normalizzate: {[(e['nome_originale'], e['nome_normalizzato']) for e in normalized_entities]}")
        
        # Crea pattern di ricerca avanzati
        all_patterns = []
        for entity in normalized_entities:
            patterns = create_search_patterns(entity['nome_originale'])
            all_patterns.extend(patterns)
        
        # Rimuovi duplicati mantenendo l'ordine
        unique_patterns = list(dict.fromkeys(all_patterns))
        
        self.logger.debug(f"Pattern di ricerca: {unique_patterns}")
        
        # Crea condizioni di ricerca
        search_conditions = []
        for pattern in unique_patterns:
            search_conditions.extend([
                f'n.name = "{pattern}"',
                f'n.name CONTAINS "{pattern}"'
            ])
        
        search_condition = " OR ".join(search_conditions)
        
        # Usa relazioni reali dal grafo
        available_rels = self._get_available_relationships()
        self.logger.debug(f"Relazioni disponibili: {available_rels[:10]}...")
        
        if intento == "find_procedure":
            # Cerca relazioni procedurali reali
            proc_rels = [r for r in available_rels if any(x in r.lower() for x in 
                        ['passo', 'azione', 'procedura', 'richiede', 'applica', 'campo', 'vincola', 'parte', 'descritto'])]
            if proc_rels:
                rel_pattern = "|".join(proc_rels[:8])  # Usa max 8 relazioni
                query = f"""
                MATCH (n) WHERE {search_condition}
                OPTIONAL MATCH p1=(n)-[:{rel_pattern}*1..2]->(connected)
                OPTIONAL MATCH p2=(n)<-[:{rel_pattern}*1..2]-(source)
                RETURN n, p1, p2 LIMIT 20
                """
            else:
                query = f"""
                MATCH (n) WHERE {search_condition}
                OPTIONAL MATCH p=(n)-[r]-(connected)
                RETURN n, p LIMIT 20
                """
        else:
            query = f"""
            MATCH (n) WHERE {search_condition}
            OPTIONAL MATCH p=(n)-[r]-(connected)
            RETURN n, p LIMIT 20
            """

        self.logger.debug(f"Query Cypher generata:\n{query}")
        return query

    def test_normalization(self):
        """Test della normalizzazione per debug."""
        test_cases = [
            "cambiare password",
            "password",
            "EmPULIA",
            "piattaforma EmPULIA",
            "accesso",
            "login",
            "modificare credenziali"
        ]
        
        print("=== TEST NORMALIZZAZIONE ===")
        for test in test_cases:
            normalized = EntityNormalizer.normalize_entity_name(test)
            patterns = create_search_patterns(test)
            print(f"'{test}' → '{normalized}' | Patterns: {patterns}")
        print("=============================")

    def _format_context_from_results(self, records: List[Dict]) -> Tuple[str, set]:
        """
        Formatta i risultati della query Neo4j in una stringa di testo leggibile
        e raccoglie gli ID dei chunk di origine.
        """
        if not records:
            return "Nessuna informazione trovata nel Knowledge Graph.", set()

        context_str = "Informazioni rilevanti trovate nel Knowledge Graph:\n\n"
        source_chunk_ids = set()
        
        nodes_processed = set()
        relationships_processed = set()

        for record in records:
            # Prima processa il nodo principale (chiave 'n')
            if 'n' in record and record['n'] is not None:
                node_data = record['n']
                node_id = str(node_data.get('name', '')) + str(node_data.get('type', ''))
                
                if node_id not in nodes_processed:
                    nodes_processed.add(node_id)
                    
                    # Aggiungi chunk IDs dalle varie possibili chiavi
                    if 'source_chunk_id' in node_data:
                        source_chunk_ids.add(node_data['source_chunk_id'])
                    if 'source_chunk_ids' in node_data:
                        source_chunk_ids.update(node_data['source_chunk_ids'])
                    if 'original_members_chunk_ids' in node_data:
                        source_chunk_ids.update(node_data['original_members_chunk_ids'])
                    
                    # Formatta le informazioni del nodo
                    context_str += f"✓ **Entità trovata**: {node_data.get('name', 'N/A')}\n"
                    if node_data.get('type'):
                        context_str += f"  • Tipo: {node_data['type']}\n"
                    if node_data.get('description'):
                        context_str += f"  • Descrizione: {node_data['description']}\n"
                    if node_data.get('occurrence_count'):
                        context_str += f"  • Occorrenze nel testo: {node_data['occurrence_count']}\n"
                    if node_data.get('original_names'):
                        context_str += f"  • Varianti trovate: {', '.join(node_data['original_names'][:3])}{'...' if len(node_data['original_names']) > 3 else ''}\n"
                    context_str += "\n"
            
            # Processa i path (p1, p2, p, etc.)
            for key, path_data in record.items():
                if key == 'n' or path_data is None:
                    continue
                
                # I path sono liste di nodi e relazioni alternati
                if isinstance(path_data, list) and len(path_data) >= 3:
                    # Formato: [nodo_start, relazione, nodo_end, ...]
                    for i in range(0, len(path_data) - 1, 2):
                        if i + 1 < len(path_data):
                            start_node = path_data[i]
                            relation = path_data[i + 1]
                            end_node = path_data[i + 2] if i + 2 < len(path_data) else None
                            
                            # Processa la relazione
                            if isinstance(relation, str) and end_node and isinstance(start_node, dict) and isinstance(end_node, dict):
                                rel_id = f"{start_node.get('name', 'N/A')}_{relation}_{end_node.get('name', 'N/A')}"
                                
                                if rel_id not in relationships_processed:
                                    relationships_processed.add(rel_id)
                                    
                                    # Aggiungi chunk IDs dai nodi collegati
                                    if 'source_chunk_id' in end_node:
                                        source_chunk_ids.add(end_node['source_chunk_id'])
                                    if 'source_chunk_ids' in end_node:
                                        source_chunk_ids.update(end_node['source_chunk_ids'])
                                    
                                    start_name = start_node.get('name', 'N/A')
                                    end_name = end_node.get('name', 'N/A')
                                    
                                    context_str += f"→ **Relazione**: {start_name} --[{relation}]--> {end_name}\n"
                                    if end_node.get('description'):
                                        context_str += f"  • Dettaglio: {end_node['description']}\n"
                                    context_str += "\n"
        
        # Se non ha processato nulla, prova a mostrare i dati raw per debug
        if not nodes_processed and not relationships_processed:
            context_str += "**Risultati trovati (formato raw):**\n"
            for i, record in enumerate(records[:2]):  # Solo primi 2 record
                context_str += f"- Record {i+1}: {list(record.keys())}\n"
                for key, value in record.items():
                    if value is not None:
                        context_str += f"  • {key}: {type(value)} - {str(value)[:100]}...\n"
        
        return context_str.strip(), source_chunk_ids

    def retrieve_knowledge(self, analysis, retrieve_text: bool = True) -> Dict[str, Any]:
        """
        Funzione principale con strategia di fallback.
        """
        self.logger.info("Avvio recupero conoscenza dal Knowledge Graph")
        
        # Check if input is a string (question) instead of analysis dict
        if isinstance(analysis, str):
            self.logger.info("Input è una stringa, chiamando QueryAnalyzer...")
            analysis = analyze_user_question(analysis)
            self.logger.debug(f"Analisi completata: {analysis}")
        
        # Validate analysis format
        if not isinstance(analysis, dict) or "entita_chiave" not in analysis:
            self.logger.error("Formato di analisi non valido")
            return {
                "graph_context": "Errore: formato di input non valido",
                "text_context": ""
            }
        
        # Tentativo principale
        cypher_query = self._generate_cypher_from_analysis(analysis)
        query_results = self._run_cypher_query(cypher_query) if cypher_query else []
        
        # Se non trova nulla, prova una ricerca più ampia
        if not query_results and analysis.get("entita_chiave"):
            self.logger.info("Tentativo di fallback con ricerca più ampia...")
            entity_name = analysis["entita_chiave"][0].get("nome", "").lower()
            
            # Ricerca molto ampia
            fallback_query = f"""
            MATCH (n) 
            WHERE n.name CONTAINS "{entity_name.split()[0] if entity_name.split() else entity_name}"
            OPTIONAL MATCH p=(n)-[r]-(neighbor)
            RETURN n, p LIMIT 20
            """
            query_results = self._run_cypher_query(fallback_query)
            self.logger.info(f"Fallback ha trovato {len(query_results)} risultati")
        
        if query_results:
            self.logger.debug(f"Primo risultato: {query_results[0]}")
        
        # Formatta il contesto dal grafo
        graph_context, source_chunk_ids = self._format_context_from_results(query_results)
        self.logger.debug(f"Chunk IDs trovati: {source_chunk_ids}")
        
        text_context = ""
        if retrieve_text and source_chunk_ids:
            self.logger.info(f"Recupero del testo originale da {len(source_chunk_ids)} chunk")
            # Recupera il testo dai chunk originali
            relevant_chunks = [self._get_chunk_by_id(chunk_id) for chunk_id in source_chunk_ids]
            # Filtra i chunk non trovati
            relevant_chunks = [chunk for chunk in relevant_chunks if chunk]
            # Ordina i chunk per pagina e ID per coerenza
            relevant_chunks.sort(key=lambda c: (c.get('page_number', 0), c.get('chunk_id', '')))
            
            text_context += "\n\n--- Testo Originale dalle Guide per Contesto Aggiuntivo ---\n\n"
            for chunk in relevant_chunks:
                text_context += f"Fonte: {chunk.get('source_file')} - Pagina {chunk.get('page_number')} - Sezione '{chunk.get('section_title', 'N/A')}'\n"
                text_context += "```\n"
                text_context += chunk.get('text', '')
                text_context += "\n```\n\n"
        
        self.logger.info("Recupero conoscenza completato")
        return {
            "graph_context": graph_context,
            "text_context": text_context.strip()
        }

    def debug_graph_content(self) -> None:
        """Funzione di debug per verificare il contenuto del grafo."""
        if self.logger.level > logging.DEBUG:
            self.logger.info("Debug del grafo disabilitato (impostare DEBUG_LEVEL='DEBUG' per attivarlo)")
            return
            
        queries = [
            "MATCH (n) RETURN count(n) as total_nodes",
            "MATCH ()-[r]->() RETURN count(r) as total_relationships", 
            "MATCH (n) RETURN DISTINCT labels(n) as node_types, count(n) as count ORDER BY count DESC",
            "MATCH (n) WHERE n.name CONTAINS 'password' RETURN n.name, labels(n) LIMIT 10",
            "MATCH (n) WHERE n.name CONTAINS 'password' OPTIONAL MATCH (n)-[r]-(connected) RETURN n.name, labels(n), type(r) as relationship_type, connected.name as connected_node LIMIT 20"
        ]
        
        for query in queries:
            self.logger.debug(f"Query: {query}")
            results = self._run_cypher_query(query)
            for result in results:
                self.logger.debug(f"Result: {result}")

if __name__ == "__main__":
    
    logging.basicConfig(
        level=getattr(logging, DEBUG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log') if DEBUG_TO_FILE else logging.StreamHandler()
        ]
    )
    
    # Simula l'output dell'analizzatore per una domanda specifica
    simulated_analysis = {
  "intento": "find_procedure",
  "entita_chiave": [
    {
      "nome": "calcolo anomalia",
      "tipo": "OperazioneSistema"
    },
    {
      "nome": "piattaforma",
      "tipo": "PiattaformaModulo"
    }
  ],
  "domanda_originale": "Come viene gestito il calcolo dell'anomalia dalla piattaforma?"
}

    # Percorso al file JSON che contiene TUTTI i chunk originali
    all_chunks_file_path = 'data\\processed\\processed_chunks_toc_enhanced.json' 
    
    # Inizializza il retriever con il livello di debug desiderato
    retriever = KnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, all_chunks_file_path, DEBUG_LEVEL)

    if retriever.driver:
        # Esegui il recupero
        retrieved_context = retriever.retrieve_knowledge("Come posso creare una commissione di gara su EmPULIA?", retrieve_text=True)

        print("\n" + "="*50)
        print("RISULTATO RECUPERO CONOSCENZA")
        print("="*50 + "\n")

        print("--- Contesto dal Grafo ---")
        print(retrieved_context["graph_context"])

        print("\n--- Contesto dal Testo Originale ---")
        print(retrieved_context["text_context"])
        
        # Debug del contenuto del grafo (solo se DEBUG_LEVEL="DEBUG")
        retriever.debug_graph_content()
        
        retriever.close()
    else:
        print("Impossibile eseguire il test a causa di un errore di connessione a Neo4j.")