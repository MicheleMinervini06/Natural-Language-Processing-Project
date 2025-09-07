import json
import logging
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import sqlite3
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from query_analyzer_rawData import analyze_user_question 
from utils.context_reranker import ContextReranker

# --- Configurazione ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password"
NEO4J_DATABASE = "test"
DEBUG_LEVEL = "INFO"
VECTOR_INDEX_NAME = "node_text_embeddings"

# Configurazione per il Reranking
INITIAL_RETRIEVAL_TOP_K = 20 # Quanti candidati recuperare prima del reranking
RERANKED_TOP_N = 5 # Quanti candidati tenere dopo il reranking

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class KnowledgeRetriever:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_database, all_chunks_filepath, debug_level="INFO"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, debug_level.upper()))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        try:
            self.embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
            self.logger.info("Modello di embedding caricato.")
        except Exception as e:
            self.logger.error(f"Errore inizializzazione embedder: {e}")
            self.embedder = None

        # ### <<< MODIFICA: Inizializza l'istanza del Reranker >>> ###
        self.reranker = ContextReranker()
            
    def _create_chunks_db(self, filepath: str) -> str: # Logica invariata
        db_path = filepath.replace('.json', '.db')
        if os.path.exists(db_path): return db_path
        try:
            with open(filepath, 'r', encoding='utf-8') as f: chunks_list = json.load(f)
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, text TEXT, source_file TEXT, page_number INTEGER, section_title TEXT)')
            for chunk in chunks_list:
                conn.execute('INSERT INTO chunks VALUES (?, ?, ?, ?, ?)', (chunk.get('chunk_id'), chunk.get('text', ''), chunk.get('source_file', ''), chunk.get('page_number'), chunk.get('section_title', '')))
            conn.commit()
            conn.close()
            return db_path
        except Exception as e:
            self.logger.error(f"Errore nella creazione del database chunks: {e}")
            return ""

    @lru_cache(maxsize=128)
    def _get_chunk_by_id(self, chunk_id: str) -> Dict: # Logica invariata
        if not self.chunks_db_path: return {}
        try:
            conn = sqlite3.connect(self.chunks_db_path)
            cursor = conn.execute('SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            conn.close()
            if row: return {'chunk_id': row[0], 'text': row[1], 'source_file': row[2], 'page_number': row[3], 'section_title': row[4]}
            return {}
        except Exception as e:
            self.logger.error(f"Errore nel recupero del chunk {chunk_id}: {e}")
            return {}

    def close(self):
        if self.driver: self.driver.close()

    def _run_cypher_query(self, query: str, parameters: Dict = None) -> List[Dict]: # Logica invariata
        if not self.driver: return []
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                return [record.data() for record in session.run(query, parameters or {})]
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione della query Cypher: {e}")
            return []

    @lru_cache(maxsize=32)
    def _embed_query(self, text: str) -> List[float]: # Logica invariata
        if not self.embedder: return []
        try:
            return self.embedder.embed_query(text)
        except Exception as e:
            self.logger.error(f"Errore durante la generazione dell'embedding della query: {e}")
            return []

    def _hybrid_retrieval(self, search_terms: List[str], query_embedding: List[float], top_k: int) -> List[Dict]: # Logica invariata
        keyword_query = """
        UNWIND $search_terms as term MATCH (node:KnowledgeNode)
        WHERE toLower(node.name) CONTAINS toLower(term) OR ANY(original IN node.original_names WHERE toLower(original) CONTAINS toLower(term))
        RETURN node, elementId(node) as element_id
        """
        keyword_results = self._run_cypher_query(keyword_query, {"search_terms": search_terms})
        vector_results = []
        if query_embedding:
            vector_query = """
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score RETURN node, elementId(node) as element_id
            """
            vector_results = self._run_cypher_query(vector_query, {"index_name": VECTOR_INDEX_NAME, "top_k": top_k, "query_embedding": query_embedding})
        anchor_ids = {record['element_id'] for record in keyword_results + vector_results}
        if not anchor_ids: return []
        self.logger.info(f"Fase di ancoraggio ibrida ha trovato {len(anchor_ids)} nodi unici.")
        expand_query = """
        MATCH (anchor:KnowledgeNode) WHERE elementId(anchor) IN $anchor_ids
        OPTIONAL MATCH (same_chunk_neighbor:KnowledgeNode)
        WHERE same_chunk_neighbor.source_chunk_id = anchor.source_chunk_id AND elementId(same_chunk_neighbor) <> elementId(anchor)
        OPTIONAL MATCH (anchor)-[r]-(direct_neighbor:KnowledgeNode)
        WITH COLLECT(DISTINCT anchor) + COLLECT(DISTINCT same_chunk_neighbor) + COLLECT(DISTINCT direct_neighbor) as all_nodes
        UNWIND all_nodes as node RETURN DISTINCT node
        """
        return self._run_cypher_query(expand_query, {"anchor_ids": list(anchor_ids)})

    def _format_context_from_subgraph(self, subgraph_nodes: List[Dict]) -> Tuple[str, set]: # Logica invariata
        if not subgraph_nodes: return "Nessuna informazione trovata nel Knowledge Graph.", set()
        context_str = "Informazioni rilevanti trovate nel Knowledge Graph:\n\n"
        source_chunk_ids, nodes_processed = set(), set()
        nodes_to_process = [record['node'] for record in subgraph_nodes if record.get('node')]
        for node_data in nodes_to_process:
            if not node_data: continue
            node_id = str(node_data.get('name', '')) + str(node_data.get('type', ''))
            if node_id in nodes_processed: continue
            nodes_processed.add(node_id)
            if node_data.get('source_chunk_id'): source_chunk_ids.add(node_data['source_chunk_id'])
            context_str += f"✓ **Entità trovata**: {node_data.get('name', 'N/A')}\n"
            if node_data.get('type'): context_str += f"  • Tipo: {node_data['type']}\n"
            if node_data.get('description'): context_str += f"  • Descrizione: {node_data['description']}\n"
            if node_data.get('original_names'): context_str += f"  • Varianti trovate: {', '.join(node_data['original_names'][:3])}{'...' if len(node_data['original_names']) > 3 else ''}\n"
            context_str += "\n"
        return context_str.strip(), source_chunk_ids

    def retrieve_knowledge(self, analysis: Dict[str, Any], retrieve_text: bool = True) -> Dict[str, Any]:
        self.logger.info("Avvio recupero ibrido con Reranking dal Knowledge Graph")
        user_question = analysis.get("domanda_originale", "")
        search_terms = analysis.get("termini_di_ricerca_espansi", [])
        entity_names = [e.get("nome", "") for e in analysis.get("entita_chiave", [])]
        final_search_terms = list(set([term.lower() for term in search_terms + entity_names if term]))
        query_embedding = self._embed_query(user_question)
        
        # 1. RECUPERO AMPIO
        subgraph_results = self._hybrid_retrieval(final_search_terms, query_embedding, top_k=INITIAL_RETRIEVAL_TOP_K)
        graph_context, source_chunk_ids = self._format_context_from_subgraph(subgraph_results)
        
        text_context = ""
        if retrieve_text and source_chunk_ids:
            # 2. COLLEZIONE DEI CHUNK CANDIDATI
            self.logger.info(f"Recupero del testo originale da {len(source_chunk_ids)} chunk candidati...")
            candidate_chunks = [chunk for chunk in [self._get_chunk_by_id(cid) for cid in source_chunk_ids] if chunk]

            # 3. RERANKING E FILTRAGGIO
            reranked_chunks = self.reranker.rerank(user_question, candidate_chunks)
            final_chunks = reranked_chunks[:RERANKED_TOP_N]
            self.logger.info(f"Contesto finale costruito con i top {len(final_chunks)} chunk dopo il reranking.")
            
            # 4. COSTRUZIONE DEL CONTESTO FINALE
            text_context += "\n\n--- Testo Originale dalle Guide per Contesto Aggiuntivo ---\n\n"
            for chunk in final_chunks:
                text_context += f"Fonte: {chunk.get('source_file')} - Pagina {chunk.get('page_number')} - Sezione '{chunk.get('section_title') or 'N/A'}'\n"
                text_context += "```\n" + str(chunk.get('text', '')) + "\n```\n\n"
        
        self.logger.info("Recupero conoscenza con reranking completato")
        return {"graph_context": graph_context, "text_context": text_context.strip()}

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, DEBUG_LEVEL.upper()), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not GEMINI_API_KEY:
        print("ERRORE: La variabile d'ambiente GEMINI_API_KEY non è impostata.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        all_chunks_file_path = 'data/processed/processed_chunks_toc_enhanced.json' 
        retriever = KnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, all_chunks_file_path, DEBUG_LEVEL)
        if retriever.driver:
            test_question = "Sono un utente di una Stazione Appaltante e ho appena ricevuto le nuove credenziali per EmPULIA. Qual è il primo passo che devo compiere dopo aver fatto l'accesso?"
            analysis_result = analyze_user_question(test_question)
            if analysis_result:
                retrieved_context = retriever.retrieve_knowledge(analysis_result, retrieve_text=True)
                print("\n" + "="*50)
                print("RISULTATO RECUPERO CONOSCENZA (v5 - Con Reranker Separato)")
                print("="*50 + "\n")
                print(f"DOMANDA: {test_question}\n")
                print("--- Analisi e Termini Espansi ---")
                print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
                print("\n--- Contesto dal Grafo (Pre-Reranking) ---")
                print(retrieved_context["graph_context"])
                print("\n--- Contesto dal Testo Originale (Post-Reranking) ---")
                print(retrieved_context["text_context"])
            retriever.close()
        else:
            print("Impossibile eseguire il test a causa di un errore di connessione a Neo4j.")