import json
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any

# --- Configurazione Neo4j ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password" # La password che hai impostato durante la creazione del DB

class Neo4jUploader:
    """
    Classe helper per interagire con un database Neo4j.
    """
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            self.driver.verify_connectivity()
            print("Connessione a Neo4j stabilita con successo.")
        except Exception as e:
            print(f"Errore: Impossibile connettersi a Neo4j. Verifica l'URI, le credenziali e che il database sia in esecuzione. Dettagli: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            print("Connessione a Neo4j chiusa.")

    def run_query(self, query, parameters=None, write_operation=True):
        """Esegue una query Cypher e gestisce la transazione."""
        if not self.driver:
            print("Errore: Driver Neo4j non disponibile.")
            return []
        
        with self.driver.session() as session:
            if write_operation:
                result = session.execute_write(self._execute_query, query, parameters)
            else:
                result = session.execute_read(self._execute_query, query, parameters)
            return result

    @staticmethod
    def _execute_query(tx, query, parameters=None):
        """Funzione helper eseguita all'interno di una transazione."""
        result = tx.run(query, parameters)
        return [record for record in result]
        
    def clear_database(self):
        """Pulisce l'intero database. Usare con cautela!"""
        print("ATTENZIONE: Pulizia dell'intero database in corso...")
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)
        print("Database pulito.")

    def setup_constraints(self):
        """Imposta vincoli di unicità sui nodi per performance e integrità dei dati."""
        print("Impostazione dei vincoli di unicità...")
        # Vincolo sul nome dell'entità. Questo è il più importante.
        # Rende le operazioni di MERGE molto più veloci.
        query = "CREATE CONSTRAINT unique_entity_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE"
        self.run_query(query)
        print("Vincoli impostati.")

    def upload_entities(self, entities: List[Dict]):
        """Crea i nodi nel grafo usando UNWIND per efficienza."""
        if not entities:
            print("Nessuna entità da caricare.")
            return
            
        print(f"Inizio caricamento di {len(entities)} nodi...")
        
        # Versione senza APOC (più compatibile)
        query = """
        UNWIND $entities AS entity
        MERGE (n:Entity {name: entity.nome_entita_cluster})
        ON CREATE SET 
            n.type = entity.tipo_entita_cluster,
            n.descriptions = entity.descrizioni_aggregate,
            n.occurrence_count = entity.conteggio_occorrenze_totale,
            n.original_members = entity.membri_cluster
        // Aggiunge etichetta dinamica usando CALL + FOREACH
        WITH n, entity
        CALL {
            WITH n, entity
            CALL apoc.create.addLabels(n, [entity.tipo_entita_cluster]) YIELD node
            RETURN node
        }
        RETURN count(n) AS created_nodes
        """
        
        # Fallback senza APOC se necessario
        try:
            result = self.run_query(query, parameters={"entities": entities})
            print(f"Caricamento nodi completato con APOC.")
        except Exception as e:
            print(f"APOC non disponibile, uso metodo alternativo: {e}")
            # Query alternativa senza APOC
            query_fallback = """
            UNWIND $entities AS entity
            MERGE (n:Entity {name: entity.nome_entita_cluster})
            ON CREATE SET 
                n.type = entity.tipo_entita_cluster,
                n.descriptions = entity.descrizioni_aggregate,
                n.occurrence_count = entity.conteggio_occorrenze_totale,
                n.original_members = entity.membri_cluster
            RETURN count(n) AS created_nodes
            """
            result = self.run_query(query_fallback, parameters={"entities": entities})
            print(f"Caricamento nodi completato senza APOC.")

    def upload_relations(self, relations: List[Dict]):
        """Crea le relazioni tra i nodi esistenti."""
        if not relations:
            print("Nessuna relazione da caricare.")
            return

        print(f"Inizio caricamento di {len(relations)} relazioni...")
        
        # Versione con controllo esistenza nodi
        query = """
        UNWIND $relations AS rel_data
        MATCH (s:Entity {name: rel_data.soggetto_cluster})
        MATCH (o:Entity {name: rel_data.oggetto_cluster})
        WITH s, o, rel_data
        CALL apoc.merge.relationship(s, rel_data.predicato_cluster, 
            {}, 
            {   
                contexts: rel_data.contesti_aggregati,
                occurrence_count: rel_data.conteggio_occorrenze_totale,
                original_predicates: rel_data.predicati_originali_cluster
            }, o) YIELD rel
        RETURN count(rel) AS created_rels
        """
        
        try:
            result = self.run_query(query, parameters={"relations": relations})
            print(f"Caricamento relazioni completato con APOC.")
        except Exception as e:
            print(f"Errore nel caricamento relazioni: {e}")
            print("Verifica che tutti i nodi referenziati esistano e che APOC sia installato.")

def load_json_file(filepath: str) -> List[Dict]:
    """Funzione helper per caricare un file JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Errore: file non trovato a {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Errore: JSON non valido in {filepath}")
        return []

if __name__ == "__main__":
    entities_file = "kg_entities_clustered_final_empulia.json"
    relations_file = "kg_relations_clustered_final_empulia.json"

    print("Caricamento dati dai file JSON...")
    entities_data = load_json_file(entities_file)
    relations_data = load_json_file(relations_file)

    if not entities_data or not relations_data:
        print("Errore nel caricamento dei dati. Interruzione dello script.")
    else:
        # Crea un'istanza dell'uploader e si connette al database
        uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Solo se il driver si è connesso correttamente
        if uploader.driver:
            # 1. Pulisci il DB (fallo solo se vuoi ricominciare da capo)
            uploader.clear_database()
            
            # 2. Imposta i vincoli per ottimizzare le performance
            uploader.setup_constraints()
            
            # 3. Carica le entità (nodi)
            uploader.upload_entities(entities_data)
            
            # 4. Carica le relazioni (archi)
            uploader.upload_relations(relations_data)
            
            # 5. Chiudi la connessione
            uploader.close()

            print("\n-------------------------------------------")
            print("Knowledge Graph creato/aggiornato su Neo4j!")
            print("-------------------------------------------")