import json
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any
from utils.entity_normalizer import EntityNormalizer

# --- Configurazione Neo4j ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password"
#ATTENNZIONE
NEO4J_DATABASE = "testaggregated"

class Neo4jUploader:
    """
    Classe helper per interagire con un database Neo4j.
    """
    def __init__(self, uri, user, password, database):
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password), database=database)
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
        
        try:
            with self.driver.session() as session:
                if write_operation:
                    result = session.execute_write(self._execute_query, query, parameters)
                else:
                    result = session.execute_read(self._execute_query, query, parameters)
                return result
        except Exception as e:
            print(f"Errore nell'esecuzione della query: {e}")
            return []

    @staticmethod
    def _execute_query(tx, query, parameters=None):
        """Funzione helper eseguita all'interno di una transazione."""
        try:
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]
        except Exception as e:
            print(f"Errore nell'esecuzione della query: {e}")
            return []
        
    def clear_database(self):
        """Pulisce l'intero database. Usare con cautela!"""
        print("ATTENZIONE: Pulizia dell'intero database in corso...")
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)
        print("Database pulito.")

    def setup_constraints(self):
        """Imposta vincoli di unicità sui nodi per performance e integrità dei dati."""
        print("Impostazione dei vincoli di unicità...")
        try:
            query = "CREATE CONSTRAINT unique_entity_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE"
            self.run_query(query)
            print("Vincoli impostati.")
        except Exception as e:
            print(f"Errore nell'impostazione dei vincoli: {e}")

    def normalize_entity_name(self, name: str) -> str:
        """Chiama la funzione di normalizazzione condivisa."""
        return EntityNormalizer.normalize_entity_name(name)

    def upload_entities_raw(self, entities: List[Dict]):
        """Crea i nodi nel grafo usando i dati grezzi con normalizzazione."""
        if not entities:
            print("Nessuna entità da caricare.")
            return
            
        print(f"Inizio caricamento di {len(entities)} nodi grezzi...")
        
        # Filtra e normalizza le entità
        valid_entities = []
        invalid_count = 0
        normalization_log = {}
        
        for entity in entities:
            original_name = entity.get("nome_entita")
            if original_name and original_name.strip():
                # Normalizza il nome
                normalized_name = self.normalize_entity_name(original_name)
                if normalized_name:
                    # Crea una nuova entità con nome normalizzato
                    normalized_entity = entity.copy()
                    normalized_entity["nome_entita_normalized"] = normalized_name
                    normalized_entity["nome_entita_original"] = original_name
                    valid_entities.append(normalized_entity)
                    
                    # Log delle normalizzazioni per debug
                    if original_name.lower() != normalized_name:
                        if normalized_name not in normalization_log:
                            normalization_log[normalized_name] = []
                        normalization_log[normalized_name].append(original_name)
                else:
                    invalid_count += 1
                    print(f"WARN: Entità scartata - nome non normalizzabile: {original_name}")
            else:
                invalid_count += 1
                print(f"WARN: Entità scartata - nome null/vuoto: {entity}")
        
        # Mostra le normalizzazioni effettuate (solo se ci sono molte)
        if normalization_log and len(normalization_log) <= 20:
            print("\nNormalizzazioni effettuate:")
            for normalized, originals in normalization_log.items():
                unique_originals = list(set(originals))
                if len(unique_originals) > 1:
                    print(f"  '{normalized}' <- {unique_originals}")
        
        if invalid_count > 0:
            print(f"ATTENZIONE: {invalid_count} entità scartate")
        
        if not valid_entities:
            print("Nessuna entità valida da caricare.")
            return
        
        print(f"Caricamento di {len(valid_entities)} entità valide...")
        
        # Carica in batch per migliorare le performance
        batch_size = 1000
        total_created = 0
        
        for i in range(0, len(valid_entities), batch_size):
            batch = valid_entities[i:i+batch_size]
            print(f"Caricamento batch {i//batch_size + 1}/{(len(valid_entities) + batch_size - 1)//batch_size}")
            
            query_raw = """
            UNWIND $entities AS entity
            WITH entity 
            WHERE entity.nome_entita_normalized IS NOT NULL 
              AND entity.nome_entita_normalized <> ''
            MERGE (n:Entity {name: entity.nome_entita_normalized})
            ON CREATE SET 
                n.type = COALESCE(entity.tipo_entita, 'Unknown'),
                n.description = COALESCE(entity.descrizione_entita, ''),
                n.source_chunk_id = COALESCE(entity.source_chunk_id, ''),
                n.source_page_number = COALESCE(entity.source_page_number, 0),
                n.source_section_title = COALESCE(entity.source_section_title, ''),
                n.original_names = [entity.nome_entita_original],
                n.occurrence_count = 1
            ON MATCH SET
                n.occurrence_count = n.occurrence_count + 1,
                n.original_names = CASE 
                    WHEN entity.nome_entita_original IN n.original_names 
                    THEN n.original_names 
                    ELSE n.original_names + [entity.nome_entita_original]
                END
            RETURN count(n) AS created_nodes
            """
            
            try:
                result = self.run_query(query_raw, parameters={"entities": batch})
                if result and result[0]:
                    count = result[0].get('created_nodes', 0)
                    total_created += count
                    print(f"  Batch completato: {count} nodi")
            except Exception as e:
                print(f"Errore nel caricamento batch: {e}")
        
        print(f"Caricamento nodi grezzi completato. Totale: {total_created}")

    def upload_relations_raw(self, relations: List[Dict]):
        """Crea le relazioni tra i nodi esistenti usando i dati grezzi con tipi dinamici."""
        if not relations:
            print("Nessuna relazione da caricare.")
            return

        print(f"Inizio caricamento di {len(relations)} relazioni grezze...")
        
        # Filtra e raggruppa per tipo di predicato
        relations_by_type = {}
        invalid_count = 0
        
        for relation in relations:
            original_soggetto = relation.get("soggetto")
            original_oggetto = relation.get("oggetto")
            predicato = relation.get("predicato")
            
            if (original_soggetto and original_soggetto.strip() and 
                predicato and predicato.strip() and 
                original_oggetto and original_oggetto.strip()):
                
                # Normalizza soggetto e oggetto
                normalized_soggetto = self.normalize_entity_name(original_soggetto)
                normalized_oggetto = self.normalize_entity_name(original_oggetto)
                
                if normalized_soggetto and normalized_oggetto:
                    # Normalizza il predicato per Neo4j (rimuovi caratteri speciali)
                    predicato_safe = (predicato.replace(" ", "_")
                                    .replace("'", "")
                                    .replace("à", "a")
                                    .replace("è", "e")
                                    .replace("ì", "i")
                                    .replace("ò", "o")
                                    .replace("ù", "u")
                                    .replace("É", "E"))
                    
                    if predicato_safe not in relations_by_type:
                        relations_by_type[predicato_safe] = []
                    
                    normalized_relation = relation.copy()
                    normalized_relation["soggetto_normalized"] = normalized_soggetto
                    normalized_relation["oggetto_normalized"] = normalized_oggetto
                    normalized_relation["predicato_safe"] = predicato_safe
                    normalized_relation["soggetto_original"] = original_soggetto
                    normalized_relation["oggetto_original"] = original_oggetto
                    relations_by_type[predicato_safe].append(normalized_relation)
                else:
                    invalid_count += 1
                    if invalid_count <= 10:  # Mostra solo i primi 10 errori
                        print(f"WARN: Relazione scartata - nomi non normalizzabili: {original_soggetto} -> {original_oggetto}")
            else:
                invalid_count += 1
                if invalid_count <= 10:
                    print(f"WARN: Relazione scartata - valori null/vuoti: {relation}")
        
        if invalid_count > 0:
            print(f"ATTENZIONE: {invalid_count} relazioni scartate")
        
        if not relations_by_type:
            print("Nessuna relazione valida da caricare.")
            return
        
        print(f"Tipi di relazioni da caricare: {list(relations_by_type.keys())}")
        
        # Carica per tipo di relazione
        total_created = 0
        for predicato_type, relations_list in relations_by_type.items():
            print(f"Caricamento {len(relations_list)} relazioni di tipo '{predicato_type}'...")
            
            # Carica in batch per migliorare le performance
            batch_size = 500
            type_total = 0
            
            for i in range(0, len(relations_list), batch_size):
                batch = relations_list[i:i+batch_size]
                
                # Query dinamica con tipo di relazione specifico
                query = f"""
                UNWIND $relations AS rel_data
                MATCH (s:Entity {{name: rel_data.soggetto_normalized}})
                MATCH (o:Entity {{name: rel_data.oggetto_normalized}})
                CREATE (s)-[r:`{predicato_type}`]->(o)
                SET r.context = COALESCE(rel_data.contesto_relazione, ''),
                    r.source_chunk_id = COALESCE(rel_data.source_chunk_id, ''),
                    r.source_page_number = COALESCE(rel_data.source_page_number, 0),
                    r.source_section_title = COALESCE(rel_data.source_section_title, ''),
                    r.original_subject = rel_data.soggetto_original,
                    r.original_object = rel_data.oggetto_original,
                    r.original_predicate = rel_data.predicato
                RETURN count(r) AS created_rels
                """
                
                try:
                    result = self.run_query(query, parameters={"relations": batch})
                    if result and result[0]:
                        count = result[0].get('created_rels', 0)
                        type_total += count
                        if len(relations_list) > batch_size:
                            print(f"  Batch {i//batch_size + 1}: {count} relazioni")
                except Exception as e:
                    print(f"  Errore batch per tipo '{predicato_type}': {e}")
            
            total_created += type_total
            print(f"  Completato '{predicato_type}': {type_total} relazioni")
        
        print(f"Caricamento relazioni completato: {total_created} relazioni create")

 
    def upload_entities(self, entities: List[Dict]):
        """Crea i nodi nel grafo usando UNWIND per efficienza (dati clusterizzati)."""
        if not entities:
            print("Nessuna entità da caricare.")
            return
            
        print(f"Inizio caricamento di {len(entities)} nodi clusterizzati...")
        
        valid_entities = [e for e in entities if e.get("nome_entita_cluster", "").strip()]
        invalid_count = len(entities) - len(valid_entities)
        if invalid_count > 0:
            print(f"ATTENZIONE: {invalid_count} entità scartate per nome cluster nullo/vuoto.")
        
        if not valid_entities:
            print("Nessuna entità valida da caricare.")
            return
        
        print(f"Caricamento di {len(valid_entities)} entità valide...")
        
        batch_size = 1000
        for i in range(0, len(valid_entities), batch_size):
            batch = valid_entities[i:i+batch_size]
            print(f"Caricamento batch {i//batch_size + 1}/{(len(valid_entities) + batch_size - 1)//batch_size}")
            
            # --- INIZIO MODIFICA QUI ---
            query_corrected = """
            UNWIND $entities AS entity
            MERGE (n:Entity {name: entity.nome_entita_cluster})
            ON CREATE SET 
                n.type = COALESCE(entity.tipo_entita_cluster, 'Unknown'),
                n.descriptions = COALESCE(entity.descrizioni_aggregate, []),
                n.occurrence_count = COALESCE(entity.conteggio_occorrenze_totale, 0),
                n.original_members = COALESCE(entity.membri_cluster, []),
                
                /* CORREZIONE: Usa il nome corretto della chiave dal JSON clusterizzato */
                n.source_chunk_ids = COALESCE(entity.fonti_aggregate_chunk_id, [])

            ON MATCH SET
                /* Aggiungiamo l'aggiornamento anche su ON MATCH per robustezza */
                n.type = COALESCE(entity.tipo_entita_cluster, 'Unknown'),
                n.descriptions = COALESCE(entity.descrizioni_aggregate, []),
                n.occurrence_count = COALESCE(entity.conteggio_occorrenze_totale, 0),
                n.original_members = COALESCE(entity.membri_cluster, []),
                n.source_chunk_ids = COALESCE(entity.fonti_aggregate_chunk_id, [])

            RETURN count(n) AS node_count
            """
            # --- FINE MODIFICA QUI ---

            try:
                # Uso la variabile con il nome corretto
                result = self.run_query(query_corrected, parameters={"entities": batch})
                if result and result[0]:
                    count = result[0].get('node_count', 0)
                    print(f"  Batch completato: {count} nodi processati.")
            except Exception as e:
                print(f"Errore nel caricamento batch: {e}")
        
        print(f"Caricamento nodi clusterizzati completato.")

    def upload_relations(self, relations: List[Dict]):
        """Crea le relazioni tra i nodi esistenti (dati clusterizzati)."""
        if not relations:
            print("Nessuna relazione da caricare.")
            return

        print(f"Inizio caricamento di {len(relations)} relazioni clusterizzate...")
        
        # Filtra le relazioni con valori null
        valid_relations = []
        invalid_count = 0
        
        for relation in relations:
            soggetto = relation.get("soggetto_cluster")
            predicato = relation.get("predicato_cluster") 
            oggetto = relation.get("oggetto_cluster")
            
            if soggetto and soggetto.strip() and predicato and predicato.strip() and oggetto and oggetto.strip():
                valid_relations.append(relation)
            else:
                invalid_count += 1
                if invalid_count <= 10:
                    print(f"WARN: Relazione scartata - valori null/vuoti: {relation}")
        
        if invalid_count > 0:
            print(f"ATTENZIONE: {invalid_count} relazioni scartate per valori null/vuoti")
        
        if not valid_relations:
            print("Nessuna relazione valida da caricare.")
            return
        
        print(f"Caricamento di {len(valid_relations)} relazioni valide...")
        
        # Metodo semplificato senza APOC
        batch_size = 500
        total_created = 0
        
        for i in range(0, len(valid_relations), batch_size):
            batch = valid_relations[i:i+batch_size]
            print(f"Caricamento batch {i//batch_size + 1}/{(len(valid_relations) + batch_size - 1)//batch_size}")
            
            query_simple = """
            UNWIND $relations AS rel_data
            WITH rel_data 
            WHERE rel_data.soggetto_cluster IS NOT NULL 
              AND rel_data.soggetto_cluster <> ''
              AND rel_data.predicato_cluster IS NOT NULL 
              AND rel_data.predicato_cluster <> ''
              AND rel_data.oggetto_cluster IS NOT NULL 
              AND rel_data.oggetto_cluster <> ''
            MATCH (s:Entity {name: rel_data.soggetto_cluster})
            MATCH (o:Entity {name: rel_data.oggetto_cluster})
            WITH s, o, rel_data
            CREATE (s)-[r:RELATED]->(o)
            SET r.type = rel_data.predicato_cluster,
                r.contexts = COALESCE(rel_data.contesti_aggregati, []),
                r.occurrence_count = COALESCE(rel_data.conteggio_occorrenze_totale, 0),
                r.original_predicates = COALESCE(rel_data.predicati_originali_cluster, []),
                r.source_chunk_ids = COALESCE(rel_data.fonti_chunk_id, [])
            RETURN count(r) AS created_rels
            """
            
            try:
                result = self.run_query(query_simple, parameters={"relations": batch})
                if result and result[0]:
                    count = result[0].get('created_rels', 0)
                    total_created += count
                    print(f"  Batch completato: {count} relazioni")
            except Exception as e:
                print(f"Errore nel caricamento batch: {e}")
        
        print(f"Caricamento relazioni clusterizzate completato. Totale: {total_created}")

def load_json_file(filepath: str) -> List[Dict]:
    """Funzione helper per caricare un file JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Errore: file non trovato a {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Errore: JSON non valido in {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Errore nel caricamento del file {filepath}: {e}")
        return []

if __name__ == "__main__":
    # Scegli il tipo di dati da caricare
    print("Scegli il tipo di dati da caricare:")
    print("1. Dati grezzi (raw)")
    print("2. Dati clusterizzati (clustered)")
    
    choice = input("Inserisci 1 o 2: ").strip()
    
    if choice == "1":
        entities_file = "kg_entities_raw_empulia.json"
        relations_file = "kg_relations_raw_empulia.json"
        use_raw_data = True
        print("Caricamento dati grezzi...")
    elif choice == "2":
        entities_file = "kg_entities_clustered_final_empulia.json"
        relations_file = "kg_relations_clustered_final_empulia.json"
        use_raw_data = False
        print("Caricamento dati clusterizzati...")
    else:
        print("Scelta non valida. Uso dati grezzi come default.")
        entities_file = "kg_entities_raw_empulia.json"
        relations_file = "kg_relations_raw_empulia.json"
        use_raw_data = True

    print("Caricamento dati dai file JSON...")
    entities_data = load_json_file(entities_file)
    relations_data = load_json_file(relations_file)

    if not entities_data and not relations_data:
        print("Errore: nessun dato caricato. Interruzione dello script.")
        exit(1)
    
    if not entities_data:
        print("ATTENZIONE: nessuna entità caricata")
    
    if not relations_data:
        print("ATTENZIONE: nessuna relazione caricata")

    # Crea un'istanza dell'uploader e si connette al database
    uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    
    # Solo se il driver si è connesso correttamente
    if uploader.driver:
        # 1. Pulisci il DB (fallo solo se vuoi ricominciare da capo)
        uploader.clear_database()
        
        # 2. Imposta i vincoli per ottimizzare le performance
        uploader.setup_constraints()
        
        # 3. Carica le entità (nodi) - usa metodo appropriato
        if entities_data:
            if use_raw_data:
                uploader.upload_entities_raw(entities_data)
            else:
                uploader.upload_entities(entities_data)
        
        # 4. Carica le relazioni (archi) - usa metodo appropriato
        if relations_data:
            if use_raw_data:
                uploader.upload_relations_raw(relations_data)
            else:
                uploader.upload_relations(relations_data)
        
        # 5. Chiudi la connessione
        uploader.close()

        print("\n-------------------------------------------")
        print("Knowledge Graph creato/aggiornato su Neo4j!")
        print("-------------------------------------------")
    else:
        print("Impossibile procedere senza connessione al database.")