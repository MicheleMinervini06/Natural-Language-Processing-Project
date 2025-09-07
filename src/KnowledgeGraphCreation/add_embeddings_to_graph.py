import os
import time
from neo4j import GraphDatabase, basic_auth
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm # Per una bella barra di progresso

# --- Configurazione ---
load_dotenv()

# Configurazione Neo4j (assicurati che corrisponda al tuo ambiente)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "test")

# Configurazione Modello di Embedding
EMBEDDING_MODEL_NAME = "models/embedding-001"
# Ogni quante entità eseguire l'aggiornamento su Neo4j (per non sovraccaricare la memoria)
BATCH_SIZE = 100 

def get_nodes_without_embedding(driver):
    """Recupera tutti i nodi che non hanno ancora una proprietà 'embedding'."""
    query = """
    MATCH (n)
    WHERE n.embedding IS NULL AND n.name IS NOT NULL
    RETURN elementId(n) AS element_id, n.name AS name, n.type AS type, 
           n.description AS description, n.original_names AS original_names
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        return [dict(record) for record in result]

def generate_embedding_text(node_data):
    """Crea una stringa di testo ricca per rappresentare il significato di un nodo."""
    name = node_data.get('name', '')
    node_type = node_data.get('type', '')
    description = node_data.get('description', '')
    
    # Unisce le varianti in una stringa, se esistono e non sono None
    original_names = node_data.get('original_names', [])
    variants_text = ""
    if original_names:
        variants_text = f"Varianti: {', '.join(original_names)}"
    
    # Combina tutto in una singola stringa di testo
    # Questo testo verrà trasformato in un vettore
    return f"Nome: {name}. Tipo: {node_type}. Descrizione: {description}. {variants_text}".strip()

def update_nodes_with_embeddings(driver, batch_data):
    """Aggiorna un batch di nodi in Neo4j con i loro embeddings."""
    query = """
    UNWIND $batch as item
    MATCH (n) WHERE elementId(n) = item.element_id
    SET n.embedding = item.embedding
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(query, batch=batch_data)

def main():
    """Funzione principale per arricchire il grafo con gli embeddings."""
    print("--- Avvio Script di Arricchimento Embeddings ---")

    # Verifica la API Key di Gemini
    if not os.getenv("GEMINI_API_KEY"):
        print("ERRORE: La variabile d'ambiente GEMINI_API_KEY non è impostata. Uscita.")
        return

    # Inizializza il modello di embedding
    try:
        print(f"Inizializzazione del modello di embedding: {EMBEDDING_MODEL_NAME}")
        embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        print(f"Errore durante l'inizializzazione del modello di embedding: {e}")
        return

    # Connessione a Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)
        driver.verify_connectivity()
        print("Connessione a Neo4j stabilita.")
    except Exception as e:
        print(f"Errore di connessione a Neo4j: {e}")
        return

    # 1. Recupera i nodi da processare
    nodes_to_process = get_nodes_without_embedding(driver)
    if not nodes_to_process:
        print("Nessun nodo da processare. Tutti i nodi hanno già un embedding.")
        driver.close()
        return
    
    print(f"Trovati {len(nodes_to_process)} nodi senza embedding. Inizio elaborazione in batch...")

    # 2. Processa i nodi in batch
    batch_to_update = []
    
    with tqdm(total=len(nodes_to_process), desc="Generazione Embeddings") as pbar:
        for node in nodes_to_process:
            # Crea il testo da vettorizzare
            text_to_embed = generate_embedding_text(node)
            
            # Genera l'embedding (gestisce errori individuali)
            try:
                embedding_vector = embedder.embed_query(text_to_embed)
                batch_to_update.append({
                    "element_id": node['element_id'],
                    "embedding": embedding_vector
                })
            except Exception as e:
                print(f"\nATTENZIONE: Errore durante la generazione dell'embedding per il nodo {node['name']}: {e}")
                pbar.update(1)
                continue

            # Se il batch è pieno, aggiorna il DB e ricomincia
            if len(batch_to_update) >= BATCH_SIZE:
                update_nodes_with_embeddings(driver, batch_to_update)
                tqdm.write(f"Aggiornati {len(batch_to_update)} nodi nel database...")
                batch_to_update = []
            
            pbar.update(1)
            time.sleep(0.05) # Piccola pausa per non sovraccaricare l'API

    # 3. Assicurati di aggiornare l'ultimo batch rimasto
    if batch_to_update:
        update_nodes_with_embeddings(driver, batch_to_update)
        print(f"Aggiornati gli ultimi {len(batch_to_update)} nodi nel database.")

    driver.close()
    print("--- Processo di Arricchimento Completato ---")
    print("\nOra puoi creare l'indice vettoriale in Neo4j con la query fornita.")

if __name__ == "__main__":
    main()