import logging
from typing import List, Dict
from sentence_transformers.cross_encoder import CrossEncoder

# Configurazione del modello di reranking
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

class ContextReranker:
    """
    Una classe dedicata per riordinare i chunk di testo recuperati
    utilizzando un modello Cross-Encoder specializzato.
    """
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.logger = logging.getLogger(__name__)
        self.model = None
        try:
            # Il modello verrà scaricato da Hugging Face la prima volta
            self.model = CrossEncoder(model_name)
            self.logger.info(f"Modello Reranker '{model_name}' caricato con successo.")
        except Exception as e:
            self.logger.error(f"Errore critico durante l'inizializzazione del Reranker: {e}")
            self.logger.error("Il reranking sarà disabilitato.")
            # Non sollevare un'eccezione, permetti al retriever di funzionare senza reranking

    def rerank(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """
        Riordina una lista di chunk in base alla loro pertinenza con la domanda.

        Args:
            question (str): La domanda originale dell'utente.
            chunks (List[Dict]): La lista di chunk di testo candidati da riordinare.

        Returns:
            List[Dict]: La lista di chunk riordinata dal più al meno pertinente.
        """
        if not self.model or not chunks:
            self.logger.warning("Reranker non disponibile o nessun chunk fornito. Restituisco i chunk in ordine originale.")
            return chunks

        self.logger.info(f"Avvio reranking di {len(chunks)} chunk candidati...")
        
        # Crea le coppie [domanda, testo del chunk] per il modello
        # Assicurati che il testo del chunk non sia None
        pairs = [[question, chunk.get('text', '') or ''] for chunk in chunks]
        
        # Calcola gli score di pertinenza. Il modello gestisce internamente i batch.
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            self.logger.error(f"Errore durante la predizione del reranker: {e}")
            return chunks # Restituisce i chunk originali in caso di errore

        # Combina i chunk con i loro score
        scored_chunks = list(zip(scores, chunks))
        
        # Ordina i chunk in base allo score, dal più alto al più basso
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Estrai solo i chunk riordinati
        reranked_chunks = [chunk for _, chunk in scored_chunks]
        
        self.logger.info("Reranking completato con successo.")
        return reranked_chunks

# Esempio di utilizzo (se eseguito direttamente)
if __name__ == '__main__':
    logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Simula una domanda e dei chunk
    test_question = "Qual è il primo passo dopo l'accesso con nuove credenziali?"
    test_chunks = [
        {'chunk_id': 'c1', 'text': 'Per specificare il primo lotto, si devono inserire le informazioni richieste.'},
        {'chunk_id': 'c2', 'text': 'Dopo aver ricevuto le credenziali, il primo accesso richiede un cambio password obbligatorio dalla Lista Attività.'},
        {'chunk_id': 'c3', 'text': 'Il sistema di e-procurement EmPULIA gestisce gare e appalti in formato digitale.'},
        {'chunk_id': 'c4', 'text': 'La password deve rispettare i criteri di sicurezza richiesti.'}
    ]

    reranker_instance = ContextReranker()
    if reranker_instance.model:
        reranked_list = reranker_instance.rerank(test_question, test_chunks)
        
        print("\n--- Risultato del Reranking ---")
        print(f"Domanda: {test_question}\n")
        print("Ordine dei chunk dopo il reranking (dal più al meno pertinente):")
        for i, chunk in enumerate(reranked_list):
            print(f"{i+1}. ID: {chunk['chunk_id']} - Testo: '{chunk['text'][:50]}...'")
        
        # L'output atteso dovrebbe essere c2, c4, c3, c1 (o simile)