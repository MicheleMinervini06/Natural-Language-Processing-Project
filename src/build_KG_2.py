import json
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

entity_types = [
    "PiattaformaModulo",            # Es. "Registrazione Utente PA", "Negozio Elettronico"
    "FunzionalitàPiattaforma",      # Sotto-funzionalità o capacità specifiche
    "RuoloUtente",                  # Es. "Operatore PA", "RUP", "Fornitore"
    "AzioneUtente",                 # Es. "Cliccare su Salva", "Compilare Modulo X"
    "OperazioneSistema",            # Es. "Generare PDF", "Inviare Notifica"
    "InterfacciaUtenteElemento",    # Es. "Pulsante Continua", "Campo Nome Utente", "Tab Dati"
    "ComandoUI",                    # Nomi specifici dei comandi/pulsanti
    "DocumentoSistema",             # Es. "PDF Riepilogativo", "DGUE", "Ordinativo di Fornitura"
    "TipoDocumento",                # Categoria del documento
    "Prerequisito",                 # Es. "Possesso SPID", "Profilazione Utente Completata"
    "Condizione",                   # Condizioni logiche o di stato
    "ParametroConfigurazione",      # Es. "Metodo Calcolo Anomalia", "Tipo Gara Sottosoglia"
    "Criterio",                     # Criteri specifici di valutazione o configurazione
    "StatoDocumento",               # Es. "In lavorazione", "Pubblicato"
    "StatoProcedura",               # Es. "Aggiudicazione Proposta"
    "EnteEsterno",                  # Es. "ANAC", "PCP ANAC"
    "Organismo",                    # Organismi o ruoli esterni
    "TermineTemporale",             # Es. "Termine Richiesta Quesiti", "31/12/2023"
    "Scadenza",                     # Date di scadenza specifiche
    "MessaggioSistema",             # Es. "Messaggio di alert", "E-mail di conferma"
    "Notifica",                     # Tipi di notifiche
    "SezioneGuida"                  # Per tracciare la provenienza, es. "REGISTRAZIONE UTENTE PA"
]

predicates = [
    "haPassoSuccessivo",    # (AzioneUtente -> AzioneUtente) o (Funzionalità -> Funzionalità)
    "precede",              # Inverso di haPassoSuccessivo
    "richiedeInput",        # (FunzionalitàPiattaforma -> InterfacciaUtenteElemento)
    "haCampo",              # (InterfacciaUtenteElemento -> NomeCampoSpecifico - trattare NomeCampo come entità o attributo?)
    "èEseguitaDa",          # (AzioneUtente -> RuoloUtente)
    "puòEseguire",          # (RuoloUtente -> AzioneUtente)
    "haSottoFunzionalità",  # (PiattaformaModulo -> FunzionalitàPiattaforma)
    "includeOperazione",    # (FunzionalitàPiattaforma -> AzioneUtente)
    "generaDocumento",      # (AzioneUtente o OperazioneSistema -> DocumentoSistema)
    "richiedeDocumento",    # (FunzionalitàPiattaforma o AzioneUtente -> DocumentoSistema)
    "haPrerequisito",       # (FunzionalitàPiattaforma o AzioneUtente -> Prerequisito)
    "èVincolatoDa",         # (AzioneUtente -> Condizione)
    "haStato",              # (DocumentoSistema o PiattaformaModulo -> StatoDocumento/StatoProcedura)
    "puòAvereStato",        # Come sopra
    "interagisceCon",       # (PiattaformaModulo -> EnteEsterno)
    "inviaDatiA",           # Come sopra
    "haTermine",            # (FunzionalitàPiattaforma o PiattaformaModulo -> TermineTemporale)
    "scadeIl",              # (DocumentoSistema o FunzionalitàPiattaforma -> Scadenza)
    "mostraMessaggio",      # (AzioneUtente o OperazioneSistema -> MessaggioSistema)
    "produceNotifica",      # (OperazioneSistema -> Notifica)
    "èDescrittoIn",         # (QualsiasiEntità -> SezioneGuida) -> CRUCIALE PER TRACCIABILITÀ
    "provieneDa",           # Come sopra
    "utilizzaFormula",      # (FunzionalitàPiattaforma -> ParametroConfigurazione/Criterio)
    "haCriterioDiValutazione", # Come sopra
    "siApplicaA",           # (ParametroConfigurazione o Criterio -> FunzionalitàPiattaforma o PiattaformaModulo)
    "riguarda",             # Generica, (Entità -> Entità)
    "èParteDi",             # (FunzionalitàPiattaforma -> PiattaformaModulo) o (AzioneUtente -> FunzionalitàPiattaforma)
    "contieneElemento",     # (InterfacciaUtenteElemento -> InterfacciaUtenteElemento) per UI nidificate
    "rimandaA"              # (SezioneGuida -> SezioneGuida) o (DocumentoSistema -> DocumentoSistema)
]

DOCUMENT_PATH = "data/processed/processed_chunks_toc_enhanced.json"

def load_chunks_from_json(filepath: str) -> List[Dict[str, Any]]:
    """Carica i chunk di testo dal file JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Errore: File non trovato a {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Errore: Formato JSON non valido in {filepath}")
        return []

def triplextract(model, tokenizer, text, entity_types, predicates, device):

    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """

    message = input_format.format(
                entity_types = json.dumps({"entity_types": entity_types}),
                predicates = json.dumps({"predicates": predicates}),
                text = text)

    messages = [{'role': 'user', 'content': message}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors="pt").to(device)
    output = tokenizer.decode(model.generate(input_ids=input_ids, max_length=2048)[0], skip_special_tokens=True)
    return output

if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("sciphi/triplex", trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("sciphi/triplex", trust_remote_code=True)
    print("Model loaded successfully!")

    document_chunks = load_chunks_from_json(DOCUMENT_PATH)

    if document_chunks:
        text = document_chunks[0]['text']

    text = """
    San Francisco,[24] officially the City and County of San Francisco, is a commercial, financial, and cultural center in Northern California. 

    With a population of 808,437 residents as of 2022, San Francisco is the fourth most populous city in the U.S. state of California behind Los Angeles, San Diego, and San Jose.
    """

    prediction = triplextract(model, tokenizer, text, entity_types, predicates, device)
    print(prediction)
 