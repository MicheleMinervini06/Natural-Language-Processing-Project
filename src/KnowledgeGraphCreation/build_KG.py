import google.generativeai as genai
import json
import os
from collections import Counter
import time
from typing import List, Dict, Any, Tuple

# --- Configurazione ---
# Assicurati che la tua API key sia impostata come variabile d'ambiente
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Oppure, se non vuoi usare variabili d'ambiente (sconsigliato per codice condiviso):
# genai.configure(api_key="LA_TUA_API_KEY_QUI") # SOSTITUISCI CON LA TUA CHIAVE REALE

LLM_MODEL_EXTRACTION = "gemini-2.0-flash"
LLM_MODEL_CLUSTERING = "gemini-2.0-flash"

ENTITY_TYPES = [
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

RELATION_TYPES = [
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

def call_llm_api(prompt: str, model: str = LLM_MODEL_EXTRACTION, max_retries: int = 3, delay: int = 5) -> str:
    """
    Chiama l'API Gemini con gestione dei tentativi.
    Restituisce la risposta dell'LLM come stringa.
    """
    for attempt in range(max_retries):
        try:
            # Crea il modello Gemini (la configurazione dovrebbe essere già stata fatta)
            gemini_model = genai.GenerativeModel(model)
            
            # Costruisce il prompt completo con il system message
            full_prompt = """Sei un assistente AI esperto nell'estrazione di informazioni strutturate da manuali utente per creare Knowledge Graph dettagliati sulla piattaforma EmPULIA. Presta attenzione ai dettagli procedurali e ai termini specifici della piattaforma.

""" + prompt

            # Genera la risposta
            response = gemini_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    candidate_count=1,
                    max_output_tokens=4096,  # Limite massimo di token per evitare output troppo lunghi
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # Controlla se la risposta è stata bloccata
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = response.candidates[0].finish_reason.name
                if finish_reason != "STOP":
                    print(f"Avviso: Risposta Gemini bloccata o incompleta. Motivo: {finish_reason}")
                    if finish_reason == "SAFETY":
                        print("  La risposta è stata bloccata per motivi di sicurezza.")
                    elif finish_reason == "MAX_TOKENS":
                        print("  La risposta è stata troncata per limite di token.")
                    return ""
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Errore API Gemini (tentativo {attempt + 1}/{max_retries}): {e}")
            if "quota" in str(e).lower() or "rate" in str(e).lower() or "429" in str(e):
                current_delay = delay * (2 ** attempt)
                print(f"Rate limit raggiunto. Attendo {current_delay} secondi...")
                time.sleep(current_delay)
            elif attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("Massimo numero di tentativi raggiunto per errore API.")
                return ""
    return ""


def build_extraction_prompt(chunk_text: str, section_title: str, chunk_id: str) -> str:
    """
    Costruisce il prompt per l'estrazione di entità e relazioni,
    utilizzando i nuovi tipi definiti.
    """
    current_section_entity_name = section_title if section_title and section_title.strip() else f"SezioneSconosciuta_{chunk_id.split('_')[-1]}"

    prompt = f"""
Analizza il seguente testo estratto dalla sezione "{section_title}" (identificata come entità "{current_section_entity_name}") della guida della piattaforma EmPULIA.
Il tuo obiettivo è estrarre entità e relazioni per costruire un Knowledge Graph che descriva le procedure e le funzionalità della piattaforma.

--- TEXT START ---
{chunk_text}
--- TEXT END ---

ISTRUZIONI DETTAGLIATE:

1.  **Identificazione Entità**:
    Estrai tutte le entità rilevanti che appartengono a uno dei seguenti tipi:
    `{', '.join(ENTITY_TYPES)}`
    Per ciascuna entità, fornisci:
    - `nome_entita`: Il nome specifico dell'entità. Se possibile, normalizza termini simili (es. plurale/singolare, piccole variazioni). Evita nomi troppo generici se non indispensabili.
    - `tipo_entita`: Uno dei tipi definiti sopra. Scegli il tipo più specifico e appropriato.
    - `descrizione_entita` (opzionale ma consigliato): Una breve descrizione contestuale dell'entità tratta dal testo.

    **Includi sempre un'entità di tipo "SezioneGuida" con "nome_entita": "{current_section_entity_name}" e "descrizione_entita": "La sezione della guida EmPULIA intitolata '{section_title}' (ID: {chunk_id}) da cui provengono queste informazioni."**

2.  **Identificazione Relazioni**:
    Estrai le relazioni significative tra le entità identificate (incluse le relazioni con l'entità SezioneGuida "{current_section_entity_name}").
    Le relazioni devono appartenere a uno dei seguenti tipi:
    `{', '.join(RELATION_TYPES)}`
    Per ogni relazione, fornisci:
    - `soggetto`: Il `nome_entita` dell'entità soggetto (deve corrispondere a un `nome_entita` estratto).
    - `predicato`: Uno dei tipi di relazione definiti sopra.
    - `oggetto`: Il `nome_entita` dell'entità oggetto (deve corrispondere a un `nome_entita` estratto).
    - `contesto_relazione` (opzionale ma consigliato): Una breve frase o porzione di testo che giustifica chiaramente la relazione.

CONSIGLI PER L'ESTRAZIONE:
- Concentrati sulle procedure, i passaggi, i ruoli degli utenti, gli elementi dell'interfaccia, i documenti e i requisiti specifici di EmPULIA.
- Una `AzioneUtente` è spesso eseguita da un `RuoloUtente` e può riguardare una `FunzionalitàPiattaforma` o un `InterfacciaUtenteElemento`.
- Le `FunzionalitàPiattaforma` possono avere `Prerequisito` o richiedere `DocumentoSistema`.
- Collega quante più entità possibile all'entità `SezioneGuida` "{current_section_entity_name}" usando la relazione `èDescrittoIn` (soggetto: entità trovata, oggetto: "{current_section_entity_name}").

FORMATO OUTPUT (JSON):
Restituisci SOLO un oggetto JSON valido, senza alcun testo aggiuntivo prima o dopo. Non utilizzare markdown code blocks (```json). 
Il JSON deve avere esattamente due chiavi principali: "entita" (una lista di dizionari entità) e "relazioni" (una lista di dizionari relazione). 

ESEMPIO ESATTO del formato richiesto:
{{
  "entita": [
    {{
      "nome_entita": "Selezione Ente",
      "tipo_entita": "FunzionalitàPiattaforma",
      "descrizione_entita": "Il primo passo della procedura di registrazione utente PA."
    }},
    {{
      "nome_entita": "{current_section_entity_name}",
      "tipo_entita": "SezioneGuida",
      "descrizione_entita": "La sezione della guida EmPULIA intitolata '{section_title}' (ID: {chunk_id}) da cui provengono queste informazioni."
    }}
  ],
  "relazioni": [
    {{
      "soggetto": "Selezione Ente",
      "predicato": "èParteDi",
      "oggetto": "Registrazione Utente PA",
      "contesto_relazione": "La procedura si compone dei seguenti STEP: Selezione Ente"
    }},
    {{
      "soggetto": "Selezione Ente",
      "predicato": "èDescrittoIn",
      "oggetto": "{current_section_entity_name}"
    }}
  ]
}}

IMPORTANTE: La tua risposta deve iniziare con {{ e finire con }}. Non aggiungere spiegazioni, commenti o altro testo.
Assicurati che tutti i nomi di entità nelle relazioni corrispondano esattamente ai "nome_entita" definiti nella sezione "entita".
Se una sezione è molto breve o non contiene informazioni estraibili per entità diverse da "{current_section_entity_name}", restituisci un JSON contenente solo l'entità SezioneGuida nella lista "entita" e una lista "relazioni" vuota.
"""
    return prompt

def parse_llm_extraction_output(llm_response_str: str) -> Tuple[List[Dict], List[Dict]]:
    """Interpreta l'output JSON dell'LLM e restituisce liste di entità e relazioni."""
    
    # Debug: mostra la risposta grezza per capire il problema
    print(f"\n--- DEBUG: Risposta LLM grezza (primi 500 caratteri) ---")
    print(f"{llm_response_str[:500]}...")
    print(f"--- Fine debug ---\n")
    
    # Prova a pulire la risposta se contiene markdown o altri formati
    cleaned_response = llm_response_str.strip()
    
    # Rimuovi eventuali markdown code blocks
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]  # Rimuovi ```json
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]   # Rimuovi ```
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]  # Rimuovi ``` finale
    
    cleaned_response = cleaned_response.strip()
    
    try:
        data = json.loads(cleaned_response)
        entities = data.get("entita", [])
        relations = data.get("relazioni", [])
        
        # Validazione più stringente
        valid_entities = []
        if isinstance(entities, list):
            for e in entities:
                if isinstance(e, dict) and "nome_entita" in e and "tipo_entita" in e:
                    if e["tipo_entita"] in ENTITY_TYPES:
                        valid_entities.append(e)
                    else:
                        print(f"Avviso: Tipo entità '{e['tipo_entita']}' non valido per '{e['nome_entita']}'. Entità scartata.")
                else:
                    print(f"Avviso: Formato entità non conforme o campi mancanti: {str(e)[:100]}. Entità scartata.")
        else:
            print(f"Avviso: 'entita' non è una lista nell'output LLM: {cleaned_response[:200]}")

        valid_relations = []
        if isinstance(relations, list):
            entity_names_extracted = {e["nome_entita"] for e in valid_entities} # Nomi delle entità valide estratte
            for r in relations:
                if isinstance(r, dict) and "soggetto" in r and "predicato" in r and "oggetto" in r:
                    if r["predicato"] in RELATION_TYPES:
                        # Controlla se soggetto e oggetto sono tra le entità estratte (opzionale ma buon controllo)
                        # if r["soggetto"] in entity_names_extracted and r["oggetto"] in entity_names_extracted:
                        valid_relations.append(r)
                        # else:
                        #     print(f"Avviso: Soggetto '{r['soggetto']}' o Oggetto '{r['oggetto']}' non trovato tra le entità valide per relazione '{r['predicato']}'. Relazione scartata.")
                    else:
                        print(f"Avviso: Tipo relazione '{r['predicato']}' non valido. Relazione scartata.")
                else:
                    print(f"Avviso: Formato relazione non conforme o campi mancanti: {str(r)[:100]}. Relazione scartata.")
        else:
            print(f"Avviso: 'relazioni' non è una lista nell'output LLM: {cleaned_response[:200]}")
             
        return valid_entities, valid_relations
        
    except json.JSONDecodeError as e:
        print(f"Errore Critico nel parsing dell'output JSON dall'LLM:")
        print(f"  Errore JSON: {e}")
        print(f"  Posizione errore: linea {e.lineno}, colonna {e.colno}")
        print(f"  Risposta pulita (primi 1000 caratteri): {cleaned_response[:1000]}")
        return [], []
    except Exception as e:
        print(f"Errore Critico imprevisto nel parsing dell'output LLM: {e}")
        print(f"  Risposta grezza: {llm_response_str[:500]}")
        return [], []

def extract_knowledge_from_chunks(chunks: List[Dict[str, Any]], output_dir: str = "llm_outputs") -> Tuple[List[Dict], List[Dict]]:
    """Itera sui chunk, chiama l'LLM per estrarre entità e relazioni."""
    all_entities: List[Dict] = []
    all_relations: List[Dict] = []
    processed_chunks_count = 0
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get('chunk_id', f"chunk_{i}")
        section_title = chunk.get('section_title', "Nessun Titolo Assegnato")
        chunk_text = chunk.get('text', "")

        # Salva il singolo chunk in un file di testo (utile per debug)
        chunk_filename = os.path.join(output_dir, f"{chunk_id}_input.txt")
        try:
            with open(chunk_filename, 'w', encoding='utf-8') as f_out:
                f_out.write(f"CHUNK_ID: {chunk_id}\nPAGE_NUMBER: {chunk.get('page_number')}\nSECTION_TITLE: {section_title}\n\n---\n{chunk_text}")
        except Exception as e:
            print(f"Errore durante il salvataggio del chunk input {chunk_id}: {e}")

        print(f"Processo il chunk {i+1}/{len(chunks)}: ID='{chunk_id}' - Sezione='{section_title}'")
        if not chunk_text.strip():
            print(f"Avviso: Chunk {chunk_id} saltato per mancanza di testo significativo.")
            continue

        prompt = build_extraction_prompt(chunk_text, section_title, chunk_id)
        llm_output_str = call_llm_api(prompt)

        llm_output_filename = os.path.join(output_dir, f"{chunk_id}_llm_output.json")
        try:
            with open(llm_output_filename, 'w', encoding='utf-8') as f_out:
                # Prova a formattare se è un JSON valido, altrimenti salva come stringa
                try:
                    parsed_json = json.loads(llm_output_str)
                    json.dump(parsed_json, f_out, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    f_out.write(llm_output_str if llm_output_str else "{}") # Salva la stringa grezza se non è JSON
        except Exception as e:
            print(f"Errore durante il salvataggio dell'output LLM per {chunk_id}: {e}")

        if llm_output_str:
            entities, relations = parse_llm_extraction_output(llm_output_str)
            # Aggiungi provenienza ai dati estratti
            for entity in entities:
                entity['source_chunk_id'] = chunk_id
                entity['source_page_number'] = chunk.get('page_number')
                entity['source_section_title'] = section_title
            for relation in relations:
                relation['source_chunk_id'] = chunk_id
                relation['source_page_number'] = chunk.get('page_number')
                relation['source_section_title'] = section_title

            all_entities.extend(entities)
            all_relations.extend(relations)
            processed_chunks_count += 1
            print(f"  Estratte {len(entities)} entità e {len(relations)} relazioni.")
        else:
            print(f"  Nessun output valido dall'LLM per il chunk {chunk_id}.")

        if i < len(chunks) - 1: # Non aspettare dopo l'ultimo chunk
            time.sleep(1.5) # Leggermente aumentato, da aggiustare in base ai rate limit effettivi

    print(f"\nElaborazione chunk completata. Processati {processed_chunks_count}/{len(chunks)} chunk con output valido.")
    print(f"Totale entità estratte (prima del clustering): {len(all_entities)}")
    print(f"Totale relazioni estratte (prima del clustering): {len(all_relations)}")
    return all_entities, all_relations

#def aggregate_knowledge(entities: List[Dict], relations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Aggrega entità e relazioni, normalizzando e unendo informazioni da occorrenze multiple.
    """
    print("\nInizio aggregazione e normalizzazione...")
    
    unique_entities_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entity in entities:
        name = entity.get("nome_entita", "").strip()
        etype = entity.get("tipo_entita", "")
        if name and etype:
            norm_name = name.lower()
            key = (norm_name, etype) # Usiamo il tipo originale per la chiave qui,
            # il clustering LLM potrà affinare anche i tipi se necessario
            if key not in unique_entities_dict:
                unique_entities_dict[key] = {
                    "nome_entita_aggregato": name, # Manteniamo una versione del nome (es. il primo incontrato con quel case)
                    "nome_entita_norm": norm_name,
                    "tipo_entita": etype,
                    "descrizioni": [d for d in [entity.get("descrizione_entita")] if d and d.strip()],
                    "fonti_chunk_id": [c_id for c_id in [entity.get("source_chunk_id")] if c_id],
                    "fonti_pagina": [p_num for p_num in [entity.get("source_page_number")] if p_num is not None],
                    "fonti_sezione": [s_title for s_title in [entity.get("source_section_title")] if s_title],
                    "conteggio_occorrenze": 1
                }
            else:
                current_entry = unique_entities_dict[key]
                if entity.get("descrizione_entita") and entity.get("descrizione_entita").strip():
                    current_entry["descrizioni"].append(entity.get("descrizione_entita").strip())
                if entity.get("source_chunk_id"):
                    current_entry["fonti_chunk_id"].append(entity.get("source_chunk_id"))
                if entity.get("source_page_number") is not None:
                    current_entry["fonti_pagina"].append(entity.get("source_page_number"))
                if entity.get("source_section_title"):
                    current_entry["fonti_sezione"].append(entity.get("source_section_title"))
                current_entry["conteggio_occorrenze"] += 1

    unique_relations_dict: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for relation in relations:
        s = relation.get("soggetto", "").strip()
        p = relation.get("predicato", "").strip()
        o = relation.get("oggetto", "").strip()
        if s and p and o:
            norm_s = s.lower()
            norm_p = p.lower()
            norm_o = o.lower()
            key = (norm_s, norm_p, norm_o) # Chiave basata su nomi normalizzati

            if key not in unique_relations_dict:
                unique_relations_dict[key] = {
                    "soggetto_aggregato": s,
                    "soggetto_norm": norm_s,
                    "predicato_aggregato": p,
                    "predicato_norm": norm_p,
                    "oggetto_aggregato": o,
                    "oggetto_norm": norm_o,
                    "contesti": [ctx for ctx in [relation.get("contesto_relazione")] if ctx and ctx.strip()],
                    "fonti_chunk_id": [c_id for c_id in [relation.get("source_chunk_id")] if c_id],
                    "fonti_pagina": [p_num for p_num in [relation.get("source_page_number")] if p_num is not None],
                    "fonti_sezione": [s_title for s_title in [relation.get("source_section_title")] if s_title],
                    "conteggio_occorrenze": 1
                }
            else:
                current_entry = unique_relations_dict[key]
                if relation.get("contesto_relazione") and relation.get("contesto_relazione").strip():
                    current_entry["contesti"].append(relation.get("contesto_relazione").strip())
                if relation.get("source_chunk_id"):
                    current_entry["fonti_chunk_id"].append(relation.get("source_chunk_id"))
                if relation.get("source_page_number") is not None:
                    current_entry["fonti_pagina"].append(relation.get("source_page_number"))
                if relation.get("source_section_title"):
                    current_entry["fonti_sezione"].append(relation.get("source_section_title"))
                current_entry["conteggio_occorrenze"] += 1

    aggregated_entities = []
    for data in unique_entities_dict.values():
        data["descrizioni"] = sorted(list(set(data["descrizioni"]))) # Rimuovi duplicati e ordina
        data["fonti_chunk_id"] = sorted(list(set(data["fonti_chunk_id"])))
        data["fonti_pagina"] = sorted(list(set(data["fonti_pagina"])))
        data["fonti_sezione"] = sorted(list(set(data["fonti_sezione"])))
        aggregated_entities.append(data)

    aggregated_relations = []
    for data in unique_relations_dict.values():
        data["contesti"] = sorted(list(set(data["contesti"])))
        data["fonti_chunk_id"] = sorted(list(set(data["fonti_chunk_id"])))
        data["fonti_pagina"] = sorted(list(set(data["fonti_pagina"])))
        data["fonti_sezione"] = sorted(list(set(data["fonti_sezione"])))
        aggregated_relations.append(data)

    print(f"Entità uniche dopo aggregazione: {len(aggregated_entities)}")
    print(f"Relazioni uniche dopo aggregazione: {len(aggregated_relations)}")
    return aggregated_entities, aggregated_relations

def aggregate_knowledge_improved(entities: List[Dict], relations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Versione migliorata di aggregate_knowledge.
    - Raggruppa le entità per nome normalizzato, gestendo tipi multipli.
    - Mantiene tutte le varianti originali di nomi e tipi.
    - Sceglie il tipo più frequente come tipo "canonico" per l'entità aggregata.
    """
    print("\nInizio aggregazione e normalizzazione (versione migliorata)...")

    # --- Aggregazione Entità Migliorata ---
    # La chiave ora è solo il nome normalizzato, per raggruppare entità con lo stesso nome ma tipi diversi
    unique_entities_dict: Dict[str, Dict[str, Any]] = {}
    
    for entity in entities:
        name = entity.get("nome_entita", "").strip()
        etype = entity.get("tipo_entita", "")
        if not name or not etype:
            continue

        norm_name = name.lower()
        key = norm_name

        if key not in unique_entities_dict:
            unique_entities_dict[key] = {
                "nomi_originali": [name],
                "tipi_rilevati": [etype],
                "descrizioni": [d for d in [entity.get("descrizione_entita")] if d and d.strip()],
                "fonti_chunk_id": [c_id for c_id in [entity.get("source_chunk_id")] if c_id],
                "fonti_pagina": [p_num for p_num in [entity.get("source_page_number")] if p_num is not None],
                "fonti_sezione": [s_title for s_title in [entity.get("source_section_title")] if s_title],
                "conteggio_occorrenze": 1
            }
        else:
            current_entry = unique_entities_dict[key]
            current_entry["nomi_originali"].append(name)
            current_entry["tipi_rilevati"].append(etype)
            
            if entity.get("descrizione_entita") and entity.get("descrizione_entita").strip():
                current_entry["descrizioni"].append(entity.get("descrizione_entita").strip())
            if entity.get("source_chunk_id"):
                current_entry["fonti_chunk_id"].append(entity.get("source_chunk_id"))
            if entity.get("source_page_number") is not None:
                current_entry["fonti_pagina"].append(entity.get("source_page_number"))
            if entity.get("source_section_title"):
                current_entry["fonti_sezione"].append(entity.get("source_section_title"))
            
            current_entry["conteggio_occorrenze"] += 1

    # Finalizzazione delle entità aggregate
    aggregated_entities = []
    for norm_name, data in unique_entities_dict.items():
        # Scegli il nome e il tipo più frequenti come "canonici" per questa fase
        most_common_name = Counter(data["nomi_originali"]).most_common(1)[0][0]
        most_common_type = Counter(data["tipi_rilevati"]).most_common(1)[0][0]

        final_entity = {
            "nome_entita_canonico_provvisorio": most_common_name, # Nome canonico provvisorio
            "nome_entita_norm": norm_name,
            "tipo_entita_canonico_provvisorio": most_common_type, # Tipo canonico provvisorio
            "tutti_nomi_originali": sorted(list(set(data["nomi_originali"]))),
            "tutti_tipi_rilevati": sorted(list(set(data["tipi_rilevati"]))),
            "descrizioni_aggregate": sorted(list(set(data["descrizioni"]))),
            "fonti_chunk_id": sorted(list(set(data["fonti_chunk_id"]))),
            "fonti_pagina": sorted(list(set(data["fonti_pagina"]))),
            "fonti_sezione": sorted(list(set(data["fonti_sezione"]))),
            "conteggio_occorrenze": data["conteggio_occorrenze"]
        }
        aggregated_entities.append(final_entity)

    # --- Aggregazione Relazioni (rimane simile, ma potremmo aggiungere conteggi) ---
    unique_relations_dict: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for relation in relations:
        s = relation.get("soggetto", "").strip()
        p = relation.get("predicato", "").strip()
        o = relation.get("oggetto", "").strip()
        if not s or not p or not o:
            continue
            
        norm_s = s.lower()
        norm_p = p.lower()
        norm_o = o.lower()
        key = (norm_s, norm_p, norm_o)

        if key not in unique_relations_dict:
             unique_relations_dict[key] = {
                "soggetto_norm": norm_s,
                "predicato_norm": norm_p,
                "oggetto_norm": norm_o,
                "contesti": [ctx for ctx in [relation.get("contesto_relazione")] if ctx and ctx.strip()],
                "fonti_chunk_id": [c_id for c_id in [relation.get("source_chunk_id")] if c_id],
                "fonti_pagina": [p_num for p_num in [relation.get("source_page_number")] if p_num is not None],
                "fonti_sezione": [s_title for s_title in [relation.get("source_section_title")] if s_title],
                "conteggio_occorrenze": 1
            }
        else:
            current_entry = unique_relations_dict[key]
            if relation.get("contesto_relazione") and relation.get("contesto_relazione").strip():
                current_entry["contesti"].append(relation.get("contesto_relazione").strip())
            if relation.get("source_chunk_id"):
                current_entry["fonti_chunk_id"].append(relation.get("source_chunk_id"))
            if relation.get("source_page_number") is not None:
                current_entry["fonti_pagina"].append(relation.get("source_page_number"))
            if relation.get("source_section_title"):
                current_entry["fonti_sezione"].append(relation.get("source_section_title"))
            current_entry["conteggio_occorrenze"] += 1

    aggregated_relations = []
    for data in unique_relations_dict.values():
        data["contesti"] = sorted(list(set(data["contesti"])))
        data["fonti_chunk_id"] = sorted(list(set(data["fonti_chunk_id"])))
        data["fonti_pagina"] = sorted(list(set(data["fonti_pagina"])))
        data["fonti_sezione"] = sorted(list(set(data["fonti_sezione"])))
        aggregated_relations.append(data)


    print(f"Entità uniche (raggruppate per nome) dopo aggregazione: {len(aggregated_entities)}")
    print(f"Relazioni uniche dopo aggregazione: {len(aggregated_relations)}")
    return aggregated_entities, aggregated_relations

def llm_cluster_knowledge(aggregated_entities: List[Dict], aggregated_relations: List[Dict], batch_size: int = 15) -> Tuple[List[Dict], List[Dict]]:
    """
    Utilizza Gemini per clusterizzare contemporaneamente entità e relazioni in un'unica chiamata.
    Riduce il numero di richieste API e mantiene la coerenza tra entità e relazioni.
    """
    print("\nInizio clustering combinato entità e relazioni con logica LLM avanzata...")
    
    if not aggregated_entities and not aggregated_relations:
        return [], []
    
    # Prepara i dati per il clustering
    entities_for_clustering = prepare_entities_for_clustering(aggregated_entities)
    relations_for_clustering = prepare_relations_for_clustering(aggregated_relations)
    
    # Processa in batch combinati
    all_entity_clusters = []
    all_relation_clusters = []
    
    total_batches = max(len(entities_for_clustering), len(relations_for_clustering))
    batch_size_entities = min(batch_size, len(entities_for_clustering)) if entities_for_clustering else 0
    batch_size_relations = min(batch_size * 2, len(relations_for_clustering)) if relations_for_clustering else 0  # Relazioni sono più semplici
    
    entities_processed = 0
    relations_processed = 0
    
    batch_num = 1
    while entities_processed < len(entities_for_clustering) or relations_processed < len(relations_for_clustering):
        # Prendi batch di entità e relazioni
        entity_batch = entities_for_clustering[entities_processed:entities_processed + batch_size_entities] if entities_processed < len(entities_for_clustering) else []
        relation_batch = relations_for_clustering[relations_processed:relations_processed + batch_size_relations] if relations_processed < len(relations_for_clustering) else []
        
        print(f"Clustering batch {batch_num}: {len(entity_batch)} entità, {len(relation_batch)} relazioni")
        
        # Processa il batch combinato
        entity_clusters, relation_clusters = process_combined_batch(entity_batch, relation_batch)
        
        all_entity_clusters.extend(entity_clusters)
        all_relation_clusters.extend(relation_clusters)
        
        entities_processed += len(entity_batch)
        relations_processed += len(relation_batch)
        batch_num += 1
        
        # Pausa tra batch per evitare rate limiting
        if entities_processed < len(entities_for_clustering) or relations_processed < len(relations_for_clustering):
            time.sleep(2)
    
    # Finalizza i cluster
    final_clustered_entities = finalize_entity_clusters(all_entity_clusters, aggregated_entities)
    final_clustered_relations = finalize_relation_clusters(all_relation_clusters, aggregated_relations, final_clustered_entities)
    
    print(f"Clustering combinato completato:")
    print(f"  Entità: {len(aggregated_entities)} → {len(final_clustered_entities)} ({len(aggregated_entities) - len(final_clustered_entities)} raggruppate)")
    print(f"  Relazioni: {len(aggregated_relations)} → {len(final_clustered_relations)} ({len(aggregated_relations) - len(final_clustered_relations)} raggruppate)")
    
    return final_clustered_entities, final_clustered_relations

def prepare_entities_for_clustering(aggregated_entities: List[Dict]) -> List[Dict]:
    """Prepara le entità per il clustering."""
    entities_for_clustering = []
    for i, entity in enumerate(aggregated_entities):
        if "nome_entita_canonico_provvisorio" in entity:
            entity_info = {
                "id": i,
                "nome_principale": entity["nome_entita_canonico_provvisorio"],
                "tipo_principale": entity["tipo_entita_canonico_provvisorio"],
                "tutti_nomi": entity["tutti_nomi_originali"],
                "tutti_tipi": entity["tutti_tipi_rilevati"],
                "descrizioni": entity["descrizioni_aggregate"][:2],  # Ridotto per combinazione
                "occorrenze": entity["conteggio_occorrenze"]
            }
        else:
            entity_info = {
                "id": i,
                "nome_principale": entity.get("nome_entita_aggregato", entity.get("nome_entita_norm", "")),
                "tipo_principale": entity.get("tipo_entita", ""),
                "tutti_nomi": [entity.get("nome_entita_aggregato", entity.get("nome_entita_norm", ""))],
                "tutti_tipi": [entity.get("tipo_entita", "")],
                "descrizioni": entity.get("descrizioni", [])[:2],
                "occorrenze": entity.get("conteggio_occorrenze", 1)
            }
        entities_for_clustering.append(entity_info)
    return entities_for_clustering

def prepare_relations_for_clustering(aggregated_relations: List[Dict]) -> List[Dict]:
    """Prepara le relazioni per il clustering."""
    relations_for_clustering = []
    for i, relation in enumerate(aggregated_relations):
        relation_info = {
            "id": i,
            "soggetto": relation["soggetto_norm"],
            "predicato": relation["predicato_norm"],
            "oggetto": relation["oggetto_norm"],
            "contesti": relation["contesti"][:1],  # Ridotto per combinazione
            "occorrenze": relation["conteggio_occorrenze"]
        }
        relations_for_clustering.append(relation_info)
    return relations_for_clustering

def process_combined_batch(entity_batch: List[Dict], relation_batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Processa un batch combinato di entità e relazioni."""
    
    prompt = build_combined_clustering_prompt(entity_batch, relation_batch)
    llm_output = call_llm_api(prompt, model=LLM_MODEL_CLUSTERING)
    
    if not llm_output:
        print("Nessun output dal LLM per clustering combinato, creando cluster singoli")
        entity_clusters = [{"membri_ids": [e["id"]], "nome_cluster": e["nome_principale"], 
                           "tipo_cluster": e["tipo_principale"], "motivazione": "Fallback: nessun clustering LLM"} 
                          for e in entity_batch]
        relation_clusters = [{"membri_ids": [r["id"]], "soggetto_cluster": r["soggetto"],
                             "predicato_cluster": r["predicato"], "oggetto_cluster": r["oggetto"],
                             "motivazione": "Fallback: nessun clustering LLM"} 
                            for r in relation_batch]
        return entity_clusters, relation_clusters
    
    try:
        entity_clusters, relation_clusters = parse_combined_clustering_output(llm_output, entity_batch, relation_batch)
        return entity_clusters, relation_clusters
    except Exception as e:
        print(f"Errore nel parsing dell'output clustering combinato: {e}")
        # Fallback ai cluster singoli
        entity_clusters = [{"membri_ids": [e["id"]], "nome_cluster": e["nome_principale"], 
                           "tipo_cluster": e["tipo_principale"], "motivazione": "Fallback: errore parsing"} 
                          for e in entity_batch]
        relation_clusters = [{"membri_ids": [r["id"]], "soggetto_cluster": r["soggetto"],
                             "predicato_cluster": r["predicato"], "oggetto_cluster": r["oggetto"],
                             "motivazione": "Fallback: errore parsing"} 
                            for r in relation_batch]
        return entity_clusters, relation_clusters

def build_combined_clustering_prompt(entity_batch: List[Dict], relation_batch: List[Dict]) -> str:
    """Costruisce il prompt per il clustering combinato."""
    
    # Sezione entità
    entities_section = ""
    if entity_batch:
        entities_section = "ENTITÀ DA ANALIZZARE:\n"
        for entity in entity_batch:
            entities_section += f"""
ID: {entity["id"]}
Nome: {entity["nome_principale"]}
Tipo: {entity["tipo_principale"]}
Nomi alternativi: {", ".join(entity["tutti_nomi"])}
Descrizioni: {" | ".join(entity["descrizioni"])}
Occorrenze: {entity["occorrenze"]}
---"""
    
    # Sezione relazioni
    relations_section = ""
    if relation_batch:
        relations_section = "\nRELAZIONI DA ANALIZZARE:\n"
        for relation in relation_batch:
            relations_section += f"""
ID: {relation["id"]}
Soggetto: {relation["soggetto"]}
Predicato: {relation["predicato"]}
Oggetto: {relation["oggetto"]}
Contesti: {" | ".join(relation["contesti"])}
Occorrenze: {relation["occorrenze"]}
---"""
    
    prompt = f"""
Analizza le seguenti entità e relazioni estratte dalla documentazione della piattaforma EmPULIA e raggruppa quelle semanticamente simili.

{entities_section}

{relations_section}

ISTRUZIONI PER IL CLUSTERING:

**Per le ENTITÀ:**
1. Raggruppa entità che rappresentano lo stesso concetto
2. Considera variazioni nei nomi, abbreviazioni, sinonimi
3. Mantieni separati concetti chiaramente distinti
4. Scegli nome e tipo più rappresentativi per ogni cluster

**Per le RELAZIONI:**
1. Raggruppa relazioni che esprimono lo stesso tipo di connessione
2. Considera predicati sinonimi o semanticamente equivalenti
3. Normalizza predicati secondo: {', '.join(RELATION_TYPES)}
4. Mantieni separate relazioni con significati distinti

FORMATO OUTPUT (JSON):
{{
  "entita_clusters": [
    {{
      "membri_ids": [0, 3],
      "nome_cluster": "Registrazione Utente PA",
      "tipo_cluster": "FunzionalitàPiattaforma",
      "motivazione": "Varianti dello stesso processo di registrazione"
    }}
  ],
  "relazioni_clusters": [
    {{
      "membri_ids": [0, 2],
      "soggetto_cluster": "Fornitore",
      "predicato_cluster": "èEseguitaDa",
      "oggetto_cluster": "Registrazione",
      "motivazione": "Predicati sinonimi per la stessa azione"
    }}
  ]
}}

IMPORTANTE:
- Ogni ID deve apparire in esattamente un cluster
- Include TUTTI gli ID forniti
- Usa solo predicati validi da RELATION_TYPES
- Risposta deve essere JSON puro senza markdown
"""
    
    return prompt

def parse_combined_clustering_output(llm_output: str, entity_batch: List[Dict], relation_batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Interpreta l'output del clustering combinato."""
    
    # Pulisci l'output
    cleaned_output = llm_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output[7:]
    if cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[3:]
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]
    cleaned_output = cleaned_output.strip()
    
    try:
        data = json.loads(cleaned_output)
        
        # Processa cluster entità
        entity_clusters = []
        for cluster in data.get("entita_clusters", []):
            if isinstance(cluster, dict) and "membri_ids" in cluster:
                if isinstance(cluster["membri_ids"], list) and cluster["membri_ids"]:
                    entity_clusters.append(cluster)
        
        # Processa cluster relazioni
        relation_clusters = []
        for cluster in data.get("relazioni_clusters", []):
            if isinstance(cluster, dict) and "membri_ids" in cluster:
                if isinstance(cluster["membri_ids"], list) and cluster["membri_ids"]:
                    # Valida predicato
                    predicato = cluster.get("predicato_cluster", "")
                    if predicato not in RELATION_TYPES:
                        print(f"Predicato non valido corretto: {predicato}")
                        # Prova a trovare un predicato simile o usa un default
                        cluster["predicato_cluster"] = find_closest_predicate(predicato)
                    relation_clusters.append(cluster)
        
        return entity_clusters, relation_clusters
        
    except json.JSONDecodeError as e:
        print(f"Errore JSON nel parsing clustering combinato: {e}")
        print(f"Output problematico: {cleaned_output[:500]}")
        raise

def find_closest_predicate(invalid_predicate: str) -> str:
    """Trova il predicato più simile da RELATION_TYPES."""
    invalid_lower = invalid_predicate.lower()
    
    # Mappature comuni
    mappings = {
        "esegue": "puòEseguire",
        "eseguita": "èEseguitaDa",
        "parte": "èParteDi",
        "contiene": "contieneElemento",
        "richiede": "richiedeInput",
        "genera": "generaDocumento",
        "interagisce": "interagisceCon",
        "descritto": "èDescrittoIn"
    }
    
    for key, value in mappings.items():
        if key in invalid_lower:
            return value
    
    # Default fallback
    return "riguarda"

def finalize_entity_clusters(all_entity_clusters: List[Dict], original_entities: List[Dict]) -> List[Dict]:
    """Finalizza i cluster delle entità combinando i dati."""
    final_entities = []
    processed_ids = set()
    
    for cluster in all_entity_clusters:
        if any(eid in processed_ids for eid in cluster["membri_ids"]):
            continue
        
        processed_ids.update(cluster["membri_ids"])
        cluster_data = combine_cluster_data(cluster, original_entities)
        final_entities.append(cluster_data)
    
    # Aggiungi entità non clusterizzate
    for i, entity in enumerate(original_entities):
        if i not in processed_ids:
            single_cluster = create_single_entity_cluster(entity, i)
            final_entities.append(single_cluster)
    
    return final_entities

def finalize_relation_clusters(all_relation_clusters: List[Dict], original_relations: List[Dict], final_entities: List[Dict]) -> List[Dict]:
    """Finalizza i cluster delle relazioni con mappatura entità."""
    
    # Crea mappa entità
    entity_map = {}
    for ce in final_entities:
        for member_name in ce["membri_cluster"]:
            entity_map[member_name.lower()] = ce["nome_entita_cluster"]
    
    final_relations = []
    processed_ids = set()
    
    for cluster in all_relation_clusters:
        if any(rid in processed_ids for rid in cluster["membri_ids"]):
            continue
        
        processed_ids.update(cluster["membri_ids"])
        
        # Mappa entità nel cluster
        s_mapped = entity_map.get(cluster["soggetto_cluster"].lower(), cluster["soggetto_cluster"])
        o_mapped = entity_map.get(cluster["oggetto_cluster"].lower(), cluster["oggetto_cluster"])
        
        cluster_data = combine_relations_cluster_data(cluster, original_relations)
        cluster_data["soggetto_cluster"] = s_mapped
        cluster_data["oggetto_cluster"] = o_mapped
        final_relations.append(cluster_data)
    
    # Aggiungi relazioni non clusterizzate
    for i, relation in enumerate(original_relations):
        if i not in processed_ids:
            s_mapped = entity_map.get(relation["soggetto_norm"], relation["soggetto_norm"])
            o_mapped = entity_map.get(relation["oggetto_norm"], relation["oggetto_norm"])
            single_cluster = create_single_relation_cluster(relation, i, s_mapped, o_mapped)
            final_relations.append(single_cluster)
    
    return final_relations

# ...existing code...

def combine_cluster_data(cluster: Dict, original_entities: List[Dict]) -> Dict:
    """Combina i dati delle entità appartenenti a un cluster."""
    
    membri_ids = cluster["membri_ids"]
    
    # Raccogli tutti i dati dalle entità del cluster
    all_names = []
    all_types = []
    all_descriptions = []
    all_chunk_ids = []
    all_page_nums = []
    all_sections = []
    total_occurrences = 0
    
    for entity_id in membri_ids:
        if entity_id < len(original_entities):
            entity = original_entities[entity_id]
            
            # Gestisce sia la struttura originale che quella migliorata
            if "nome_entita_canonico_provvisorio" in entity:
                # Struttura migliorata
                all_names.extend(entity.get("tutti_nomi_originali", []))
                all_types.extend(entity.get("tutti_tipi_rilevati", []))
                all_descriptions.extend(entity.get("descrizioni_aggregate", []))
                all_chunk_ids.extend(entity.get("fonti_chunk_id", []))
                all_page_nums.extend(entity.get("fonti_pagina", []))
                all_sections.extend(entity.get("fonti_sezione", []))
                total_occurrences += entity.get("conteggio_occorrenze", 0)
            else:
                # Struttura originale
                all_names.append(entity.get("nome_entita_aggregato", entity.get("nome_entita_norm", "")))
                all_types.append(entity.get("tipo_entita", ""))
                all_descriptions.extend(entity.get("descrizioni", []))
                all_chunk_ids.extend(entity.get("fonti_chunk_id", []))
                all_page_nums.extend(entity.get("fonti_pagina", []))
                all_sections.extend(entity.get("fonti_sezione", []))
                total_occurrences += entity.get("conteggio_occorrenze", 0)
    
    # Rimuovi duplicati e ordina
    return {
        "nome_entita_cluster": cluster.get("nome_cluster", "Entità_Sconosciuta"),
        "tipo_entita_cluster": cluster.get("tipo_cluster", "TipoSconosciuto"),
        "membri_cluster": sorted(list(set(filter(None, all_names)))),
        "tipi_membri_cluster": sorted(list(set(filter(None, all_types)))),
        "descrizioni_aggregate": sorted(list(set(filter(None, all_descriptions)))),
        "fonti_aggregate_chunk_id": sorted(list(set(filter(None, all_chunk_ids)))),
        "fonti_aggregate_pagina": sorted(list(set(filter(None, all_page_nums)))),
        "fonti_aggregate_sezione": sorted(list(set(filter(None, all_sections)))),
        "conteggio_occorrenze_totale": total_occurrences,
        "motivazione_clustering": cluster.get("motivazione", ""),
        "membri_ids_originali": membri_ids
    }

def create_single_entity_cluster(entity: Dict, entity_id: int) -> Dict:
    """Crea un cluster per una singola entità non raggruppata."""
    
    # Gestisce sia la struttura originale che quella migliorata
    if "nome_entita_canonico_provvisorio" in entity:
        # Struttura migliorata
        return {
            "nome_entita_cluster": entity["nome_entita_canonico_provvisorio"],
            "tipo_entita_cluster": entity["tipo_entita_canonico_provvisorio"],
            "membri_cluster": entity.get("tutti_nomi_originali", []),
            "tipi_membri_cluster": entity.get("tutti_tipi_rilevati", []),
            "descrizioni_aggregate": entity.get("descrizioni_aggregate", []),
            "fonti_aggregate_chunk_id": entity.get("fonti_chunk_id", []),
            "fonti_aggregate_pagina": entity.get("fonti_pagina", []),
            "fonti_aggregate_sezione": entity.get("fonti_sezione", []),
            "conteggio_occorrenze_totale": entity.get("conteggio_occorrenze", 0),
            "motivazione_clustering": "Entità singola, nessun raggruppamento necessario",
            "membri_ids_originali": [entity_id]
        }
    else:
        # Struttura originale
        return {
            "nome_entita_cluster": entity.get("nome_entita_aggregato", entity.get("nome_entita_norm", "")),
            "tipo_entita_cluster": entity.get("tipo_entita", ""),
            "membri_cluster": [entity.get("nome_entita_aggregato", entity.get("nome_entita_norm", ""))],
            "tipi_membri_cluster": [entity.get("tipo_entita", "")],
            "descrizioni_aggregate": entity.get("descrizioni", []),
            "fonti_aggregate_chunk_id": entity.get("fonti_chunk_id", []),
            "fonti_aggregate_pagina": entity.get("fonti_pagina", []),
            "fonti_aggregate_sezione": entity.get("fonti_sezione", []),
            "conteggio_occorrenze_totale": entity.get("conteggio_occorrenze", 0),
            "motivazione_clustering": "Entità singola, nessun raggruppamento necessario",
            "membri_ids_originali": [entity_id]
        }

def combine_relations_cluster_data(cluster: Dict, original_relations: List[Dict]) -> Dict:
    """Combina i dati delle relazioni appartenenti a un cluster."""
    
    membri_ids = cluster["membri_ids"]
    
    # Raccogli tutti i dati dalle relazioni del cluster
    all_contexts = []
    all_chunk_ids = []
    all_page_nums = []
    all_sections = []
    total_occurrences = 0
    original_predicates = []
    
    for relation_id in membri_ids:
        if relation_id < len(original_relations):
            relation = original_relations[relation_id]
            
            all_contexts.extend(relation.get("contesti", []))
            all_chunk_ids.extend(relation.get("fonti_chunk_id", []))
            all_page_nums.extend(relation.get("fonti_pagina", []))
            all_sections.extend(relation.get("fonti_sezione", []))
            total_occurrences += relation.get("conteggio_occorrenze", 0)
            original_predicates.append(relation.get("predicato_norm", ""))
    
    # Rimuovi duplicati e ordina
    return {
        "soggetto_cluster": cluster.get("soggetto_cluster", "Soggetto_Sconosciuto"),
        "predicato_cluster": cluster.get("predicato_cluster", "Predicato_Sconosciuto"),
        "oggetto_cluster": cluster.get("oggetto_cluster", "Oggetto_Sconosciuto"),
        "contesti_aggregati": sorted(list(set(filter(None, all_contexts)))),
        "fonti_aggregate_chunk_id": sorted(list(set(filter(None, all_chunk_ids)))),
        "fonti_aggregate_pagina": sorted(list(set(filter(None, all_page_nums)))),
        "fonti_aggregate_sezione": sorted(list(set(filter(None, all_sections)))),
        "predicati_originali_cluster": sorted(list(set(filter(None, original_predicates)))),
        "conteggio_occorrenze_totale": total_occurrences,
        "motivazione_clustering": cluster.get("motivazione", ""),
        "membri_ids_originali": membri_ids
    }

def create_single_relation_cluster(relation: Dict, relation_id: int, s_mapped: str, o_mapped: str) -> Dict:
    """Crea un cluster per una relazione singola non raggruppata."""
    
    return {
        "soggetto_cluster": s_mapped,
        "predicato_cluster": relation["predicato_norm"],
        "oggetto_cluster": o_mapped,
        "contesti_aggregati": relation.get("contesti", []),
        "fonti_aggregate_chunk_id": relation.get("fonti_chunk_id", []),
        "fonti_aggregate_pagina": relation.get("fonti_pagina", []),
        "fonti_aggregate_sezione": relation.get("fonti_sezione", []),
        "predicati_originali_cluster": [relation["predicato_norm"]],
        "conteggio_occorrenze_totale": relation.get("conteggio_occorrenze", 0),
        "motivazione_clustering": "Relazione singola, nessun raggruppamento necessario",
        "membri_ids_originali": [relation_id]
    }

def save_kg_to_json(data: List[Dict], filepath: str, description: str):
    """Salva i dati (entità o relazioni) in un file JSON."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{description} salvate in {filepath}")
    except IOError:
        print(f"Errore: Impossibile scrivere il file {description} a {filepath}")

if __name__ == "__main__":
    # Configura API Key
    api_key_from_env = os.getenv("GEMINI_API_KEY")
    if api_key_from_env:
        genai.configure(api_key=api_key_from_env)
        print("API Key Gemini caricata dalla variabile d'ambiente.")
    else:
        print("ATTENZIONE: API Key Gemini non trovata. Impostala nella variabile d'ambiente GEMINI_API_KEY.")
        exit()
    
    # Scelta del metodo di elaborazione
    print("\n=== SISTEMA DI ELABORAZIONE KNOWLEDGE GRAPH ===")
    print("1. Elaborazione standard (senza checkpoint)")
    print("2. Elaborazione con sistema di checkpoint completo")
    
    choice = input("Scegli il metodo (1 o 2): ").strip()
    
    input_json_path = "data\\processed\\processed_chunks_toc_enhanced.json"
    output_dir_llm = "llm_extraction_outputs"
    
    if choice == "1":
        # Elaborazione standard (codice originale)
        print("\n=== ELABORAZIONE STANDARD ===")
        
        # Definisci i percorsi di output
        output_entities_raw_path = "kg_entities_raw_empulia.json"
        output_relations_raw_path = "kg_relations_raw_empulia.json"
        output_entities_aggregated_improved_path = "kg_entities_aggregated_improved_empulia.json"
        output_relations_aggregated_improved_path = "kg_relations_aggregated_improved_empulia.json"
        output_entities_clustered_path = "kg_entities_clustered_final_empulia.json"
        output_relations_clustered_path = "kg_relations_clustered_final_empulia.json"

        # Carica i chunk
        document_chunks = load_chunks_from_json(input_json_path)

        if document_chunks:
            # Estrai conoscenza grezza
            raw_entities, raw_relations = extract_knowledge_from_chunks(document_chunks, output_dir_llm)
            save_kg_to_json(raw_entities, output_entities_raw_path, "Entità grezze")
            save_kg_to_json(raw_relations, output_relations_raw_path, "Relazioni grezze")

            # Aggrega e normalizza
            aggregated_entities_improved, aggregated_relations_improved = aggregate_knowledge_improved(raw_entities, raw_relations)
            save_kg_to_json(aggregated_entities_improved, output_entities_aggregated_improved_path, "Entità aggregate (versione migliorata)")
            save_kg_to_json(aggregated_relations_improved, output_relations_aggregated_improved_path, "Relazioni aggregate (versione migliorata)")

            # Clusterizza
            print("\n=== INIZIO CLUSTERING COMBINATO CON LLM ===")
            final_clustered_entities, final_clustered_relations = llm_cluster_knowledge(
                aggregated_entities_improved, 
                aggregated_relations_improved
            )

            save_kg_to_json(final_clustered_entities, output_entities_clustered_path, "Entità clusterizzate finali (LLM)")
            save_kg_to_json(final_clustered_relations, output_relations_clustered_path, "Relazioni clusterizzate finali (LLM)")

            print("\n--- Generazione Knowledge Graph Completata ---")
            print(f"Entità finali: {len(final_clustered_entities)}")
            print(f"Relazioni finali: {len(final_clustered_relations)}")
            print(f"Riduzione entità tramite clustering: {len(aggregated_entities_improved)} → {len(final_clustered_entities)}")
            print(f"Riduzione relazioni tramite clustering: {len(aggregated_relations_improved)} → {len(final_clustered_relations)}")

        else:
            print(f"Nessun chunk caricato da {input_json_path}. Verifica il file.")
    
    elif choice == "2":
        # Elaborazione con checkpoint - import locale per evitare circolarità
        print("\n=== ELABORAZIONE CON CHECKPOINT ===")
        from KG_checkpoint import process_with_full_checkpoint_system
        process_with_full_checkpoint_system(input_json_path, output_dir_llm)
    
    else:
        print("Scelta non valida. Esecuzione terminata.")