## Action Plan

**Fase 1: Costruzione della Knowledge Base (KG) dai PDF delle Guide EmPULIA**

1.  **Preprocessing dei PDF:**
    *   **Estrazione del Testo:** Il primo passo è estrarre il testo pulito dai PDF. Strumenti come `PyPDF2`, `pdfminer.six` o servizi cloud possono essere utili. Considera la struttura dei PDF (intestazioni, tabelle, elenchi) e cerca di preservarla o convertirla in un formato testuale più lineare.
    *   **Chunking (Suddivisione del Testo):** Dividi il testo estratto in "chunk" (paragrafi, sezioni o blocchi di dimensioni fisse). Questo è importante perché gli LLM hanno una finestra di contesto limitata. La dimensione dei chunk influenzerà la granularità delle informazioni estratte. (Rif. *GraphRAG* per l'importanza del chunking).

2.  **Generazione del Knowledge Graph tramite LLM:**
    *   **Selezione dell'LLM Estrattore:** Scegli un LLM potente per l'estrazione di informazioni (es. GPT-4, modelli open-source fine-tuned per estrazione, o modelli specifici se disponibili).
    *   **Prompt Engineering per Estrazione:**
        *   **Estrazione di Entità:** Definisci le tipologie di entità rilevanti per EmPULIA. Potrebbero essere:
            *   `FunzionalitàPiattaforma` (es. "Presentazione Domanda", "Gestione Notifiche", "Firma Digitale")
            *   `RuoloUtente` (es. "Operatore PA", "Cittadino", "Impresa")
            *   `DocumentoRichiesto` (es. "Carta d'Identità", "Visura Camerale")
            *   `AzioneUtente` (es. "Cliccare su Salva", "Compilare il Modulo X", "Allegare Documento")
            *   `Prerequisito` (es. "Possesso SPID", "Iscrizione al Registro Imprese")
            *   `MessaggioErrore` (es. "Dati Mancanti", "Firma Non Valida")
            *   `TermineTemporale` (es. "Scadenza Bando", "30 giorni dalla notifica")
        *   **Estrazione di Relazioni:** Definisci le relazioni tra queste entità. Esempi:
            *   `richiede` (FunzionalitàPiattaforma `richiede` DocumentoRichiesto)
            *   `esegue` (RuoloUtente `esegue` AzioneUtente)
            *   `haPrerequisito` (FunzionalitàPiattaforma `haPrerequisito` Prerequisito)
            *   `portaA` (AzioneUtente `portaA` FunzionalitàPiattaforma)
            *   `visualizza` (RuoloUtente `visualizza` FunzionalitàPiattaforma)
            *   `genera` (AzioneUtente `genera` MessaggioErrore)
            *   `haScadenza` (FunzionalitàPiattaforma `haScadenza` TermineTemporale)
            *   `èParteDi` (AzioneUtente `èParteDi` FunzionalitàPiattaforma)
        *   Usa tecniche di prompt engineering come quelle descritte in *KGGen* (Mo et al.): fornisci esempi (few-shot), definisci chiaramente il formato di output (es. JSON con triple soggetto-predicato-oggetto), e magari istruisci l'LLM a fornire anche descrizioni per entità e relazioni.
    *   **Iterazione e Clustering (come in *KGGen*):**
        *   Dopo una prima estrazione, aggrega le triple da tutti i chunk/documenti.
        *   Utilizza un LLM (o tecniche di clustering più tradizionali supportate da LLM per la validazione) per identificare e raggruppare entità e relazioni sinonime o semanticamente molto vicine (es. "Carica documento" vs "Allega file"). Questo è cruciale per creare un KG denso e utile.
        *   Normalizza i nomi delle entità e delle relazioni.
    *   **Memorizzazione del KG:** Scegli un database per grafi (es. Neo4j, Amazon Neptune, o anche librerie in-memory come `RDFLib` per prototipi) per memorizzare le triple generate.

**Fase 2: Sviluppo del Sistema di Question Answering (LLM + KG)**

1.  **Architettura di Q&A:**
    *   L'idea è combinare la comprensione del linguaggio naturale dell'LLM con la conoscenza fattuale e strutturata del KG. Si può pensare a un approccio RAG (Retrieval Augmented Generation) dove il recupero avviene *anche o principalmente* dal KG.

2.  **Componente di Comprensione della Domanda (LLM):**
    *   Quando l'utente pone una domanda (es. "Come presento una domanda per il bando X?", "Cosa devo fare se dimentico la password?"), un LLM analizza la domanda per:
        *   Identificare le entità chiave menzionate (es. "bando X", "password").
        *   Capire l'intento dell'utente (es. cerca una procedura, una soluzione a un problema).
        *   Riformulare la domanda in modo più strutturato se necessario, o scomporla in sotto-domande.

3.  **Componente di Recupero della Conoscenza (KG + opzionalmente testo originale):**
    *   **Recupero dal KG:**
        *   Trasforma le entità e l'intento identificati dall'LLM in query per il KG (es. SPARQL, Cypher, o query testuali che poi un altro LLM traduce in query formali).
        *   Recupera le triple rilevanti dal KG. Ad esempio, se la domanda è "Cosa serve per presentare la domanda Y?", si potrebbero cercare le triple: (`Presentazione Domanda Y`, `richiede`, `?documento`) e (`Presentazione Domanda Y`, `haPrerequisito`, `?prerequisito`).
    *   **Recupero dal Testo Originale (opzionale ma consigliato per contesto):**
        *   Se le triple del KG non sono sufficientemente esplicative, o se la domanda richiede dettagli specifici non catturati nel KG, si può anche recuperare i chunk di testo originali da cui quelle triple sono state estratte, o chunk semanticamente simili alla domanda. Questo è il RAG più tradizionale. L'approccio *GraphRAG* (Edge et al.) potrebbe essere molto interessante qui, specialmente se le guide sono lunghe e le domande richiedono una visione d'insieme ("Quali sono i passaggi principali per interagire con la piattaforma come impresa?").

4.  **Componente di Generazione della Risposta (LLM):**
    *   L'LLM riceve in input:
        *   La domanda originale dell'utente.
        *   Le triple/informazioni recuperate dal KG.
        *   Eventuali chunk di testo recuperati.
    *   L'LLM sintetizza queste informazioni per generare una risposta coerente, completa e in linguaggio naturale.
    *   **Citazione delle Fonti/Spiegabilità:** Idealmente, la risposta dovrebbe indicare da quali sezioni della guida (o quali fatti del KG) proviene l'informazione, per aumentare la fiducia e permettere all'utente di approfondire (Rif. *AlKhamissi et al.* sulla trasparenza dei KG).

5.  **Interfaccia Utente e Feedback:**
    *   Un'interfaccia conversazionale.
    *   Meccanismo di feedback per migliorare il sistema (es. l'utente valuta l'utilità della risposta).

**Considerazioni Aggiuntive Basate sui Paper:**

*   **Riduzione delle Allucinazioni (*Agrawal et al.*):** Fare grounding delle risposte dell'LLM sui fatti recuperati dal KG è una strategia chiave per ridurre le allucinazioni. Il KG funge da "ancora di verità".
*   **Freschezza e Modificabilità (*AlKhamissi et al.*):** Se le guide EmPULIA vengono aggiornate frequentemente, il KG può essere aggiornato più facilmente rispetto a ri-addestrare un LLM. La pipeline di generazione del KG dovrebbe essere rieseguita (o aggiornata incrementalmente) quando i PDF cambiano.
*   **Consistenza (*AlKhamissi et al.*):** Il KG aiuta a garantire che risposte a domande simili o parafrasate siano coerenti, poiché si basano sullo stesso set di fatti strutturati.
*   **Domande Complesse/Multi-Hop (*Hogan et al.*):** Il KG è particolarmente utile per rispondere a domande che richiedono di combinare più pezzi di informazione (es. "Un cittadino che vuole accedere al servizio X e ha SPID di livello 2, quali altri documenti deve preparare?").
*   **Validazione del KG:** Anche se l'LLM aiuta a generare il KG, una qualche forma di validazione (magari umana a campione, o tramite regole di coerenza) potrebbe essere necessaria per assicurare l'accuratezza del KG stesso.

**Prossimi Passi Suggeriti:**

1.  **Analisi Dettagliata dei PDF:** Comprendi la struttura tipica delle guide EmPULIA. Ci sono schemi ricorrenti? Tabelle? Glossari? Questo informerà il preprocessing e il prompt engineering.
2.  **Definizione dell'Ontologia/Schema del KG:** Sulla base dell'analisi dei PDF e del tipo di domande che prevedi, finalizza le tipologie di entità e relazioni.
3.  **Prototipazione della Generazione KG:** Inizia con un singolo PDF o una sezione. Sperimenta con i prompt per l'estrazione di entità e relazioni. Valuta la qualità dell'output dell'LLM.
4.  **Prototipazione del Q&A:** Implementa un flusso semplice: domanda utente -> LLM per estrarre entità dalla domanda -> query al KG (anche manuale all'inizio) -> LLM per generare risposta dalle triple.

**Prossimi Passi**
-
- ~~Tentare di ricreare la KB~~
- Riconsiderare il golden datset, in quanto la recall/precision sono propabilmente caloclate sulla base dei chunk di riferimento e la KB non viene di fatti considerata
- ~~Controllare se i chunk vengano effetivamente recuperato con ilk nuovo approccio~~

- Miglioramento #1: Ibridare la Ricerca "Anchor" (Keyword + Vettoriale)

    Attualmente, la tua fase di "Ancoraggio" (trovare i nodi iniziali) si basa su una ricerca testuale con `CONTAINS`. Questo è efficace ma può mancare le sfumature semantiche. Il passo successivo è renderla ibrida.
    1.  **Crea Indici Vettoriali:** Se non l'hai già fatto, crea un indice vettoriale in Neo4j sulle proprietà `name` e `description` dei tuoi nodi.
    2.  **Modifica la Fase di Ancoraggio:** Invece di una sola `MATCH`, eseguine due in parallelo:
        *   Una `MATCH` con `CONTAINS` come quella attuale per catturare le corrispondenze esatte delle parole chiave.
        *   Una chiamata all'indice vettoriale (`CALL db.index.vector.queryNodes`) per trovare i nodi semanticamente più simili alla domanda dell'utente (o ai termini espansi).
    3.  **Unisci i Risultati:** Prendi i nodi trovati da entrambi gli approcci e usali come punto di partenza per la fase di "Espansione".

    **Perché Funziona:**
    *   Otterrai il meglio di entrambi i mondi. La ricerca per keyword è ottima per acronimi e termini tecnici precisi (come "DGUE"). La ricerca vettoriale è eccellente per catturare il significato e le domande parafrasate (come "come si fa a firmare un contratto?" che potrebbe non contenere la parola "sottoscrizione").
    *   Questo aumenterà drasticamente il `Context Recall` perché avrai molte più probabilità di trovare i nodi "ancora" corretti.

- Miglioramento #2: Implementare un "Reranker" per il Contesto Recuperato

    La tua strategia di espansione ("Expand") è ottima ma potrebbe recuperare più contesto del necessario, includendo nodi vicini ma non strettamente pertinenti. Questo può "annacquare" il contesto finale e abbassare la `Context Precision`.

   **Cosa Fare:**
    1.  **Recupera un Contesto Ampio:** Esegui la tua pipeline di recupero (quella ibrida del Miglioramento #1) per ottenere un insieme di potenziali nodi/chunk di contesto (diciamo 10-15 candidati).
    2.  **Fase di Reranking:** Prima di passare questo contesto al prompt finale, introduci un nuovo passaggio. Utilizza un modello specializzato (un "reranker" o cross-encoder, o anche una chiamata mirata a Gemini) per calcolare un punteggio di pertinenza per **ciascun pezzo di contesto recuperato** rispetto alla **domanda originale dell'utente**.
    3.  **Filtra e Ordina:** Riordina i pezzi di contesto in base a questo punteggio e passa al prompt finale solo i top 3-5 più pertinenti.

   **Perché Funziona:**
    *   Questo agisce come un filtro di qualità. Permette alla fase di recupero di essere "aggressiva" (alto `Recall`), mentre la fase di reranking garantisce che solo le informazioni migliori vengano utilizzate per la risposta finale (alto `Precision`).
    *   Migliorerà drasticamente la `Context Precision` e, di conseguenza, la `Faithfulness`, perché l'LLM finale riceverà un contesto più pulito e focalizzato.

- Miglioramento #3: Esplorazione Guidata dalle Relazioni ("Relationship-Aware Traversal")

    La tua attuale esplorazione del grafo tratta tutte le connessioni allo stesso modo. Possiamo renderla più intelligente, facendole dare priorità a certi tipi di relazioni in base all'intento della domanda.

    **Cosa Fare:**
    1.  **Mappa Intenti a Relazioni:** Crea una piccola mappa logica. Ad esempio:
        *   Se l'intento è `find_procedure`, le relazioni importanti sono `HA_PASSO_SUCCESSIVO`, `RICHIEDE`, `ESEGUITA_DA`.
        *   Se l'intento è `find_definition`, le relazioni importanti sono `SI_APPLICA_A`, `CONTIENE_ELEMENTO`.
    2.  **Modifica la Query di Espansione:** Rendi la tua query Cypher dinamica. Dopo aver trovato i nodi "ancora", invece di un generico `(anchor)-[r]-(direct_neighbor)`, specifica i tipi di relazione a cui dare la priorità.

    *Esempio di query Cypher per un intento procedurale:*
    ```cypher
    // ... fase di ancoraggio ...
    WITH anchor
    // Espansione che favorisce le relazioni procedurali
    OPTIONAL MATCH p=(anchor)-[:RICHIEDE|ESEGUITA_DA*1..2]-(procedural_neighbor)
    // ... raccogli e restituisci i nodi del path 'p' ...
    ```

   **Perché Funziona:**
    *   Questo trasforma la tua esplorazione da una "passeggiata casuale" a un'"indagine mirata". Il sistema non solo trova i vicini, ma trova i vicini *giusti* per il tipo di domanda posta.
    *   Questo migliorerà ulteriormente sia il `Context Recall` (trovando percorsi più lunghi ma significativi) sia la `Context Precision` (ignorando le connessioni irrilevanti).