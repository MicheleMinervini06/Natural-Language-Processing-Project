from typing import List, Dict, Set
from functools import lru_cache

class EntityNormalizer:
    """
    Classe per la normalizzazione consistente delle entità.
    """
    
    # Mappa di normalizzazione condivisa
    NORMALIZATION_MAP = {
        # Password e varianti
        "password": "password",
        "parola d'accesso": "password",
        "credenziali": "password",
        "pwd": "password",
        
        # DGUE e varianti
        "dgue": "dgue",
        "documento unico europeo": "dgue",
        "documento unico": "dgue",
        "d.g.u.e.": "dgue",
        
        # EmPULIA e varianti
        "empulia": "empulia",
        "piattaforma empulia": "empulia",
        "sistema empulia": "empulia",
        "em pulia": "empulia",
        
        # Albo Fornitori e varianti
        "albo fornitori": "albo fornitori",
        "albo": "albo fornitori",
        "registro fornitori": "albo fornitori",
        "elenco fornitori": "albo fornitori",
        
        # Iscrizione e varianti
        "iscrizione": "iscrizione",
        "registrazione": "iscrizione",
        "iscrizione albo": "iscrizione",
        "registrazione albo": "iscrizione",
        
        # Fornitore e varianti
        "fornitore": "fornitore",
        "operatore economico": "fornitore",
        "impresa": "fornitore",
        "ditta": "fornitore",
        "società": "fornitore",
        
        # Documenti e varianti
        "documento": "documento",
        "documentazione": "documento",
        "documentazione richiesta": "documento",
        "allegato": "documento",
        
        # Accesso e varianti
        "accesso": "accesso",
        "login": "accesso",
        "autenticazione": "accesso",
        "log in": "accesso",
        "entrata": "accesso",
        
        # Procedure e varianti
        "procedura": "procedura",
        "processo": "procedura",
        "iter": "procedura",
        "prassi": "procedura",
        
        # Cambiare/Modificare e varianti
        "cambiare": "modificare",
        "modificare": "modificare",
        "aggiornare": "modificare",
        "sostituire": "modificare",
        "reimpostare": "modificare",
        
        # Reset e varianti
        "reset": "reset",
        "ripristino": "reset",
        "azzeramento": "reset",
        "reimpostazione": "reset"
    }
    
    # Stop words italiane comuni
    STOP_WORDS = {
        "il", "la", "lo", "gli", "le", "i", 
        "un", "una", "uno", 
        "del", "della", "dei", "delle", "dello", "degli",
        "di", "da", "in", "con", "per", "su", "tra", "fra", 
        "a", "e", "o", "ma", "se", "come", "quando", "dove",
        "che", "chi", "cui", "quanto", "quale", "questo", "quello",
        "mio", "tuo", "suo", "nostro", "vostro", "loro",
        "mia", "tua", "sua", "nostra", "vostra"
    }
    
    @classmethod
    def normalize_entity_name(cls, name: str) -> str:
        """
        Normalizza il nome di un'entità per garantire consistenza nel grafo.
        
        Args:
            name: Nome dell'entità da normalizzare
            
        Returns:
            Nome normalizzato dell'entità
        """
        if not name or not name.strip():
            return ""
        
        # Converte in minuscolo e rimuove spazi extra
        normalized = name.strip().lower()
        
        # Normalizza caratteri speciali comuni
        normalized = normalized.replace("'", "'").replace("`", "'")
        normalized = normalized.replace("–", "-").replace("—", "-")
        
        # Controlla se c'è una normalizzazione diretta
        if normalized in cls.NORMALIZATION_MAP:
            return cls.NORMALIZATION_MAP[normalized]
        
        # Controlla se contiene parole chiave da normalizzare
        for key, value in cls.NORMALIZATION_MAP.items():
            if key in normalized:
                return value
        
        # Rimuovi articoli e preposizioni comuni
        words = normalized.split()
        filtered_words = [word for word in words if word not in cls.STOP_WORDS and len(word) > 1]
        
        if filtered_words:
            return " ".join(filtered_words)
        
        return normalized
    
    @classmethod
    def normalize_entity_list(cls, entities: List[Dict]) -> List[Dict]:
        """
        Normalizza una lista di entità, aggiungendo il nome normalizzato.
        
        Args:
            entities: Lista di dizionari con chiave 'nome'
            
        Returns:
            Lista di entità con campi 'nome_originale' e 'nome_normalizzato'
        """
        normalized_entities = []
        
        for entity in entities:
            original_name = entity.get("nome", "")
            normalized_name = cls.normalize_entity_name(original_name)
            
            if normalized_name:
                new_entity = entity.copy()
                new_entity["nome_originale"] = original_name
                new_entity["nome_normalizzato"] = normalized_name
                normalized_entities.append(new_entity)
        
        return normalized_entities
    
    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        """
        Estrae parole chiave da un testo, applicando normalizzazione.
        
        Args:
            text: Testo da cui estrarre parole chiave
            
        Returns:
            Lista di parole chiave normalizzate
        """
        if not text:
            return []
        
        # Normalizza il testo
        normalized_text = text.strip().lower()
        
        # Dividi in parole
        words = normalized_text.split()
        
        # Filtra e normalizza ogni parola
        keywords = []
        for word in words:
            if word not in cls.STOP_WORDS and len(word) > 2:
                normalized_word = cls.normalize_entity_name(word)
                if normalized_word and normalized_word not in keywords:
                    keywords.append(normalized_word)
        
        return keywords
    
    @classmethod
    def create_search_patterns(cls, entity_name: str) -> List[str]:
        """
        Crea pattern di ricerca per un'entità, includendo varianti.
        
        Args:
            entity_name: Nome dell'entità
            
        Returns:
            Lista di pattern di ricerca
        """
        patterns = []
        
        # Nome originale
        if entity_name:
            patterns.append(entity_name.lower())
        
        # Nome normalizzato
        normalized = cls.normalize_entity_name(entity_name)
        if normalized and normalized not in patterns:
            patterns.append(normalized)
        
        # Trova varianti inverse dalla mappa
        for key, value in cls.NORMALIZATION_MAP.items():
            if value == normalized and key not in patterns:
                patterns.append(key)
        
        # Aggiungi pattern per parole singole se l'entità è composta
        words = normalized.split() if normalized else []
        for word in words:
            if len(word) > 2 and word not in patterns:
                patterns.append(word)
        
        return patterns

# Funzioni di convenienza per compatibilità
def normalize_entity_name(name: str) -> str:
    """Funzione di convenienza per normalizzare un singolo nome."""
    return EntityNormalizer.normalize_entity_name(name)

def normalize_entity_list(entities: List[Dict]) -> List[Dict]:
    """Funzione di convenienza per normalizzare una lista di entità."""
    return EntityNormalizer.normalize_entity_list(entities)

def extract_keywords(text: str) -> List[str]:
    """Funzione di convenienza per estrarre parole chiave."""
    return EntityNormalizer.extract_keywords(text)

def create_search_patterns(entity_name: str) -> List[str]:
    """Funzione di convenienza per creare pattern di ricerca."""
    return EntityNormalizer.create_search_patterns(entity_name)