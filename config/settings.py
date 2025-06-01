# config/settings.py
"""
Główny plik konfiguracyjny dla systemu RAG AI Devs 3 Reloaded
Centralizuje wszystkie ustawienia i konfiguracje
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging
from enum import Enum

# Wczytaj zmienne środowiskowe
load_dotenv()

# ============== ŚCIEŻKI ==============

# Ścieżka główna projektu
BASE_DIR = Path(__file__).resolve().parent.parent

# Katalogi danych
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Katalogi logów
LOGS_DIR = BASE_DIR / "logs"

# Tworzenie katalogów jeśli nie istnieją
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============== KONFIGURACJA OPENAI ==============

@dataclass
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    organization: Optional[str] = os.getenv("OPENAI_ORGANIZATION")
    
    # Modele
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4-turbo-preview")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Parametry
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))
    
    # Rate limiting
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "50"))
    tokens_per_minute: int = int(os.getenv("TOKENS_PER_MINUTE", "40000"))
    
    # Retry
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))

# ============== KONFIGURACJA POSTGRESQL ==============

@dataclass
class PostgreSQLConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "aidevs3_rag")
    user: str = os.getenv("POSTGRES_USER", "aidevs")
    password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    # Connection pool
    min_connections: int = int(os.getenv("POSTGRES_MIN_CONNECTIONS", "2"))
    max_connections: int = int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10"))
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

# ============== KONFIGURACJA QDRANT ==============

@dataclass
class QdrantConfig:
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # Kolekcje
    default_collection: str = os.getenv("QDRANT_COLLECTION", "aidevs3_documents")
    
    # Parametry wektorów
    vector_size: int = int(os.getenv("VECTOR_SIZE", "1536"))  # dla text-embedding-3-small
    distance_metric: str = os.getenv("DISTANCE_METRIC", "Cosine")
    
    # Parametry wyszukiwania
    search_limit: int = int(os.getenv("SEARCH_LIMIT", "10"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.7"))
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

# ============== KONFIGURACJA NEO4J ==============

@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    
    # Database
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Connection pool
    max_connection_lifetime: int = int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600"))
    max_connection_pool_size: int = int(os.getenv("NEO4J_MAX_POOL_SIZE", "50"))
    connection_acquisition_timeout: int = int(os.getenv("NEO4J_ACQUISITION_TIMEOUT", "60"))

# ============== KONFIGURACJA REDIS (CACHE) ==============

@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    db: int = int(os.getenv("REDIS_DB", "0"))
    
    # TTL dla różnych typów cache
    embedding_ttl: int = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))  # 24h
    document_ttl: int = int(os.getenv("DOCUMENT_CACHE_TTL", "3600"))    # 1h
    answer_ttl: int = int(os.getenv("ANSWER_CACHE_TTL", "1800"))        # 30min
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

# ============== KONFIGURACJA PRZETWARZANIA DOKUMENTÓW ==============

@dataclass
class DocumentProcessingConfig:
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Separatory dla różnych typów dokumentów
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ": ",
        " ",
        ""
    ])
    
    # Limity
    max_document_size_mb: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "8000"))
    
    # Whisper
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    whisper_language: str = os.getenv("WHISPER_LANGUAGE", "pl")
    
    # OCR
    tesseract_lang: str = os.getenv("TESSERACT_LANG", "pol+eng")
    
    # Obsługiwane formaty
    supported_formats: Set[str] = {
        ".txt", ".pdf", ".html", ".htm", ".md", ".json",
        ".mp3", ".wav", ".m4a", ".ogg",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp",
        ".docx", ".xlsx", ".csv"
    }

# ============== KONFIGURACJA EKSTRAKCJI ENCJI ==============

@dataclass
class EntityExtractionConfig:
    # Typy encji
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
        "EVENT", "PRODUCT", "TECHNOLOGY", "CONCEPT", "DOCUMENT"
    ])
    
    # Modele NER
    use_spacy: bool = os.getenv("USE_SPACY", "true").lower() == "true"
    spacy_model: str = os.getenv("SPACY_MODEL", "pl_core_news_lg")
    
    # Ekstrakcja relacji
    extract_relations: bool = os.getenv("EXTRACT_RELATIONS", "true").lower() == "true"
    relation_types: List[str] = field(default_factory=lambda: [
        "WORKS_FOR", "LOCATED_IN", "KNOWS", "CREATED_BY",
        "MENTIONED_IN", "RELATED_TO", "PART_OF", "HAPPENED_AT"
    ])
    
    # Confidence thresholds
    entity_confidence_threshold: float = float(os.getenv("ENTITY_CONFIDENCE_THRESHOLD", "0.8"))
    relation_confidence_threshold: float = float(os.getenv("RELATION_CONFIDENCE_THRESHOLD", "0.7"))

# ============== KONFIGURACJA AGENTÓW ==============

@dataclass
class AgentConfig:
    # Typy pytań i odpowiadające im agenty
    question_type_agents: Dict[str, List[str]] = field(default_factory=lambda: {
        "factual": ["sql_agent"],
        "semantic": ["semantic_agent"],
        "relational": ["graph_agent"],
        "complex": ["sql_agent", "semantic_agent", "graph_agent"],
        "temporal": ["sql_agent", "graph_agent"]
    })
    
    # Parametry orchestratora
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
    agent_timeout: int = int(os.getenv("AGENT_TIMEOUT", "30"))
    
    # Strategia agregacji odpowiedzi
    aggregation_strategy: str = os.getenv("AGGREGATION_STRATEGY", "weighted")  # weighted, voting, llm
    
    # Wagi dla różnych źródeł (używane w strategii "weighted")
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "sql": 0.3,
        "semantic": 0.4,
        "graph": 0.3
    })

# ============== MAPOWANIE ZADAŃ DO BAZ DANYCH ==============

TASK_DATABASE_REQUIREMENTS: Dict[str, List[str]] = {
    # Zadania podstawowe - tylko cache
    "S01": [],
    "S02": [],
    "S03": [],
    
    # Zadania z przetwarzaniem tekstu - PostgreSQL
    "S04": ["postgresql"],
    "S05": ["postgresql"],
    "S06": ["postgresql"],
    "S07": ["postgresql"],
    
    # Zadania z embeddingami - PostgreSQL + Qdrant
    "S08": ["postgresql", "qdrant"],
    "S09": ["postgresql", "qdrant"],
    "S10": ["postgresql", "qdrant"],
    "S11": ["postgresql", "qdrant"],
    "S12": ["postgresql", "qdrant"],
    
    # Zadania z relacjami - wszystkie bazy
    "S13": ["postgresql", "qdrant", "neo4j"],
    "S14": ["postgresql", "qdrant", "neo4j"],
    "S15": ["postgresql", "qdrant", "neo4j"],
    "S16": ["postgresql", "qdrant", "neo4j"],
    "S17": ["postgresql", "qdrant", "neo4j"],
    "S18": ["postgresql", "qdrant", "neo4j"],
    "S19": ["postgresql", "qdrant", "neo4j"],
    
    # Zadania zaawansowane - wszystkie bazy
    "S20": ["postgresql", "qdrant", "neo4j"],  # dane
    "S21": ["postgresql", "qdrant", "neo4j"],
    "S22": ["postgresql", "qdrant", "neo4j"],
    "S23": ["postgresql", "qdrant", "neo4j"],
    "S24": ["postgresql", "qdrant", "neo4j"],
    "S25": ["postgresql", "qdrant", "neo4j"],  # finalne pytania
}

# ============== KONFIGURACJA LOGOWANIA ==============

@dataclass
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Osobne poziomy dla różnych modułów
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        "database": "INFO",
        "extractors": "INFO",
        "processors": "INFO",
        "agents": "DEBUG",
        "openai": "WARNING",
        "httpx": "WARNING"
    })
    
    # Rotacja logów
    max_bytes: int = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# ============== KONFIGURACJA SYSTEMU ==============

@dataclass
class SystemConfig:
    # Tryb pracy
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")  # development, staging, production
    
    # Współbieżność
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    async_enabled: bool = os.getenv("ASYNC_ENABLED", "true").lower() == "true"
    
    # Monitoring
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # Bezpieczeństwo
    enable_rate_limiting: bool = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

# ============== GŁÓWNA KLASA KONFIGURACJI ==============

class Config:
    """Główna klasa agregująca wszystkie konfiguracje"""
    
    def __init__(self):
        self.openai = OpenAIConfig()
        self.postgresql = PostgreSQLConfig()
        self.qdrant = QdrantConfig()
        self.neo4j = Neo4jConfig()
        self.redis = RedisConfig()
        self.document_processing = DocumentProcessingConfig()
        self.entity_extraction = EntityExtractionConfig()
        self.agents = AgentConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
        
        # Ścieżki
        self.base_dir = BASE_DIR
        self.data_dir = DATA_DIR
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.cache_dir = CACHE_DIR
        self.logs_dir = LOGS_DIR
        
        # Mapowanie zadań
        self.task_database_requirements = TASK_DATABASE_REQUIREMENTS
        
        # Walidacja konfiguracji
        self._validate_config()
        
        # Konfiguracja logowania
        self._setup_logging()
    
    def _validate_config(self):
        """Walidacja kluczowych ustawień"""
        errors = []
        
        # Sprawdź klucze API
        if not self.openai.api_key:
            errors.append("OPENAI_API_KEY nie jest ustawiony")
        
        # Sprawdź połączenia z bazami
        if not self.postgresql.password:
            errors.append("POSTGRES_PASSWORD nie jest ustawiony")
        
        if not self.neo4j.password:
            errors.append("NEO4J_PASSWORD nie jest ustawiony")
        
        # Sprawdź katalogi
        if not self.data_dir.exists():
            errors.append(f"Katalog danych nie istnieje: {self.data_dir}")
        
        if errors and self.system.environment == "production":
            raise ValueError(f"Błędy konfiguracji: {', '.join(errors)}")
        elif errors:
            for error in errors:
                logging.warning(f"Ostrzeżenie konfiguracji: {error}")
    
    def _setup_logging(self):
        """Konfiguracja systemu logowania"""
        log_level = getattr(logging, self.logging.level.upper())
        
        # Główny logger
        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            handlers=[
                logging.FileHandler(self.logs_dir / "aidevs3_rag.log"),
                logging.StreamHandler()
            ]
        )
        
        # Ustawienie poziomów dla modułów
        for module, level in self.logging.module_levels.items():
            logging.getLogger(module).setLevel(getattr(logging, level.upper()))
    
    def get_task_databases(self, task_id: str) -> List[str]:
        """Zwraca listę wymaganych baz danych dla zadania"""
        return self.task_database_requirements.get(task_id, [])
    
    def is_production(self) -> bool:
        """Sprawdza czy system działa w trybie produkcyjnym"""
        return self.system.environment == "production"
    
    def get_connection_params(self, database: str) -> Dict:
        """Zwraca parametry połączenia dla danej bazy"""
        if database == "postgresql":
            return {
                "host": self.postgresql.host,
                "port": self.postgresql.port,
                "database": self.postgresql.database,
                "user": self.postgresql.user,
                "password": self.postgresql.password
            }
        elif database == "qdrant":
            return {
                "url": self.qdrant.url,
                "api_key": self.qdrant.api_key
            }
        elif database == "neo4j":
            return {
                "uri": self.neo4j.uri,
                "auth": (self.neo4j.user, self.neo4j.password)
            }
        elif database == "redis":
            return {
                "host": self.redis.host,
                "port": self.redis.port,
                "password": self.redis.password,
                "db": self.redis.db
            }
        else:
            raise ValueError(f"Nieznana baza danych: {database}")
    
    def __repr__(self):
        return f"Config(environment={self.system.environment}, debug={self.system.debug})"

# ============== SINGLETON ==============

# Globalna instancja konfiguracji
_config_instance: Optional[Config] = None

def get_config() -> Config:
    """Zwraca singleton instancji konfiguracji"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

# ============== EKSPORT ==============

# Eksportuj główną funkcję i klasę
__all__ = ["Config", "get_config"]

# Przykład użycia
if __name__ == "__main__":
    config = get_config()
    print(f"Konfiguracja załadowana: {config}")
    print(f"OpenAI API Key obecny: {'Tak' if config.openai.api_key else 'Nie'}")
    print(f"Wymagane bazy dla S25: {config.get_task_databases('S25')}")
