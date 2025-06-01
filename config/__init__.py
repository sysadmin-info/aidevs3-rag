# config/__init__.py
"""
Moduł konfiguracji dla systemu RAG AI Devs 3 Reloaded

Centralizuje wszystkie ustawienia, konfiguracje baz danych,
prompty LLM oraz parametry systemu.

Przykład użycia:
    from config import config, get_database_schemas, LLMPrompts
    
    # Dostęp do konfiguracji
    api_key = config.openai.api_key
    
    # Inicjalizacja baz
    schemas = get_database_schemas()
    
    # Użycie promptów
    prompt = LLMPrompts.get_system_prompt("sql_agent")
"""

import logging
from typing import Dict, Any, Optional

# Import głównej konfiguracji
from .settings import (
    Config,
    get_config,
    OpenAIConfig,
    PostgreSQLConfig,
    QdrantConfig,
    Neo4jConfig,
    RedisConfig,
    DocumentProcessingConfig,
    EntityExtractionConfig,
    AgentConfig,
    LoggingConfig,
    SystemConfig,
    TASK_DATABASE_REQUIREMENTS,
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    LOGS_DIR
)

# Import konfiguracji baz danych
from .database_config import (
    POSTGRES_SCHEMAS,
    POSTGRES_VIEWS,
    POSTGRES_FUNCTIONS,
    QDRANT_COLLECTIONS,
    NEO4J_NODE_TYPES,
    NEO4J_RELATIONSHIP_TYPES,
    NEO4J_ANALYSIS_QUERIES,
    DatabaseInitializer,
    SQL_QUERIES,
    DATABASE_MIGRATIONS
)

# Import konfiguracji LLM (jeśli istnieje)
try:
    from .llm_config import (
        LLMPrompts,
        SYSTEM_PROMPTS,
        AGENT_PROMPTS,
        TASK_PROMPTS,
        get_prompt
    )
    _llm_config_available = True
except ImportError:
    _llm_config_available = False
    logging.debug("llm_config.py nie jest jeszcze dostępny")

# Import konfiguracji logowania (jeśli istnieje)
try:
    from .logging_config import setup_logging, get_logger
    _logging_config_available = True
except ImportError:
    _logging_config_available = False
    logging.debug("logging_config.py nie jest jeszcze dostępny")

# ============== SINGLETON KONFIGURACJI ==============

# Globalna instancja konfiguracji
config = get_config()

# ============== FUNKCJE POMOCNICZE ==============

def get_database_config(db_name: str) -> Dict[str, Any]:
    """
    Zwraca konfigurację dla konkretnej bazy danych
    
    Args:
        db_name: Nazwa bazy ('postgresql', 'qdrant', 'neo4j', 'redis')
        
    Returns:
        Dict z konfiguracją bazy
    """
    configs = {
        'postgresql': config.postgresql,
        'qdrant': config.qdrant,
        'neo4j': config.neo4j,
        'redis': config.redis
    }
    
    if db_name not in configs:
        raise ValueError(f"Nieznana baza danych: {db_name}")
    
    return configs[db_name]

def get_database_schemas() -> Dict[str, Any]:
    """
    Zwraca wszystkie schematy baz danych
    
    Returns:
        Dict ze schematami dla każdej bazy
    """
    return {
        'postgresql': {
            'schemas': POSTGRES_SCHEMAS,
            'views': POSTGRES_VIEWS,
            'functions': POSTGRES_FUNCTIONS
        },
        'qdrant': {
            'collections': QDRANT_COLLECTIONS
        },
        'neo4j': {
            'nodes': NEO4J_NODE_TYPES,
            'relationships': NEO4J_RELATIONSHIP_TYPES,
            'queries': NEO4J_ANALYSIS_QUERIES
        }
    }

def get_task_config(task_id: str) -> Dict[str, Any]:
    """
    Zwraca konfigurację dla konkretnego zadania
    
    Args:
        task_id: ID zadania (np. 'S01', 'S25')
        
    Returns:
        Dict z konfiguracją zadania
    """
    databases = config.get_task_databases(task_id)
    
    task_config = {
        'task_id': task_id,
        'required_databases': databases,
        'has_postgresql': 'postgresql' in databases,
        'has_qdrant': 'qdrant' in databases,
        'has_neo4j': 'neo4j' in databases
    }
    
    # Dodaj prompty jeśli dostępne
    if _llm_config_available:
        task_config['prompts'] = TASK_PROMPTS.get(task_id, {})
    
    return task_config

def validate_environment() -> Dict[str, bool]:
    """
    Sprawdza czy wszystkie wymagane komponenty są skonfigurowane
    
    Returns:
        Dict z wynikami walidacji
    """
    validation_results = {
        'openai_configured': bool(config.openai.api_key),
        'postgresql_configured': bool(config.postgresql.password),
        'qdrant_configured': True,  # Qdrant nie wymaga hasła
        'neo4j_configured': bool(config.neo4j.password),
        'redis_configured': True,  # Redis opcjonalnie wymaga hasła
        'directories_exist': all(d.exists() for d in [DATA_DIR, LOGS_DIR]),
        'llm_config_available': _llm_config_available,
        'logging_config_available': _logging_config_available
    }
    
    validation_results['all_configured'] = all([
        validation_results['openai_configured'],
        validation_results['postgresql_configured'],
        validation_results['neo4j_configured'],
        validation_results['directories_exist']
    ])
    
    return validation_results

def get_model_config(model_type: str = "chat") -> Dict[str, Any]:
    """
    Zwraca konfigurację dla modelu AI
    
    Args:
        model_type: Typ modelu ('chat', 'embedding')
        
    Returns:
        Dict z konfiguracją modelu
    """
    if model_type == "chat":
        return {
            'model': config.openai.chat_model,
            'temperature': config.openai.temperature,
            'max_tokens': config.openai.max_tokens,
            'api_key': config.openai.api_key
        }
    elif model_type == "embedding":
        return {
            'model': config.openai.embedding_model,
            'api_key': config.openai.api_key,
            'vector_size': config.qdrant.vector_size
        }
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

def get_processing_config() -> DocumentProcessingConfig:
    """
    Zwraca konfigurację przetwarzania dokumentów
    
    Returns:
        DocumentProcessingConfig
    """
    return config.document_processing

# ============== KLASY KONFIGURACYJNE ==============

class ConfigManager:
    """
    Manager konfiguracji z dodatkowymi funkcjami
    """
    
    def __init__(self):
        self.config = config
        self._cache = {}
    
    def reload(self):
        """Przeładowuje konfigurację"""
        global config
        config = get_config()
        self.config = config
        self._cache.clear()
        logging.info("Konfiguracja została przeładowana")
    
    def get_connection_string(self, db_name: str, async_mode: bool = False) -> str:
        """Zwraca connection string dla bazy"""
        if db_name == 'postgresql':
            return (self.config.postgresql.async_connection_string 
                   if async_mode else self.config.postgresql.connection_string)
        elif db_name == 'qdrant':
            return self.config.qdrant.url
        elif db_name == 'neo4j':
            return self.config.neo4j.uri
        elif db_name == 'redis':
            return self.config.redis.url
        else:
            raise ValueError(f"Nieznana baza: {db_name}")
    
    def get_rate_limits(self) -> Dict[str, int]:
        """Zwraca limity rate limiting"""
        return {
            'openai_rpm': self.config.openai.requests_per_minute,
            'openai_tpm': self.config.openai.tokens_per_minute,
            'system_rpm': self.config.system.max_requests_per_minute
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Sprawdza czy funkcja jest włączona"""
        feature_flags = {
            'metrics': self.config.system.enable_metrics,
            'rate_limiting': self.config.system.enable_rate_limiting,
            'async': self.config.system.async_enabled,
            'spacy': self.config.entity_extraction.use_spacy,
            'extract_relations': self.config.entity_extraction.extract_relations
        }
        return feature_flags.get(feature, False)

# ============== STAŁE KONFIGURACYJNE ==============

# Wersja API
API_VERSION = "1.0.0"

# Domyślne timeouty (sekundy)
DEFAULT_TIMEOUTS = {
    'openai': 30,
    'database': 10,
    'web_scraping': 20,
    'agent': config.agents.agent_timeout
}

# Limity systemowe
SYSTEM_LIMITS = {
    'max_document_size_mb': config.document_processing.max_document_size_mb,
    'max_chunk_tokens': config.document_processing.max_chunk_tokens,
    'max_workers': config.system.max_workers,
    'max_agent_iterations': config.agents.max_agent_iterations
}

# ============== EKSPORT ==============

# Główne eksporty
__all__ = [
    # Klasy konfiguracji
    'Config',
    'ConfigManager',
    'OpenAIConfig',
    'PostgreSQLConfig', 
    'QdrantConfig',
    'Neo4jConfig',
    'RedisConfig',
    'DocumentProcessingConfig',
    'EntityExtractionConfig',
    'AgentConfig',
    'LoggingConfig',
    'SystemConfig',
    
    # Instancje i funkcje
    'config',
    'get_config',
    'get_database_config',
    'get_database_schemas',
    'get_task_config',
    'validate_environment',
    'get_model_config',
    'get_processing_config',
    
    # Schematy baz
    'POSTGRES_SCHEMAS',
    'POSTGRES_VIEWS',
    'POSTGRES_FUNCTIONS',
    'QDRANT_COLLECTIONS',
    'NEO4J_NODE_TYPES',
    'NEO4J_RELATIONSHIP_TYPES',
    'NEO4J_ANALYSIS_QUERIES',
    'DatabaseInitializer',
    'SQL_QUERIES',
    'DATABASE_MIGRATIONS',
    
    # Ścieżki
    'BASE_DIR',
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'CACHE_DIR',
    'LOGS_DIR',
    
    # Mapowania
    'TASK_DATABASE_REQUIREMENTS',
    
    # Stałe
    'API_VERSION',
    'DEFAULT_TIMEOUTS',
    'SYSTEM_LIMITS'
]

# Dodaj eksporty LLM jeśli dostępne
if _llm_config_available:
    __all__.extend([
        'LLMPrompts',
        'SYSTEM_PROMPTS',
        'AGENT_PROMPTS',
        'TASK_PROMPTS',
        'get_prompt'
    ])

# Dodaj eksporty logowania jeśli dostępne
if _logging_config_available:
    __all__.extend([
        'setup_logging',
        'get_logger'
    ])

# ============== INICJALIZACJA ==============

# Log informacji o konfiguracji przy imporcie
logger = logging.getLogger(__name__)
validation = validate_environment()

if not validation['all_configured']:
    missing = [k for k, v in validation.items() if not v and k != 'all_configured']
    logger.warning(f"Brakujące komponenty konfiguracji: {missing}")
else:
    logger.info("Konfiguracja załadowana pomyślnie")

# Utwórz instancję managera
config_manager = ConfigManager()
