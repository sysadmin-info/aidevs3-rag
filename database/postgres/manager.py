# database/postgres/manager.py
"""
Manager PostgreSQL - zarządzanie połączeniami, zapytaniami i transakcjami
Implementuje wszystkie operacje na bazie danych dla systemu RAG
"""

import asyncio
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager
import hashlib
from dataclasses import dataclass, asdict

import psycopg2
from psycopg2.extras import RealDictCursor, Json, execute_batch
from psycopg2.pool import ThreadedConnectionPool
import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_config, get_logger, log_execution, log_database_query
from config.database_config import POSTGRES_SCHEMAS, SQL_QUERIES, DatabaseInitializer

# Konfiguracja i logger
config = get_config()
logger = get_logger(__name__)

# ============== MODELE DANYCH ==============

@dataclass
class Document:
    """Model dokumentu"""
    id: Optional[str] = None
    task_id: str = None
    source_path: str = None
    doc_type: str = None
    content: str = None
    cleaned_content: Optional[str] = None
    metadata: Dict[str, Any] = None
    file_hash: Optional[str] = None
    processing_status: str = "pending"
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
        if self.file_hash is None and self.content:
            self.file_hash = hashlib.sha256(self.content.encode()).hexdigest()

@dataclass
class DocumentChunk:
    """Model fragmentu dokumentu"""
    id: Optional[str] = None
    document_id: str = None
    chunk_index: int = None
    content: str = None
    tokens: int = None
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Entity:
    """Model encji"""
    id: Optional[str] = None
    name: str = None
    type: str = None
    normalized_name: str = None
    properties: Dict[str, Any] = None
    confidence_score: float = 1.0
    occurrence_count: int = 1
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.properties is None:
            self.properties = {}
        if self.normalized_name is None and self.name:
            self.normalized_name = self.name.lower().strip()

# ============== MANAGER POŁĄCZEŃ ==============

class PostgresConnectionManager:
    """Zarządza połączeniami z PostgreSQL"""
    
    def __init__(self):
        self.config = config.postgresql
        self._sync_pool: Optional[ThreadedConnectionPool] = None
        self._async_pool: Optional[asyncpg.Pool] = None
        self._is_initialized = False
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Zwraca parametry połączenia"""
        return {
            'host': self.config.host,
            'port': self.config.port,
            'database': self.config.database,
            'user': self.config.user,
            'password': self.config.password
        }
    
    def initialize_sync_pool(self):
        """Inicjalizuje synchroniczny pool połączeń"""
        if self._sync_pool is None:
            self._sync_pool = ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                **self._get_connection_params()
            )
            logger.info("Initialized PostgreSQL sync connection pool")
    
    async def initialize_async_pool(self):
        """Inicjalizuje asynchroniczny pool połączeń"""
        if self._async_pool is None:
            self._async_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections
            )
            logger.info("Initialized PostgreSQL async connection pool")
    
    @contextmanager
    def get_sync_connection(self):
        """Context manager dla synchronicznego połączenia"""
        if self._sync_pool is None:
            self.initialize_sync_pool()
        
        conn = self._sync_pool.getconn()
        try:
            yield conn
        finally:
            self._sync_pool.putconn(conn)
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Context manager dla asynchronicznego połączenia"""
        if self._async_pool is None:
            await self.initialize_async_pool()
        
        async with self._async_pool.acquire() as conn:
            yield conn
    
    def close(self):
        """Zamyka wszystkie połączenia"""
        if self._sync_pool:
            self._sync_pool.closeall()
            self._sync_pool = None
        
        if self._async_pool:
            asyncio.create_task(self._async_pool.close())
            self._async_pool = None
        
        logger.info("Closed all PostgreSQL connections")

# ============== GŁÓWNY MANAGER ==============

class PostgreSQLManager:
    """Główny manager operacji PostgreSQL"""
    
    def __init__(self):
        self.connection_manager = PostgresConnectionManager()
        self._is_initialized = False
    
    # ===== INICJALIZACJA =====
    
    @log_execution(measure_time=True)
    def initialize_database(self):
        """Inicjalizuje schemat bazy danych"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                # Wykonaj wszystkie schematy
                for table_name, schema_sql in POSTGRES_SCHEMAS.items():
                    try:
                        cursor.execute(schema_sql)
                        logger.info(f"Created/verified table: {table_name}")
                    except Exception as e:
                        logger.error(f"Error creating table {table_name}: {e}")
                        raise
                
                conn.commit()
                
        self._is_initialized = True
        logger.info("Database schema initialized successfully")
    
    def check_connection(self) -> bool:
        """Sprawdza połączenie z bazą"""
        try:
            with self.connection_manager.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return cursor.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    # ===== OPERACJE NA DOKUMENTACH =====
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def save_document(self, document: Document) -> str:
        """Zapisuje dokument do bazy"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Sprawdź czy dokument już istnieje (po hashu)
                if document.file_hash:
                    cursor.execute(
                        "SELECT id FROM documents WHERE file_hash = %s",
                        (document.file_hash,)
                    )
                    existing = cursor.fetchone()
                    if existing:
                        logger.info(f"Document already exists: {existing['id']}")
                        return existing['id']
                
                # Wstaw nowy dokument
                cursor.execute("""
                    INSERT INTO documents (
                        id, task_id, source_path, doc_type, content, 
                        cleaned_content, metadata, file_hash, processing_status,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        cleaned_content = EXCLUDED.cleaned_content,
                        metadata = EXCLUDED.metadata,
                        processing_status = EXCLUDED.processing_status,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    document.id,
                    document.task_id,
                    document.source_path,
                    document.doc_type,
                    document.content,
                    document.cleaned_content,
                    Json(document.metadata),
                    document.file_hash,
                    document.processing_status,
                    datetime.now()
                ))
                
                result = cursor.fetchone()
                conn.commit()
                
                log_database_query(
                    database="postgresql",
                    query="INSERT INTO documents",
                    duration=0.01,
                    rows_affected=1
                )
                
                logger.info(f"Saved document: {result['id']}")
                return result['id']
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Pobiera dokument po ID"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM documents WHERE id = %s",
                    (document_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return Document(**row)
                return None
    
    def get_documents_by_task(self, task_id: str, status: Optional[str] = None) -> List[Document]:
        """Pobiera dokumenty dla zadania"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if status:
                    cursor.execute(
                        "SELECT * FROM documents WHERE task_id = %s AND processing_status = %s ORDER BY created_at DESC",
                        (task_id, status)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM documents WHERE task_id = %s ORDER BY created_at DESC",
                        (task_id,)
                    )
                
                rows = cursor.fetchall()
                return [Document(**row) for row in rows]
    
    def update_document_status(
        self, 
        document_id: str, 
        status: str, 
        error_message: Optional[str] = None
    ):
        """Aktualizuje status dokumentu"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE documents 
                    SET processing_status = %s,
                        error_message = %s,
                        processed_at = CASE WHEN %s IN ('completed', 'error') THEN CURRENT_TIMESTAMP ELSE processed_at END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, error_message, status, document_id))
                
                conn.commit()
                logger.info(f"Updated document {document_id} status to {status}")
    
    # ===== OPERACJE NA CHUNKACH =====
    
    def save_document_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Zapisuje chunki dokumentu"""
        if not chunks:
            return []
        
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                # Przygotuj dane do batch insert
                values = [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.tokens,
                        chunk.embedding_id,
                        Json(chunk.metadata)
                    )
                    for chunk in chunks
                ]
                
                # Batch insert
                execute_batch(
                    cursor,
                    """
                    INSERT INTO document_chunks (
                        id, document_id, chunk_index, content, tokens, 
                        embedding_id, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                        content = EXCLUDED.content,
                        tokens = EXCLUDED.tokens,
                        embedding_id = EXCLUDED.embedding_id,
                        metadata = EXCLUDED.metadata
                    """,
                    values,
                    page_size=100
                )
                
                conn.commit()
                
                chunk_ids = [chunk.id for chunk in chunks]
                logger.info(f"Saved {len(chunks)} document chunks")
                return chunk_ids
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Pobiera chunki dokumentu"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM document_chunks WHERE document_id = %s ORDER BY chunk_index",
                    (document_id,)
                )
                
                rows = cursor.fetchall()
                return [DocumentChunk(**row) for row in rows]
    
    # ===== OPERACJE NA ENCJACH =====
    
    def save_entity(self, entity: Entity) -> str:
        """Zapisuje encję"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Sprawdź czy encja już istnieje
                cursor.execute(
                    "SELECT id, occurrence_count FROM entities WHERE normalized_name = %s AND type = %s",
                    (entity.normalized_name, entity.type)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Aktualizuj istniejącą encję
                    cursor.execute("""
                        UPDATE entities 
                        SET occurrence_count = occurrence_count + 1,
                            last_seen_at = CURRENT_TIMESTAMP,
                            properties = %s
                        WHERE id = %s
                        RETURNING id
                    """, (Json(entity.properties), existing['id']))
                else:
                    # Wstaw nową encję
                    cursor.execute("""
                        INSERT INTO entities (
                            id, name, type, normalized_name, properties,
                            confidence_score, occurrence_count
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        entity.id,
                        entity.name,
                        entity.type,
                        entity.normalized_name,
                        Json(entity.properties),
                        entity.confidence_score,
                        entity.occurrence_count
                    ))
                
                result = cursor.fetchone()
                conn.commit()
                
                return result['id']
    
    def save_document_entities(
        self, 
        document_id: str, 
        entities: List[Tuple[str, Dict[str, Any]]]
    ):
        """Zapisuje powiązania dokument-encje"""
        if not entities:
            return
        
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                values = [
                    (
                        str(uuid.uuid4()),
                        document_id,
                        entity_id,
                        details.get('chunk_id'),
                        details.get('position_start'),
                        details.get('position_end'),
                        details.get('context'),
                        details.get('confidence_score', 1.0)
                    )
                    for entity_id, details in entities
                ]
                
                execute_batch(
                    cursor,
                    """
                    INSERT INTO document_entities (
                        id, document_id, entity_id, chunk_id,
                        position_start, position_end, context, confidence_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, entity_id, position_start) DO NOTHING
                    """,
                    values,
                    page_size=100
                )
                
                conn.commit()
                logger.info(f"Saved {len(entities)} document-entity relations")
    
    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Entity]:
        """Pobiera encje po typie"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM entities WHERE type = %s ORDER BY occurrence_count DESC LIMIT %s",
                    (entity_type, limit)
                )
                
                rows = cursor.fetchall()
                return [Entity(**row) for row in rows]
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Wyszukuje encje"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT * FROM entities 
                    WHERE normalized_name ILIKE %s OR name ILIKE %s
                    ORDER BY occurrence_count DESC
                    LIMIT %s
                    """,
                    (f"%{query}%", f"%{query}%", limit)
                )
                
                rows = cursor.fetchall()
                return [Entity(**row) for row in rows]
    
    # ===== OPERACJE NA RELACJACH =====
    
    def save_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence_score: float = 1.0
    ) -> str:
        """Zapisuje relację między encjami"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Sprawdź czy relacja już istnieje
                cursor.execute(
                    """
                    SELECT id, occurrence_count FROM relationships 
                    WHERE source_entity_id = %s AND target_entity_id = %s 
                    AND relationship_type = %s
                    """,
                    (source_entity_id, target_entity_id, relationship_type)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Aktualizuj istniejącą relację
                    cursor.execute("""
                        UPDATE relationships 
                        SET occurrence_count = occurrence_count + 1,
                            last_seen_at = CURRENT_TIMESTAMP,
                            properties = %s,
                            confidence_score = GREATEST(confidence_score, %s)
                        WHERE id = %s
                        RETURNING id
                    """, (Json(properties or {}), confidence_score, existing['id']))
                else:
                    # Wstaw nową relację
                    cursor.execute("""
                        INSERT INTO relationships (
                            id, source_entity_id, target_entity_id,
                            relationship_type, properties, confidence_score
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        str(uuid.uuid4()),
                        source_entity_id,
                        target_entity_id,
                        relationship_type,
                        Json(properties or {}),
                        confidence_score
                    ))
                
                result = cursor.fetchone()
                conn.commit()
                
                return result['id']
    
    # ===== OPERACJE NA PYTANIACH I ODPOWIEDZIACH =====
    
    def save_question_answer(
        self,
        question: str,
        answer: str,
        question_type: Optional[str] = None,
        sources: Optional[List[str]] = None,
        confidence_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
        used_agents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Zapisuje pytanie i odpowiedź"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    INSERT INTO questions_answers (
                        id, question, answer, question_type, sources,
                        confidence_score, processing_time_ms, used_agents, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    str(uuid.uuid4()),
                    question,
                    answer,
                    question_type,
                    Json(sources or []),
                    confidence_score,
                    processing_time_ms,
                    Json(used_agents or []),
                    Json(metadata or {})
                ))
                
                result = cursor.fetchone()
                conn.commit()
                
                return result['id']
    
    # ===== OPERACJE CACHE =====
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Pobiera wartość z cache"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT value FROM cache_entries 
                    WHERE key = %s AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    """,
                    (key,)
                )
                
                row = cursor.fetchone()
                if row:
                    # Aktualizuj licznik dostępów
                    cursor.execute(
                        """
                        UPDATE cache_entries 
                        SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                        WHERE key = %s
                        """,
                        (key,)
                    )
                    conn.commit()
                    
                    return row['value']
                return None
    
    def cache_set(
        self, 
        key: str, 
        value: Any, 
        category: str = "general",
        ttl_seconds: Optional[int] = None
    ):
        """Zapisuje wartość do cache"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                expires_at = None
                if ttl_seconds:
                    cursor.execute(
                        "SELECT CURRENT_TIMESTAMP + INTERVAL '%s seconds'",
                        (ttl_seconds,)
                    )
                    expires_at = cursor.fetchone()[0]
                
                cursor.execute("""
                    INSERT INTO cache_entries (key, value, category, expires_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        category = EXCLUDED.category,
                        expires_at = EXCLUDED.expires_at,
                        accessed_at = CURRENT_TIMESTAMP,
                        access_count = cache_entries.access_count + 1
                """, (key, Json(value), category, expires_at))
                
                conn.commit()
    
    def cache_delete(self, key: str):
        """Usuwa wartość z cache"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM cache_entries WHERE key = %s", (key,))
                conn.commit()
    
    def cache_cleanup(self):
        """Czyści wygasłe wpisy cache"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM cache_entries WHERE expires_at < CURRENT_TIMESTAMP")
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} expired cache entries")
    
    # ===== STATYSTYKI I ANALIZA =====
    
    def get_statistics(self) -> Dict[str, Any]:
        """Pobiera statystyki systemu"""
        with self.connection_manager.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                stats = {}
                
                # Statystyki dokumentów
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents,
                        COUNT(CASE WHEN processing_status = 'error' THEN 1 END) as error_documents,
                        COUNT(DISTINCT task_id) as total_tasks
                    FROM documents
                """)
                stats['documents'] = cursor.fetchone()
                
                # Statystyki encji
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_entities,
                        COUNT(DISTINCT type) as entity_types,
                        SUM(occurrence_count) as total_occurrences
                    FROM entities
                """)
                stats['entities'] = cursor.fetchone()
                
                # Statystyki relacji
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_relationships,
                        COUNT(DISTINCT relationship_type) as relationship_types
                    FROM relationships
                """)
                stats['relationships'] = cursor.fetchone()
                
                # Statystyki pytań
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_questions,
                        AVG(confidence_score) as avg_confidence,
                        AVG(processing_time_ms) as avg_processing_time
                    FROM questions_answers
                """)
                stats['questions'] = cursor.fetchone()
                
                return stats
    
    # ===== OPERACJE ASYNC =====
    
    async def async_get_document(self, document_id: str) -> Optional[Document]:
        """Asynchroniczne pobranie dokumentu"""
        async with self.connection_manager.get_async_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1",
                document_id
            )
            
            if row:
                return Document(**dict(row))
            return None
    
    async def async_search_documents(
        self,
        query: str,
        task_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Document]:
        """Asynchroniczne wyszukiwanie dokumentów"""
        async with self.connection_manager.get_async_connection() as conn:
            if task_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM documents 
                    WHERE task_id = $1 AND 
                    (content ILIKE $2 OR cleaned_content ILIKE $2)
                    LIMIT $3
                    """,
                    task_id, f"%{query}%", limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM documents 
                    WHERE content ILIKE $1 OR cleaned_content ILIKE $1
                    LIMIT $2
                    """,
                    f"%{query}%", limit
                )
            
            return [Document(**dict(row)) for row in rows]
    
    # ===== TRANSAKCJE =====
    
    @contextmanager
    def transaction(self):
        """Context manager dla transakcji"""
        with self.connection_manager.get_sync_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    # ===== CLEANUP =====
    
    def close(self):
        """Zamyka połączenia"""
        self.connection_manager.close()

# ============== SINGLETON ==============

_manager_instance: Optional[PostgreSQLManager] = None

def get_postgres_manager() -> PostgreSQLManager:
    """Zwraca singleton instancji managera"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = PostgreSQLManager()
    return _manager_instance

# ============== EKSPORT ==============

__all__ = [
    'PostgreSQLManager',
    'get_postgres_manager',
    'Document',
    'DocumentChunk',
    'Entity'
]
