# database/postgres/migrations.py
"""
System migracji dla PostgreSQL
Zarządza wersjami schematu bazy danych i umożliwia bezpieczne aktualizacje
"""

import os
import hashlib
import importlib.util
from typing import List, Dict, Tuple, Optional, Callable, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import json

import psycopg2
from psycopg2.extras import RealDictCursor

from config import get_config, get_logger, log_execution
from config.database_config import DATABASE_MIGRATIONS, DatabaseInitializer

# Konfiguracja i logger
config = get_config()
logger = get_logger(__name__)

# ============== MODELE ==============

@dataclass
class Migration:
    """Model migracji"""
    id: str
    name: str
    description: str
    up_sql: str
    down_sql: Optional[str] = None
    checksum: Optional[str] = None
    dependencies: Optional[List[str]] = None
    
    def __post_init__(self):
        # Oblicz checksum jeśli nie podano
        if self.checksum is None:
            content = f"{self.up_sql}{self.down_sql or ''}"
            self.checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

@dataclass
class MigrationHistory:
    """Rekord historii migracji"""
    id: int
    migration_id: str
    name: str
    checksum: str
    applied_at: datetime
    execution_time_ms: int
    success: bool
    error_message: Optional[str] = None
    rolled_back: bool = False

# ============== SCHEMAT TABELI MIGRACJI ==============

MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    checksum VARCHAR(32) NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    rolled_back BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_migrations_applied_at ON schema_migrations(applied_at DESC);
CREATE INDEX IF NOT EXISTS idx_migrations_success ON schema_migrations(success);
"""

# ============== DEFINICJE MIGRACJI ==============

MIGRATIONS: List[Migration] = [
    Migration(
        id="001_initial_schema",
        name="Initial database schema",
        description="Creates all base tables for the RAG system",
        up_sql=DatabaseInitializer.get_postgres_init_sql(),
        down_sql="""
            DROP TABLE IF EXISTS questions_answers CASCADE;
            DROP TABLE IF EXISTS cache_entries CASCADE;
            DROP TABLE IF EXISTS processing_history CASCADE;
            DROP TABLE IF EXISTS relationships CASCADE;
            DROP TABLE IF EXISTS document_entities CASCADE;
            DROP TABLE IF EXISTS entities CASCADE;
            DROP TABLE IF EXISTS document_chunks CASCADE;
            DROP TABLE IF EXISTS documents CASCADE;
            DROP VIEW IF EXISTS document_stats CASCADE;
            DROP VIEW IF EXISTS entity_stats CASCADE;
            DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;
            DROP FUNCTION IF EXISTS normalize_text CASCADE;
        """
    ),
    
    Migration(
        id="002_add_embeddings_metadata",
        name="Add embeddings metadata",
        description="Adds embedding model and dimensions to chunks",
        up_sql="""
            ALTER TABLE document_chunks 
            ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100),
            ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
            
            COMMENT ON COLUMN document_chunks.embedding_model IS 'Model used to generate embeddings';
            COMMENT ON COLUMN document_chunks.embedding_dimensions IS 'Dimensions of the embedding vector';
        """,
        down_sql="""
            ALTER TABLE document_chunks 
            DROP COLUMN IF EXISTS embedding_model,
            DROP COLUMN IF EXISTS embedding_dimensions;
        """
    ),
    
    Migration(
        id="003_add_task_metadata",
        name="Add task metadata",
        description="Adds task-specific metadata to documents",
        up_sql="""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS task_metadata JSONB DEFAULT '{}';
            
            CREATE INDEX IF NOT EXISTS idx_documents_task_metadata 
            ON documents USING GIN (task_metadata);
            
            COMMENT ON COLUMN documents.task_metadata IS 'Task-specific metadata and configuration';
        """,
        down_sql="""
            DROP INDEX IF EXISTS idx_documents_task_metadata;
            ALTER TABLE documents DROP COLUMN IF EXISTS task_metadata;
        """
    ),
    
    Migration(
        id="004_add_vector_search_index",
        name="Add vector search optimization",
        description="Adds indexes for vector search optimization",
        up_sql="""
            -- Index for faster chunk retrieval by embedding_id
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_lookup 
            ON document_chunks(embedding_id) 
            WHERE embedding_id IS NOT NULL;
            
            -- Composite index for document-chunk queries
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_order 
            ON document_chunks(document_id, chunk_index);
            
            -- Add search_vector column for full-text search
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS search_vector tsvector;
            
            -- Update search vector for existing documents
            UPDATE documents 
            SET search_vector = to_tsvector('polish', COALESCE(cleaned_content, content))
            WHERE search_vector IS NULL;
            
            -- Create GIN index for full-text search
            CREATE INDEX IF NOT EXISTS idx_documents_search 
            ON documents USING GIN (search_vector);
            
            -- Trigger to maintain search vector
            CREATE OR REPLACE FUNCTION update_search_vector()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.search_vector := to_tsvector('polish', COALESCE(NEW.cleaned_content, NEW.content));
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            CREATE TRIGGER trigger_update_search_vector 
            BEFORE INSERT OR UPDATE OF content, cleaned_content 
            ON documents
            FOR EACH ROW EXECUTE FUNCTION update_search_vector();
        """,
        down_sql="""
            DROP TRIGGER IF EXISTS trigger_update_search_vector ON documents;
            DROP FUNCTION IF EXISTS update_search_vector();
            DROP INDEX IF EXISTS idx_documents_search;
            ALTER TABLE documents DROP COLUMN IF EXISTS search_vector;
            DROP INDEX IF EXISTS idx_chunks_doc_order;
            DROP INDEX IF EXISTS idx_chunks_embedding_lookup;
        """
    ),
    
    Migration(
        id="005_add_graph_sync_tables",
        name="Add graph synchronization tables",
        description="Tables for tracking Neo4j synchronization",
        up_sql="""
            CREATE TABLE IF NOT EXISTS graph_sync_queue (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id UUID,
                document_id UUID,
                operation VARCHAR(20) NOT NULL CHECK (operation IN ('create', 'update', 'delete')),
                entity_type VARCHAR(50),
                data JSONB,
                status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP WITH TIME ZONE
            );
            
            CREATE INDEX idx_sync_queue_status ON graph_sync_queue(status, created_at);
            CREATE INDEX idx_sync_queue_entity ON graph_sync_queue(entity_id);
            CREATE INDEX idx_sync_queue_document ON graph_sync_queue(document_id);
            
            -- Track last sync times
            CREATE TABLE IF NOT EXISTS sync_checkpoints (
                sync_type VARCHAR(50) PRIMARY KEY,
                last_sync_at TIMESTAMP WITH TIME ZONE,
                last_entity_id UUID,
                last_document_id UUID,
                metadata JSONB DEFAULT '{}'
            );
        """,
        down_sql="""
            DROP TABLE IF EXISTS sync_checkpoints;
            DROP TABLE IF EXISTS graph_sync_queue;
        """
    ),
    
    Migration(
        id="006_add_performance_indexes",
        name="Add performance optimization indexes",
        description="Additional indexes for query performance",
        up_sql="""
            -- Covering index for document listing queries
            CREATE INDEX IF NOT EXISTS idx_documents_listing 
            ON documents(task_id, created_at DESC) 
            INCLUDE (doc_type, processing_status);
            
            -- Index for entity occurrence queries
            CREATE INDEX IF NOT EXISTS idx_entities_occurrence 
            ON entities(occurrence_count DESC, type);
            
            -- Partial index for pending documents
            CREATE INDEX IF NOT EXISTS idx_documents_pending 
            ON documents(created_at) 
            WHERE processing_status = 'pending';
            
            -- Index for relationship queries
            CREATE INDEX IF NOT EXISTS idx_relationships_composite 
            ON relationships(source_entity_id, relationship_type, target_entity_id);
            
            -- Optimize cache queries
            CREATE INDEX IF NOT EXISTS idx_cache_category_expires 
            ON cache_entries(category, expires_at) 
            WHERE expires_at IS NOT NULL;
        """,
        down_sql="""
            DROP INDEX IF EXISTS idx_cache_category_expires;
            DROP INDEX IF EXISTS idx_relationships_composite;
            DROP INDEX IF EXISTS idx_documents_pending;
            DROP INDEX IF EXISTS idx_entities_occurrence;
            DROP INDEX IF EXISTS idx_documents_listing;
        """
    ),
    
    Migration(
        id="007_add_audit_fields",
        name="Add audit fields",
        description="Add created_by and updated_by fields for audit trail",
        up_sql="""
            -- Add audit fields to main tables
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS created_by VARCHAR(100),
            ADD COLUMN IF NOT EXISTS updated_by VARCHAR(100);
            
            ALTER TABLE entities 
            ADD COLUMN IF NOT EXISTS created_by VARCHAR(100),
            ADD COLUMN IF NOT EXISTS updated_by VARCHAR(100);
            
            ALTER TABLE relationships 
            ADD COLUMN IF NOT EXISTS created_by VARCHAR(100),
            ADD COLUMN IF NOT EXISTS updated_by VARCHAR(100);
            
            -- Audit log table
            CREATE TABLE IF NOT EXISTS audit_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                table_name VARCHAR(50) NOT NULL,
                record_id UUID NOT NULL,
                action VARCHAR(20) NOT NULL CHECK (action IN ('insert', 'update', 'delete')),
                old_data JSONB,
                new_data JSONB,
                changed_by VARCHAR(100),
                changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'
            );
            
            CREATE INDEX idx_audit_log_table_record ON audit_log(table_name, record_id);
            CREATE INDEX idx_audit_log_changed_at ON audit_log(changed_at DESC);
        """,
        down_sql="""
            DROP TABLE IF EXISTS audit_log;
            ALTER TABLE relationships DROP COLUMN IF EXISTS created_by, DROP COLUMN IF EXISTS updated_by;
            ALTER TABLE entities DROP COLUMN IF EXISTS created_by, DROP COLUMN IF EXISTS updated_by;
            ALTER TABLE documents DROP COLUMN IF EXISTS created_by, DROP COLUMN IF EXISTS updated_by;
        """
    ),
    
    Migration(
        id="008_add_task_specific_tables",
        name="Add task-specific tables",
        description="Tables for specific AI Devs tasks",
        up_sql="""
            -- Table for S20_dane specific data
            CREATE TABLE IF NOT EXISTS task_s20_data (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                data_type VARCHAR(50) NOT NULL,
                extracted_data JSONB NOT NULL,
                confidence_score FLOAT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX idx_s20_data_document ON task_s20_data(document_id);
            CREATE INDEX idx_s20_data_type ON task_s20_data(data_type);
            
            -- Table for S25 questions
            CREATE TABLE IF NOT EXISTS task_s25_questions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                question_number INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                expected_answer_type VARCHAR(50),
                related_tasks TEXT[],
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(question_number)
            );
            
            -- Table for S25 answers
            CREATE TABLE IF NOT EXISTS task_s25_answers (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                question_id UUID REFERENCES task_s25_questions(id) ON DELETE CASCADE,
                answer_text TEXT NOT NULL,
                confidence_score FLOAT,
                sources JSONB NOT NULL,
                evidence JSONB,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX idx_s25_answers_question ON task_s25_answers(question_id);
        """,
        down_sql="""
            DROP TABLE IF EXISTS task_s25_answers;
            DROP TABLE IF EXISTS task_s25_questions;
            DROP TABLE IF EXISTS task_s20_data;
        """
    )
]

# ============== MIGRATION MANAGER ==============

class MigrationManager:
    """Zarządza migracjami bazy danych"""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        if connection_params:
            self.connection_params = connection_params
        else:
            self.connection_params = {
                'host': config.postgresql.host,
                'port': config.postgresql.port,
                'database': config.postgresql.database,
                'user': config.postgresql.user,
                'password': config.postgresql.password
            }
        
        self._ensure_migrations_table()
    
    def _get_connection(self):
        """Tworzy połączenie z bazą"""
        return psycopg2.connect(**self.connection_params)
    
    def _ensure_migrations_table(self):
        """Upewnia się, że tabela migracji istnieje"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(MIGRATIONS_TABLE_SQL)
                conn.commit()
                logger.info("Migrations table ready")
    
    def _get_applied_migrations(self) -> Dict[str, MigrationHistory]:
        """Pobiera listę zastosowanych migracji"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM schema_migrations 
                    WHERE success = TRUE AND rolled_back = FALSE
                    ORDER BY applied_at
                """)
                
                rows = cursor.fetchall()
                return {
                    row['migration_id']: MigrationHistory(**row) 
                    for row in rows
                }
    
    def _record_migration(
        self,
        migration: Migration,
        success: bool,
        execution_time_ms: int,
        error_message: Optional[str] = None
    ):
        """Zapisuje wykonanie migracji"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO schema_migrations 
                    (migration_id, name, checksum, execution_time_ms, success, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    migration.id,
                    migration.name,
                    migration.checksum,
                    execution_time_ms,
                    success,
                    error_message
                ))
                conn.commit()
    
    def _mark_as_rolled_back(self, migration_id: str):
        """Oznacza migrację jako wycofaną"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE schema_migrations 
                    SET rolled_back = TRUE 
                    WHERE migration_id = %s
                """, (migration_id,))
                conn.commit()
    
    @log_execution(measure_time=True)
    def get_pending_migrations(self) -> List[Migration]:
        """Zwraca listę migracji do wykonania"""
        applied = self._get_applied_migrations()
        pending = []
        
        for migration in MIGRATIONS:
            if migration.id not in applied:
                pending.append(migration)
            elif applied[migration.id].checksum != migration.checksum:
                logger.warning(
                    f"Migration {migration.id} has been modified! "
                    f"Applied checksum: {applied[migration.id].checksum}, "
                    f"Current checksum: {migration.checksum}"
                )
        
        return pending
    
    @log_execution(measure_time=True)
    def apply_migration(self, migration: Migration) -> bool:
        """Wykonuje pojedynczą migrację"""
        logger.info(f"Applying migration: {migration.id} - {migration.name}")
        
        start_time = datetime.now()
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Wykonaj SQL migracji
                    cursor.execute(migration.up_sql)
                    conn.commit()
            
            # Oblicz czas wykonania
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Zapisz sukces
            self._record_migration(migration, True, execution_time_ms)
            logger.info(
                f"Migration {migration.id} applied successfully in {execution_time_ms}ms"
            )
            return True
            
        except Exception as e:
            # Oblicz czas wykonania
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Zapisz błąd
            self._record_migration(migration, False, execution_time_ms, str(e))
            logger.error(f"Migration {migration.id} failed: {e}")
            return False
    
    @log_execution(measure_time=True)
    def rollback_migration(self, migration: Migration) -> bool:
        """Wycofuje migrację"""
        if not migration.down_sql:
            logger.error(f"Migration {migration.id} does not support rollback")
            return False
        
        logger.info(f"Rolling back migration: {migration.id}")
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(migration.down_sql)
                    conn.commit()
            
            # Oznacz jako wycofaną
            self._mark_as_rolled_back(migration.id)
            logger.info(f"Migration {migration.id} rolled back successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback of migration {migration.id} failed: {e}")
            return False
    
    def migrate(self, target: Optional[str] = None) -> int:
        """
        Wykonuje wszystkie pending migracje do określonego targetu
        
        Args:
            target: ID migracji do której migrować (None = wszystkie)
            
        Returns:
            Liczba wykonanych migracji
        """
        pending = self.get_pending_migrations()
        
        if not pending:
            logger.info("No pending migrations")
            return 0
        
        executed = 0
        
        for migration in pending:
            # Jeśli osiągnęliśmy target, przerwij
            if target and migration.id == target:
                if self.apply_migration(migration):
                    executed += 1
                break
            
            # Wykonaj migrację
            if self.apply_migration(migration):
                executed += 1
            else:
                # Przerwij przy błędzie
                logger.error(f"Migration failed, stopping at {migration.id}")
                break
        
        logger.info(f"Executed {executed} migrations")
        return executed
    
    def rollback(self, steps: int = 1) -> int:
        """
        Wycofuje określoną liczbę migracji
        
        Args:
            steps: Liczba migracji do wycofania
            
        Returns:
            Liczba wycofanych migracji
        """
        applied = self._get_applied_migrations()
        
        if not applied:
            logger.info("No migrations to rollback")
            return 0
        
        # Posortuj po dacie aplikacji (najnowsze pierwsze)
        sorted_migrations = sorted(
            applied.values(),
            key=lambda x: x.applied_at,
            reverse=True
        )
        
        rolled_back = 0
        
        for history in sorted_migrations[:steps]:
            # Znajdź definicję migracji
            migration = next(
                (m for m in MIGRATIONS if m.id == history.migration_id),
                None
            )
            
            if migration and self.rollback_migration(migration):
                rolled_back += 1
            else:
                logger.error(f"Failed to rollback {history.migration_id}")
                break
        
        logger.info(f"Rolled back {rolled_back} migrations")
        return rolled_back
    
    def status(self) -> Dict[str, Any]:
        """Zwraca status migracji"""
        applied = self._get_applied_migrations()
        pending = self.get_pending_migrations()
        
        return {
            'applied_count': len(applied),
            'pending_count': len(pending),
            'total_count': len(MIGRATIONS),
            'applied': [
                {
                    'id': h.migration_id,
                    'name': h.name,
                    'applied_at': h.applied_at.isoformat(),
                    'execution_time_ms': h.execution_time_ms
                }
                for h in applied.values()
            ],
            'pending': [
                {
                    'id': m.id,
                    'name': m.name,
                    'description': m.description
                }
                for m in pending
            ],
            'latest_applied': (
                max(applied.values(), key=lambda x: x.applied_at).migration_id
                if applied else None
            )
        }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Waliduje integralność migracji"""
        issues = []
        applied = self._get_applied_migrations()
        
        # Sprawdź checksumy
        for migration in MIGRATIONS:
            if migration.id in applied:
                if applied[migration.id].checksum != migration.checksum:
                    issues.append(
                        f"Migration {migration.id} has been modified after application"
                    )
        
        # Sprawdź zależności
        for migration in MIGRATIONS:
            if migration.dependencies:
                for dep in migration.dependencies:
                    if dep not in [m.id for m in MIGRATIONS]:
                        issues.append(
                            f"Migration {migration.id} depends on unknown migration {dep}"
                        )
        
        # Sprawdź duplikaty ID
        ids = [m.id for m in MIGRATIONS]
        if len(ids) != len(set(ids)):
            issues.append("Duplicate migration IDs found")
        
        return len(issues) == 0, issues

# ============== FUNKCJE POMOCNICZE ==============

def create_migration_file(
    name: str,
    description: str,
    up_sql: str,
    down_sql: Optional[str] = None
) -> str:
    """
    Tworzy plik z nową migracją
    
    Args:
        name: Nazwa migracji (np. "add_user_table")
        description: Opis migracji
        up_sql: SQL do wykonania
        down_sql: SQL do wycofania (opcjonalny)
        
    Returns:
        Ścieżka do utworzonego pliku
    """
    # Generuj ID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    migration_id = f"{timestamp}_{name}"
    
    # Przygotuj zawartość
    content = f'''"""
Migration: {migration_id}
Description: {description}
Created: {datetime.now().isoformat()}
"""

from database.postgres.migrations import Migration, MIGRATIONS

MIGRATIONS.append(
    Migration(
        id="{migration_id}",
        name="{name.replace('_', ' ').title()}",
        description="""{description}""",
        up_sql="""
{up_sql}
        """,
        down_sql="""
{down_sql or '-- Rollback not supported'}
        """
    )
)
'''
    
    # Zapisz plik
    migrations_dir = Path(__file__).parent / "migrations"
    migrations_dir.mkdir(exist_ok=True)
    
    file_path = migrations_dir / f"{migration_id}.py"
    file_path.write_text(content)
    
    logger.info(f"Created migration file: {file_path}")
    return str(file_path)

def load_migrations_from_directory(directory: Path) -> List[Migration]:
    """Ładuje migracje z katalogu"""
    migrations = []
    
    for file_path in sorted(directory.glob("*.py")):
        if file_path.name.startswith("__"):
            continue
        
        # Dynamicznie importuj moduł
        spec = importlib.util.spec_from_file_location(
            f"migration_{file_path.stem}",
            file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Sprawdź czy moduł ma migrację
        if hasattr(module, "migration") and isinstance(module.migration, Migration):
            migrations.append(module.migration)
    
    return migrations

# ============== CLI INTERFACE ==============

def migration_cli():
    """Prosty interfejs CLI dla migracji"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Status
    subparsers.add_parser("status", help="Show migration status")
    
    # Migrate
    migrate_parser = subparsers.add_parser("migrate", help="Run pending migrations")
    migrate_parser.add_argument(
        "--target",
        help="Migrate to specific migration ID"
    )
    
    # Rollback
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of migrations to rollback"
    )
    
    # Validate
    subparsers.add_parser("validate", help="Validate migrations")
    
    # Create
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("name", help="Migration name")
    create_parser.add_argument("--description", help="Migration description")
    
    args = parser.parse_args()
    
    # Wykonaj komendę
    manager = MigrationManager()
    
    if args.command == "status":
        status = manager.status()
        print(f"Applied: {status['applied_count']}")
        print(f"Pending: {status['pending_count']}")
        print(f"Total: {status['total_count']}")
        
        if status['pending']:
            print("\nPending migrations:")
            for m in status['pending']:
                print(f"  - {m['id']}: {m['name']}")
    
    elif args.command == "migrate":
        count = manager.migrate(args.target)
        print(f"Executed {count} migrations")
    
    elif args.command == "rollback":
        count = manager.rollback(args.steps)
        print(f"Rolled back {count} migrations")
    
    elif args.command == "validate":
        valid, issues = manager.validate()
        if valid:
            print("All migrations are valid")
        else:
            print("Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
    
    elif args.command == "create":
        print("Creating migration...")
        # Tutaj można dodać interaktywny kreator migracji

# ============== EKSPORT ==============

__all__ = [
    'Migration',
    'MigrationHistory',
    'MigrationManager',
    'MIGRATIONS',
    'create_migration_file',
    'load_migrations_from_directory',
    'migration_cli'
]

if __name__ == "__main__":
    migration_cli()
