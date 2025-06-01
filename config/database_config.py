# config/database_config.py
"""
Konfiguracja schematów i struktur dla wszystkich baz danych
PostgreSQL, Qdrant, Neo4j
"""

from typing import Dict, List, Any
from datetime import datetime
from .settings import get_config

config = get_config()

# ============== POSTGRESQL SCHEMAS ==============

# Główne tabele
POSTGRES_SCHEMAS = {
    "documents": """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_id VARCHAR(10) NOT NULL,
            source_path TEXT NOT NULL,
            doc_type VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            cleaned_content TEXT,
            metadata JSONB DEFAULT '{}',
            file_hash VARCHAR(64) UNIQUE,
            processing_status VARCHAR(20) DEFAULT 'pending',
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP WITH TIME ZONE
        );
        
        CREATE INDEX IF NOT EXISTS idx_documents_task_id ON documents(task_id);
        CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
        CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);
    """,
    
    "document_chunks": """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER NOT NULL,
            embedding_id VARCHAR(100),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(document_id, chunk_index)
        );
        
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON document_chunks(embedding_id);
    """,
    
    "entities": """
        CREATE TABLE IF NOT EXISTS entities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            type VARCHAR(50) NOT NULL,
            normalized_name TEXT NOT NULL,
            properties JSONB DEFAULT '{}',
            first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            occurrence_count INTEGER DEFAULT 1,
            confidence_score FLOAT DEFAULT 1.0
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_unique ON entities(normalized_name, type);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_properties ON entities USING GIN (properties);
    """,
    
    "document_entities": """
        CREATE TABLE IF NOT EXISTS document_entities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE,
            position_start INTEGER,
            position_end INTEGER,
            context TEXT,
            confidence_score FLOAT DEFAULT 1.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(document_id, entity_id, position_start)
        );
        
        CREATE INDEX IF NOT EXISTS idx_doc_entities_document ON document_entities(document_id);
        CREATE INDEX IF NOT EXISTS idx_doc_entities_entity ON document_entities(entity_id);
    """,
    
    "relationships": """
        CREATE TABLE IF NOT EXISTS relationships (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relationship_type VARCHAR(100) NOT NULL,
            properties JSONB DEFAULT '{}',
            confidence_score FLOAT DEFAULT 1.0,
            first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            occurrence_count INTEGER DEFAULT 1
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_unique 
            ON relationships(source_entity_id, target_entity_id, relationship_type);
        CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
    """,
    
    "processing_history": """
        CREATE TABLE IF NOT EXISTS processing_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_id VARCHAR(10) NOT NULL,
            document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            action VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            details JSONB DEFAULT '{}',
            error_message TEXT,
            duration_seconds FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_history_task_id ON processing_history(task_id);
        CREATE INDEX IF NOT EXISTS idx_history_document_id ON processing_history(document_id);
        CREATE INDEX IF NOT EXISTS idx_history_created_at ON processing_history(created_at DESC);
    """,
    
    "questions_answers": """
        CREATE TABLE IF NOT EXISTS questions_answers (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            question TEXT NOT NULL,
            question_type VARCHAR(50),
            answer TEXT NOT NULL,
            sources JSONB DEFAULT '[]',
            confidence_score FLOAT,
            processing_time_ms INTEGER,
            used_agents JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_qa_created_at ON questions_answers(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_qa_question_type ON questions_answers(question_type);
    """,
    
    "cache_entries": """
        CREATE TABLE IF NOT EXISTS cache_entries (
            key VARCHAR(255) PRIMARY KEY,
            value JSONB NOT NULL,
            category VARCHAR(50) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1
        );
        
        CREATE INDEX IF NOT EXISTS idx_cache_category ON cache_entries(category);
        CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
    """
}

# Widoki PostgreSQL
POSTGRES_VIEWS = {
    "document_stats": """
        CREATE OR REPLACE VIEW document_stats AS
        SELECT 
            task_id,
            doc_type,
            COUNT(*) as document_count,
            SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as completed_count,
            SUM(CASE WHEN processing_status = 'error' THEN 1 ELSE 0 END) as error_count,
            AVG(LENGTH(content)) as avg_content_length,
            MAX(created_at) as last_added
        FROM documents
        GROUP BY task_id, doc_type;
    """,
    
    "entity_stats": """
        CREATE OR REPLACE VIEW entity_stats AS
        SELECT 
            e.type,
            COUNT(DISTINCT e.id) as unique_entities,
            SUM(e.occurrence_count) as total_occurrences,
            COUNT(DISTINCT de.document_id) as document_count,
            AVG(e.confidence_score) as avg_confidence
        FROM entities e
        LEFT JOIN document_entities de ON e.id = de.entity_id
        GROUP BY e.type;
    """
}

# Funkcje PostgreSQL
POSTGRES_FUNCTIONS = {
    "update_timestamp": """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    
    "normalize_text": """
        CREATE OR REPLACE FUNCTION normalize_text(input_text TEXT)
        RETURNS TEXT AS $$
        BEGIN
            RETURN LOWER(TRIM(regexp_replace(input_text, '[^a-zA-Z0-9żźćńółęąś ]', '', 'g')));
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """
}

# ============== QDRANT CONFIGURATION ==============

QDRANT_COLLECTIONS = {
    "documents": {
        "name": config.qdrant.default_collection,
        "vector_config": {
            "size": config.qdrant.vector_size,
            "distance": config.qdrant.distance_metric
        },
        "payload_schema": {
            "document_id": "keyword",
            "task_id": "keyword",
            "doc_type": "keyword",
            "chunk_index": "integer",
            "content": "text",
            "metadata": "json",
            "created_at": "datetime"
        },
        "indexes": [
            {"field": "task_id", "type": "keyword"},
            {"field": "doc_type", "type": "keyword"},
            {"field": "document_id", "type": "keyword"}
        ]
    },
    
    "entities": {
        "name": "entities_embeddings",
        "vector_config": {
            "size": config.qdrant.vector_size,
            "distance": config.qdrant.distance_metric
        },
        "payload_schema": {
            "entity_id": "keyword",
            "entity_name": "text",
            "entity_type": "keyword",
            "description": "text",
            "properties": "json"
        }
    },
    
    "questions": {
        "name": "questions_embeddings",
        "vector_config": {
            "size": config.qdrant.vector_size,
            "distance": config.qdrant.distance_metric
        },
        "payload_schema": {
            "question": "text",
            "answer": "text",
            "question_type": "keyword",
            "sources": "json"
        }
    }
}

# ============== NEO4J CONFIGURATION ==============

# Typy węzłów (node labels)
NEO4J_NODE_TYPES = {
    "Document": {
        "properties": {
            "id": "STRING",
            "task_id": "STRING",
            "source_path": "STRING",
            "doc_type": "STRING",
            "created_at": "DATETIME",
            "title": "STRING"
        },
        "constraints": [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
        ],
        "indexes": [
            "CREATE INDEX document_task_id IF NOT EXISTS FOR (d:Document) ON (d.task_id)",
            "CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.doc_type)"
        ]
    },
    
    "Person": {
        "properties": {
            "id": "STRING",
            "name": "STRING",
            "normalized_name": "STRING",
            "properties": "JSON"
        },
        "constraints": [
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE"
        ],
        "indexes": [
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.normalized_name)"
        ]
    },
    
    "Organization": {
        "properties": {
            "id": "STRING",
            "name": "STRING",
            "normalized_name": "STRING",
            "type": "STRING",
            "properties": "JSON"
        },
        "constraints": [
            "CREATE CONSTRAINT org_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE"
        ]
    },
    
    "Location": {
        "properties": {
            "id": "STRING",
            "name": "STRING",
            "normalized_name": "STRING",
            "coordinates": "POINT",
            "properties": "JSON"
        },
        "constraints": [
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE"
        ]
    },
    
    "Event": {
        "properties": {
            "id": "STRING",
            "name": "STRING",
            "date": "DATETIME",
            "properties": "JSON"
        },
        "constraints": [
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE"
        ]
    },
    
    "Concept": {
        "properties": {
            "id": "STRING",
            "name": "STRING",
            "category": "STRING",
            "properties": "JSON"
        },
        "constraints": [
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE"
        ]
    },
    
    "Task": {
        "properties": {
            "id": "STRING",
            "name": "STRING",
            "created_at": "DATETIME"
        },
        "constraints": [
            "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE"
        ]
    }
}

# Typy relacji
NEO4J_RELATIONSHIP_TYPES = {
    "MENTIONED_IN": {
        "properties": {
            "context": "STRING",
            "confidence": "FLOAT",
            "position": "INTEGER"
        }
    },
    "WORKS_FOR": {
        "properties": {
            "role": "STRING",
            "since": "DATETIME",
            "until": "DATETIME"
        }
    },
    "LOCATED_IN": {
        "properties": {
            "type": "STRING"
        }
    },
    "RELATED_TO": {
        "properties": {
            "type": "STRING",
            "strength": "FLOAT"
        }
    },
    "KNOWS": {
        "properties": {
            "since": "DATETIME",
            "context": "STRING"
        }
    },
    "CREATED_BY": {
        "properties": {
            "date": "DATETIME"
        }
    },
    "PART_OF": {
        "properties": {
            "role": "STRING"
        }
    },
    "HAPPENED_AT": {
        "properties": {
            "date": "DATETIME"
        }
    },
    "CONTAINS": {
        "properties": {
            "chunk_index": "INTEGER"
        }
    },
    "REFERENCES": {
        "properties": {
            "context": "STRING"
        }
    },
    "PROCESSED_BY": {
        "properties": {
            "date": "DATETIME",
            "status": "STRING"
        }
    }
}

# Zapytania Cypher do analizy
NEO4J_ANALYSIS_QUERIES = {
    "most_connected_entities": """
        MATCH (n)
        WITH n, size((n)--()) as degree
        ORDER BY degree DESC
        LIMIT 10
        RETURN n.name as entity, labels(n)[0] as type, degree
    """,
    
    "entity_relationships": """
        MATCH (e1)-[r]->(e2)
        WHERE e1.id = $entity_id
        RETURN e1, r, e2
    """,
    
    "document_entity_graph": """
        MATCH (d:Document {id: $document_id})-[:CONTAINS]->(e)
        OPTIONAL MATCH (e)-[r]-(related)
        RETURN d, e, r, related
    """,
    
    "path_between_entities": """
        MATCH path = shortestPath((e1 {id: $entity1_id})-[*..5]-(e2 {id: $entity2_id}))
        RETURN path
    """,
    
    "task_knowledge_graph": """
        MATCH (t:Task {id: $task_id})-[:PROCESSED_BY]->(d:Document)
        OPTIONAL MATCH (d)-[:CONTAINS]->(e)
        OPTIONAL MATCH (e)-[r]-(related)
        RETURN t, d, e, r, related
    """
}

# ============== INICJALIZACJA BAZ DANYCH ==============

class DatabaseInitializer:
    """Klasa do inicjalizacji wszystkich baz danych"""
    
    @staticmethod
    def get_postgres_init_sql() -> str:
        """Zwraca kompletny SQL do inicjalizacji PostgreSQL"""
        sql_parts = ["-- PostgreSQL Initialization Script\n"]
        
        # Dodaj schematy
        for table_name, schema in POSTGRES_SCHEMAS.items():
            sql_parts.append(f"-- Table: {table_name}")
            sql_parts.append(schema)
            sql_parts.append("")
        
        # Dodaj widoki
        for view_name, view_sql in POSTGRES_VIEWS.items():
            sql_parts.append(f"-- View: {view_name}")
            sql_parts.append(view_sql)
            sql_parts.append("")
        
        # Dodaj funkcje
        for func_name, func_sql in POSTGRES_FUNCTIONS.items():
            sql_parts.append(f"-- Function: {func_name}")
            sql_parts.append(func_sql)
            sql_parts.append("")
        
        return "\n".join(sql_parts)
    
    @staticmethod
    def get_qdrant_collections() -> List[Dict[str, Any]]:
        """Zwraca konfigurację kolekcji Qdrant"""
        return list(QDRANT_COLLECTIONS.values())
    
    @staticmethod
    def get_neo4j_init_cypher() -> List[str]:
        """Zwraca listę komend Cypher do inicjalizacji Neo4j"""
        cypher_commands = []
        
        # Dodaj constraints i indeksy dla węzłów
        for node_type, config in NEO4J_NODE_TYPES.items():
            cypher_commands.extend(config.get("constraints", []))
            cypher_commands.extend(config.get("indexes", []))
        
        return cypher_commands
    
    @staticmethod
    def get_redis_init_commands() -> List[tuple]:
        """Zwraca komendy do inicjalizacji Redis"""
        return [
            ("CONFIG", "SET", "maxmemory", "2gb"),
            ("CONFIG", "SET", "maxmemory-policy", "allkeys-lru"),
            ("CONFIG", "SET", "save", "900 1 300 10 60 10000")
        ]

# ============== ZAPYTANIA GOTOWE ==============

# Przydatne zapytania SQL
SQL_QUERIES = {
    "get_unprocessed_documents": """
        SELECT * FROM documents 
        WHERE processing_status = 'pending' 
        ORDER BY created_at ASC 
        LIMIT $1
    """,
    
    "get_task_documents": """
        SELECT d.*, 
               COUNT(dc.id) as chunk_count,
               COUNT(DISTINCT de.entity_id) as entity_count
        FROM documents d
        LEFT JOIN document_chunks dc ON d.id = dc.document_id
        LEFT JOIN document_entities de ON d.id = de.document_id
        WHERE d.task_id = $1
        GROUP BY d.id
        ORDER BY d.created_at DESC
    """,
    
    "search_entities": """
        SELECT e.*, COUNT(de.id) as occurrence_count
        FROM entities e
        JOIN document_entities de ON e.id = de.entity_id
        WHERE e.normalized_name ILIKE $1
        GROUP BY e.id
        ORDER BY occurrence_count DESC
        LIMIT $2
    """,
    
    "get_related_documents": """
        WITH entity_docs AS (
            SELECT DISTINCT de2.document_id
            FROM document_entities de1
            JOIN entities e ON de1.entity_id = e.id
            JOIN document_entities de2 ON e.id = de2.entity_id
            WHERE de1.document_id = $1 AND de2.document_id != $1
        )
        SELECT d.*, COUNT(*) as shared_entities
        FROM documents d
        JOIN entity_docs ed ON d.id = ed.document_id
        GROUP BY d.id
        ORDER BY shared_entities DESC
        LIMIT $2
    """
}

# ============== MIGRACJE ==============

DATABASE_MIGRATIONS = {
    "001_initial_schema": {
        "postgres": lambda: DatabaseInitializer.get_postgres_init_sql(),
        "neo4j": lambda: DatabaseInitializer.get_neo4j_init_cypher(),
        "description": "Początkowy schemat baz danych"
    },
    
    "002_add_embeddings_metadata": {
        "postgres": """
            ALTER TABLE document_chunks 
            ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100),
            ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
        """,
        "description": "Dodanie metadanych embeddingów"
    },
    
    "003_add_task_metadata": {
        "postgres": """
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS task_metadata JSONB DEFAULT '{}';
            
            CREATE INDEX IF NOT EXISTS idx_documents_task_metadata 
            ON documents USING GIN (task_metadata);
        """,
        "description": "Dodanie metadanych zadań"
    }
}

# ============== EKSPORT ==============

__all__ = [
    "POSTGRES_SCHEMAS",
    "POSTGRES_VIEWS", 
    "POSTGRES_FUNCTIONS",
    "QDRANT_COLLECTIONS",
    "NEO4J_NODE_TYPES",
    "NEO4J_RELATIONSHIP_TYPES",
    "NEO4J_ANALYSIS_QUERIES",
    "DatabaseInitializer",
    "SQL_QUERIES",
    "DATABASE_MIGRATIONS"
]
