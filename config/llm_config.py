# config/llm_config.py
"""
Konfiguracja promptów i szablonów dla Large Language Models
Zawiera prompty systemowe dla agentów, analizatorów i zadań
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import json

# ============== PROMPTY SYSTEMOWE DLA AGENTÓW ==============

SYSTEM_PROMPTS = {
    "question_analyzer": """You are an expert question analyzer for a RAG (Retrieval-Augmented Generation) system.
Your task is to analyze questions and determine:
1. Question type: factual, semantic, relational, complex, or temporal
2. Required data sources: SQL database, vector search, graph database, or combination
3. Key entities and concepts mentioned
4. Temporal context if any
5. Complexity level and suggested approach

Respond in JSON format with these fields:
{
    "question_type": "type",
    "required_sources": ["source1", "source2"],
    "entities": [{"name": "entity", "type": "type"}],
    "temporal_context": "description or null",
    "complexity": "low|medium|high",
    "suggested_approach": "description"
}

Be precise and consider the Polish language context when analyzing.""",

    "sql_agent": """You are an SQL expert assistant working with a PostgreSQL database containing documents, entities, and their relationships from the AI Devs 3 course.

Database schema:
- documents: id, task_id, content, metadata, created_at
- entities: id, name, type, properties
- document_entities: links documents to entities
- relationships: source_entity_id, target_entity_id, type

Your responsibilities:
1. Convert natural language questions to SQL queries
2. Focus on factual information retrieval
3. Use appropriate JOINs and filters
4. Consider Polish language specifics in LIKE queries
5. Always limit results appropriately

Return SQL queries that are safe, efficient, and precise.
Explain your reasoning before providing the query.""",

    "semantic_agent": """You are a semantic search specialist working with vector embeddings and similarity search.

Your role:
1. Understand the semantic meaning of questions
2. Identify key concepts for embedding search
3. Determine optimal search strategies
4. Rank and filter results by relevance
5. Synthesize information from multiple chunks

You have access to:
- Document embeddings in Qdrant
- Chunk-level search with metadata
- Similarity scores and thresholds

Focus on finding semantically related content even when exact keywords don't match.
Consider context and implied meanings in Polish language queries.""",

    "graph_agent": """You are a graph database expert working with Neo4j to analyze relationships and connections.

Graph structure:
- Nodes: Document, Person, Organization, Location, Event, Concept, Task
- Relationships: MENTIONED_IN, WORKS_FOR, LOCATED_IN, RELATED_TO, etc.

Your tasks:
1. Identify relationship-based questions
2. Convert questions to Cypher queries
3. Find paths and connections between entities
4. Analyze graph patterns
5. Discover hidden relationships

Use graph algorithms when appropriate:
- Shortest path for connections
- PageRank for importance
- Community detection for groupings

Provide clear Cypher queries with explanations.""",

    "orchestrator_agent": """You are the master orchestrator coordinating multiple specialized agents to answer complex questions.

Your responsibilities:
1. Receive questions and analyzer results
2. Delegate sub-tasks to appropriate agents (SQL, Semantic, Graph)
3. Collect and synthesize responses
4. Resolve conflicts between sources
5. Provide comprehensive, accurate answers

Decision process:
- For factual queries: primarily SQL agent
- For concept understanding: semantic agent
- For relationships: graph agent
- For complex queries: coordinate multiple agents

Always provide source attribution and confidence levels.
Synthesize information coherently in Polish when needed.""",

    "cache_agent": """You are a cache management specialist optimizing data retrieval and storage.

Your tasks:
1. Determine if queries can be answered from cache
2. Identify cache keys and TTL values
3. Manage cache invalidation strategies
4. Optimize frequently accessed data

Cache categories:
- embedding_cache: 24h TTL
- document_cache: 1h TTL
- answer_cache: 30min TTL

Consider query patterns and data freshness requirements."""
}

# ============== PROMPTY DLA RÓŻNYCH TYPÓW ZADAŃ ==============

TASK_PROMPTS = {
    "document_extraction": """Extract and clean text from the provided {doc_type} document.
Requirements:
1. Remove all formatting artifacts and HTML/XML tags
2. Preserve meaningful structure (paragraphs, lists)
3. Handle Polish characters correctly (ą, ć, ę, ł, ń, ó, ś, ź, ż)
4. Extract metadata: title, author, date if available
5. Identify document language

Output clean, readable text ready for further processing.""",

    "entity_extraction": """Extract all named entities from the text.
Focus on:
1. People (PERSON) - full names, nicknames, titles
2. Organizations (ORGANIZATION) - companies, institutions, groups
3. Locations (LOCATION) - cities, countries, addresses
4. Events (EVENT) - conferences, meetings, historical events
5. Technologies (TECHNOLOGY) - tools, frameworks, concepts
6. Dates and times (DATE/TIME)

For each entity provide:
- Text as found
- Normalized form
- Type
- Context snippet
- Confidence score

Consider Polish naming conventions and inflections.""",

    "relationship_extraction": """Identify relationships between entities in the text.
Look for:
1. Professional relationships (works for, collaborates with)
2. Location relationships (located in, happened at)
3. Temporal relationships (before, after, during)
4. Conceptual relationships (related to, part of)
5. Creation relationships (created by, authored by)

Provide:
- Source entity
- Target entity  
- Relationship type
- Direction
- Context
- Confidence score""",

    "question_generation": """Generate relevant questions based on the document content.
Create questions that:
1. Test factual understanding
2. Explore relationships between concepts
3. Require inference or synthesis
4. Cover different complexity levels
5. Are answerable from the document

Provide 5-10 questions with expected answer types.""",

    "summary_generation": """Create a comprehensive summary of the document.
Include:
1. Main topics and themes
2. Key entities and their roles
3. Important relationships
4. Temporal context
5. Technical concepts explained

Keep summary concise but informative (200-300 words).
Maintain factual accuracy and preserve Polish terms where important."""
}

# ============== PROMPTY DLA KONKRETNYCH ZADAŃ AI DEVS ==============

AIDEVS_TASK_PROMPTS = {
    "S01": "Process basic text input and prepare for storage.",
    
    "S20": """Process 'dane' files with mixed content types.
Extract all information about:
1. People and their activities
2. Locations and events
3. Technical concepts
4. Temporal information
Create comprehensive knowledge graph entries.""",

    "S25": """You are answering final questions using the complete knowledge base.
Steps:
1. Analyze question type and complexity
2. Search across all data sources
3. Synthesize information from multiple documents
4. Provide precise, well-sourced answers
5. Include confidence scores

Remember: This is the culmination of all previous tasks."""
}

# ============== PROMPTY DLA RÓŻNYCH TYPÓW PYTAŃ ==============

QUESTION_TYPE_PROMPTS = {
    "factual": """Answer this factual question with precise information.
Look for: specific dates, names, numbers, events.
Cite sources directly.""",

    "semantic": """Answer this conceptual question by understanding the deeper meaning.
Consider: context, implications, related concepts.
Synthesize information from multiple sources.""",

    "relational": """Answer this question about relationships and connections.
Analyze: how entities relate, interaction patterns, network effects.
Use graph data to show connections.""",

    "temporal": """Answer this time-based question with chronological precision.
Consider: sequences, timelines, cause and effect.
Provide temporal context.""",

    "complex": """Answer this multi-faceted question comprehensively.
Break down into: sub-questions, multiple perspectives, synthesis.
Use all available data sources."""
}

# ============== SZABLONY ODPOWIEDZI ==============

ANSWER_TEMPLATES = {
    "single_fact": """Based on the available data:

**Answer**: {answer}

**Source**: {source}
**Confidence**: {confidence}%""",

    "multiple_sources": """Based on analysis of multiple sources:

**Answer**: {answer}

**Supporting Evidence**:
{evidence_list}

**Sources**: {sources}
**Overall Confidence**: {confidence}%""",

    "relationship_answer": """The relationship analysis shows:

**Connection**: {entity1} → {relationship} → {entity2}

**Context**: {context}

**Additional Relationships**:
{related_connections}

**Graph Visualization**: {graph_description}""",

    "not_found": """I couldn't find sufficient information to answer this question.

**What I searched**: {search_description}
**Suggestion**: {suggestion}

Please provide more context or rephrase the question.""",

    "synthesis": """Based on comprehensive analysis:

**Summary**: {summary}

**Key Points**:
{key_points}

**Evidence from Multiple Sources**:
{evidence_synthesis}

**Confidence**: {confidence}%
**Note**: {additional_notes}"""
}

# ============== PROMPTY DLA PRZETWARZANIA DOKUMENTÓW ==============

PROCESSING_PROMPTS = {
    "pdf_extraction": """Extract text from PDF maintaining structure.
Handle: tables, lists, headers, footnotes.
Preserve: formatting intent, reading order.""",

    "html_cleaning": """Clean HTML content for text extraction.
Remove: scripts, styles, navigation.
Preserve: semantic structure, links context.""",

    "audio_transcription": """Transcribe audio with attention to:
1. Polish language specifics
2. Technical terminology
3. Speaker identification
4. Timestamps for key moments""",

    "image_analysis": """Analyze image content:
1. Extract visible text (OCR)
2. Describe visual elements
3. Identify entities in images
4. Note relevant context"""
}

# ============== CHAIN OF THOUGHT PROMPTS ==============

CHAIN_OF_THOUGHT_PROMPTS = {
    "reasoning": """Let me think through this step by step:

1. Understanding the question: {question_analysis}
2. Identifying required information: {info_needed}
3. Searching relevant sources: {search_strategy}
4. Analyzing findings: {analysis}
5. Formulating answer: {conclusion}""",

    "error_analysis": """Encountering an issue:

1. What went wrong: {error_description}
2. Possible causes: {causes}
3. Alternative approach: {alternative}
4. Retry strategy: {retry_plan}"""
}

# ============== KLASA ZARZĄDZAJĄCA PROMPTAMI ==============

class LLMPrompts:
    """Zarządza promptami i ich dynamicznym generowaniem"""
    
    @staticmethod
    def get_system_prompt(agent_type: str) -> str:
        """Pobiera prompt systemowy dla agenta"""
        if agent_type not in SYSTEM_PROMPTS:
            raise ValueError(f"Nieznany typ agenta: {agent_type}")
        return SYSTEM_PROMPTS[agent_type]
    
    @staticmethod
    def get_task_prompt(task_type: str, **kwargs) -> str:
        """Generuje prompt dla zadania z parametrami"""
        if task_type in TASK_PROMPTS:
            return TASK_PROMPTS[task_type].format(**kwargs)
        return TASK_PROMPTS.get("document_extraction", "").format(**kwargs)
    
    @staticmethod
    def get_question_prompt(question_type: str) -> str:
        """Pobiera prompt dla typu pytania"""
        return QUESTION_TYPE_PROMPTS.get(
            question_type, 
            QUESTION_TYPE_PROMPTS["semantic"]
        )
    
    @staticmethod
    def format_answer(template_type: str, **kwargs) -> str:
        """Formatuje odpowiedź według szablonu"""
        template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["single_fact"])
        return template.format(**kwargs)
    
    @staticmethod
    def create_rag_prompt(question: str, context: List[str], metadata: Optional[Dict] = None) -> str:
        """Tworzy prompt dla RAG z kontekstem"""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""Based on the following context, answer the question accurately.

CONTEXT:
{context_str}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say so
3. Cite the context numbers [1], [2] etc. when referencing information
4. Be concise but complete
5. Maintain factual accuracy

ANSWER:"""
        
        if metadata:
            prompt += f"\n\nADDITIONAL METADATA: {json.dumps(metadata, ensure_ascii=False)}"
        
        return prompt
    
    @staticmethod
    def create_extraction_prompt(text: str, extraction_type: str) -> str:
        """Tworzy prompt do ekstrakcji informacji"""
        base_prompts = {
            "entities": TASK_PROMPTS["entity_extraction"],
            "relationships": TASK_PROMPTS["relationship_extraction"],
            "summary": TASK_PROMPTS["summary_generation"],
            "questions": TASK_PROMPTS["question_generation"]
        }
        
        prompt = base_prompts.get(extraction_type, "Extract relevant information from the text.")
        return f"{prompt}\n\nTEXT:\n{text}\n\nEXTRACTION:"
    
    @staticmethod
    def create_multiagent_prompt(task: str, agent_results: Dict[str, Any]) -> str:
        """Tworzy prompt do syntezy wyników z wielu agentów"""
        results_text = "\n\n".join([
            f"=== {agent.upper()} AGENT RESULTS ===\n{result}"
            for agent, result in agent_results.items()
        ])
        
        return f"""Synthesize the following results from multiple specialized agents:

TASK: {task}

AGENT RESULTS:
{results_text}

SYNTHESIS INSTRUCTIONS:
1. Identify agreements and conflicts between agents
2. Prioritize information based on source reliability
3. Create a coherent, comprehensive answer
4. Note any uncertainties or contradictions
5. Provide confidence assessment

FINAL SYNTHESIS:"""

# ============== PROMPT ENGINEERING UTILITIES ==============

def optimize_prompt_tokens(prompt: str, max_tokens: int = 2000) -> str:
    """Optymalizuje prompt do limitu tokenów"""
    # Tutaj można dodać logikę liczenia tokenów
    # i skracania promptu jeśli potrzeba
    if len(prompt) > max_tokens * 4:  # Przybliżone
        return prompt[:max_tokens * 4] + "..."
    return prompt

def add_few_shot_examples(base_prompt: str, examples: List[Dict[str, str]]) -> str:
    """Dodaje przykłady few-shot do promptu"""
    if not examples:
        return base_prompt
    
    examples_text = "\n\n".join([
        f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
        for i, ex in enumerate(examples)
    ])
    
    return f"{base_prompt}\n\nEXAMPLES:\n{examples_text}"

def create_prompt_with_schema(prompt: str, output_schema: Dict[str, Any]) -> str:
    """Dodaje schemat wyjścia do promptu"""
    schema_text = json.dumps(output_schema, indent=2, ensure_ascii=False)
    return f"{prompt}\n\nOUTPUT SCHEMA:\n{schema_text}"

# ============== PROMPTY DLA WALIDACJI ==============

VALIDATION_PROMPTS = {
    "answer_quality": """Evaluate the quality of this answer:

Answer: {answer}
Question: {question}

Criteria:
1. Accuracy (0-10)
2. Completeness (0-10)
3. Clarity (0-10)
4. Source support (0-10)

Provide scores and brief justification.""",

    "extraction_validation": """Validate the extracted information:

Original: {original}
Extracted: {extracted}

Check for:
1. Completeness
2. Accuracy
3. Format correctness
4. No hallucinations

Valid? (true/false) with explanation."""
}

# ============== KONFIGURACJA MODELI ==============

MODEL_CONFIGS = {
    "gpt-4-turbo": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "best_for": ["complex_reasoning", "synthesis", "extraction"]
    },
    "gpt-3.5-turbo": {
        "max_tokens": 2048,
        "temperature": 0.5,
        "top_p": 0.9,
        "best_for": ["simple_queries", "classification", "quick_answers"]
    }
}

# ============== EKSPORT ==============

def get_prompt(prompt_type: str, **kwargs) -> str:
    """Uniwersalna funkcja do pobierania promptów"""
    prompt_sources = {
        "system": SYSTEM_PROMPTS,
        "task": TASK_PROMPTS,
        "aidevs": AIDEVS_TASK_PROMPTS,
        "question": QUESTION_TYPE_PROMPTS,
        "template": ANSWER_TEMPLATES,
        "processing": PROCESSING_PROMPTS,
        "validation": VALIDATION_PROMPTS
    }
    
    for source_name, source_dict in prompt_sources.items():
        if prompt_type in source_dict:
            prompt = source_dict[prompt_type]
            if kwargs:
                return prompt.format(**kwargs)
            return prompt
    
    raise ValueError(f"Nieznany typ promptu: {prompt_type}")

# ============== EKSPORT ==============

__all__ = [
    "LLMPrompts",
    "SYSTEM_PROMPTS",
    "TASK_PROMPTS",
    "AIDEVS_TASK_PROMPTS",
    "QUESTION_TYPE_PROMPTS",
    "ANSWER_TEMPLATES",
    "PROCESSING_PROMPTS",
    "CHAIN_OF_THOUGHT_PROMPTS",
    "VALIDATION_PROMPTS",
    "MODEL_CONFIGS",
    "get_prompt",
    "optimize_prompt_tokens",
    "add_few_shot_examples",
    "create_prompt_with_schema"
]
