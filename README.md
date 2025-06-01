# AI Devs 3 Reloaded - System RAG

Kompletny system RAG (Retrieval-Augmented Generation) dla kursu AI Devs 3 Reloaded.

## 🏗️ Architektura

System integruje trzy bazy danych:
- **PostgreSQL** - przechowywanie surowych danych i metadanych
- **Qdrant** - wyszukiwanie wektorowe i embeddingi
- **Neo4j** - graf wiedzy i relacje między encjami

## 🚀 Instalacja

1. Sklonuj repozytorium
2. Skopiuj `.env.example` do `.env` i uzupełnij dane
3. Uruchom bazy danych: `docker-compose up -d`
4. Zainstaluj zależności: `pip install -r requirements.txt`
5. Zainicjalizuj bazy: `python scripts/setup_databases.py`

## 📁 Struktura projektu

```
aidevs3-rag/
├── config/          # Konfiguracja systemu
├── database/        # Managery baz danych
├── extractors/      # Ekstraktory różnych typów plików
├── processors/      # Procesory dokumentów
├── agents/          # Agenty AI do odpowiadania na pytania
├── models/          # Modele danych
├── tasks/           # Zadania S01-S25
└── tests/           # Testy jednostkowe i integracyjne
```

## 🎯 Użycie

```python
from rag_system import RAGSystem

# Inicjalizacja systemu
rag = RAGSystem()

# Przetworzenie dokumentu
rag.process_document("path/to/document.pdf")

# Odpowiedź na pytanie
answer = rag.answer_question("Jakie są główne koncepcje w dokumencie?")
```

## 📝 Zadania

Każde zadanie (S01-S25) ma swój katalog w `tasks/` z dedykowanym procesorem.

## 🧪 Testy

Uruchom testy: `pytest tests/`

## 📊 Monitoring

Logi zapisywane są w katalogu `logs/`. Poziom logowania można zmienić w `.env`.
