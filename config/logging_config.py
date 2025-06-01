# config/logging_config.py
"""
Konfiguracja systemu logowania dla RAG AI Devs 3
Obsługuje różne poziomy logowania, rotację plików, metryki wydajności
"""

import os
import sys
import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from functools import wraps
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

# Import konfiguracji głównej
from .settings import get_config

config = get_config()

# ============== FORMATTERY ==============

class ColoredFormatter(logging.Formatter):
    """Formatter z kolorami dla konsoli"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """Formatter dla logów w formacie JSON"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Dodaj dodatkowe pola jeśli istnieją
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id
        if hasattr(record, 'document_id'):
            log_data['document_id'] = record.document_id
        if hasattr(record, 'duration'):
            log_data['duration'] = record.duration
        if hasattr(record, 'metadata'):
            log_data['metadata'] = record.metadata
            
        # Dodaj stack trace dla błędów
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False)

class StructuredFormatter(logging.Formatter):
    """Formatter dla logów strukturalnych"""
    
    def format(self, record):
        # Podstawowe informacje
        base_format = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "
            f"[{record.levelname:8}] "
            f"[{record.name:20}] "
        )
        
        # Dodaj kontekst jeśli istnieje
        context_parts = []
        if hasattr(record, 'task_id'):
            context_parts.append(f"task={record.task_id}")
        if hasattr(record, 'document_id'):
            context_parts.append(f"doc={record.document_id[:8]}")
        if hasattr(record, 'agent'):
            context_parts.append(f"agent={record.agent}")
            
        if context_parts:
            base_format += f"[{' '.join(context_parts)}] "
            
        # Wiadomość
        base_format += record.getMessage()
        
        # Dodaj metryki jeśli istnieją
        if hasattr(record, 'duration'):
            base_format += f" [duration={record.duration:.3f}s]"
        if hasattr(record, 'tokens'):
            base_format += f" [tokens={record.tokens}]"
            
        # Stack trace dla błędów
        if record.exc_info:
            base_format += f"\n{''.join(traceback.format_exception(*record.exc_info))}"
            
        return base_format

# ============== FILTRY ==============

class TaskFilter(logging.Filter):
    """Filtr logów po task_id"""
    
    def __init__(self, task_ids: Optional[list] = None):
        super().__init__()
        self.task_ids = task_ids or []
    
    def filter(self, record):
        if not self.task_ids:
            return True
        return getattr(record, 'task_id', None) in self.task_ids

class LevelFilter(logging.Filter):
    """Filtr po poziomie logowania"""
    
    def __init__(self, min_level: str, max_level: str):
        super().__init__()
        self.min_level = getattr(logging, min_level.upper())
        self.max_level = getattr(logging, max_level.upper())
    
    def filter(self, record):
        return self.min_level <= record.levelno <= self.max_level

class RateLimitFilter(logging.Filter):
    """Ogranicza liczbę podobnych logów"""
    
    def __init__(self, rate: int = 10, per: int = 60):
        super().__init__()
        self.rate = rate
        self.per = per
        self.messages = defaultdict(list)
        self._lock = threading.Lock()
    
    def filter(self, record):
        with self._lock:
            now = time.time()
            key = (record.name, record.levelno, record.getMessage())
            
            # Usuń stare wpisy
            self.messages[key] = [
                timestamp for timestamp in self.messages[key]
                if now - timestamp < self.per
            ]
            
            # Sprawdź czy można zalogować
            if len(self.messages[key]) < self.rate:
                self.messages[key].append(now)
                return True
            
            return False

# ============== HANDLERY ==============

class DatabaseHandler(logging.Handler):
    """Handler zapisujący logi do bazy danych"""
    
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
    
    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'metadata': {
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'task_id': getattr(record, 'task_id', None),
                    'document_id': getattr(record, 'document_id', None),
                }
            }
            
            if record.exc_info:
                log_entry['exception'] = self.format(record)
                
            # Zapisz do bazy (async jeśli możliwe)
            self.db_manager.save_log(log_entry)
            
        except Exception:
            self.handleError(record)

class MetricsHandler(logging.Handler):
    """Handler zbierający metryki z logów"""
    
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'errors': 0,
            'last_timestamp': None
        })
    
    def emit(self, record):
        if hasattr(record, 'metric_name'):
            metric = self.metrics[record.metric_name]
            metric['count'] += 1
            metric['last_timestamp'] = datetime.now()
            
            if hasattr(record, 'duration'):
                metric['total_duration'] += record.duration
                
            if record.levelno >= logging.ERROR:
                metric['errors'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Zwraca zebrane metryki"""
        return dict(self.metrics)

# ============== LOGGERY CONTEXTUALNE ==============

class ContextLogger:
    """Logger z kontekstem dla śledzenia operacji"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context = {}
    
    def set_context(self, **kwargs):
        """Ustawia kontekst dla wszystkich logów"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Czyści kontekst"""
        self.context.clear()
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        """Loguje z kontekstem"""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

# ============== DEKORATORY ==============

def log_execution(
    level: str = "INFO",
    include_args: bool = True,
    include_result: bool = False,
    measure_time: bool = True
):
    """Dekorator do logowania wykonania funkcji"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            log_level = getattr(logging, level.upper())
            
            # Przygotuj informacje o wywołaniu
            func_name = func.__name__
            if include_args:
                args_str = f"args={args}, kwargs={kwargs}"
            else:
                args_str = "..."
                
            # Log rozpoczęcia
            logger.log(log_level, f"Starting {func_name}({args_str})")
            
            # Wykonaj funkcję z pomiarem czasu
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log sukcesu
                msg = f"Completed {func_name}"
                if measure_time:
                    msg += f" in {duration:.3f}s"
                if include_result:
                    msg += f" with result: {result}"
                    
                logger.log(log_level, msg, extra={'duration': duration})
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {func_name} after {duration:.3f}s: {str(e)}",
                    exc_info=True,
                    extra={'duration': duration}
                )
                raise
                
        return wrapper
    return decorator

def log_async_execution(
    level: str = "INFO",
    include_args: bool = True,
    include_result: bool = False,
    measure_time: bool = True
):
    """Dekorator do logowania wykonania funkcji asynchronicznych"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            log_level = getattr(logging, level.upper())
            
            # Przygotuj informacje o wywołaniu
            func_name = func.__name__
            if include_args:
                args_str = f"args={args}, kwargs={kwargs}"
            else:
                args_str = "..."
                
            # Log rozpoczęcia
            logger.log(log_level, f"Starting async {func_name}({args_str})")
            
            # Wykonaj funkcję z pomiarem czasu
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log sukcesu
                msg = f"Completed async {func_name}"
                if measure_time:
                    msg += f" in {duration:.3f}s"
                if include_result:
                    msg += f" with result: {result}"
                    
                logger.log(log_level, msg, extra={'duration': duration})
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed async {func_name} after {duration:.3f}s: {str(e)}",
                    exc_info=True,
                    extra={'duration': duration}
                )
                raise
                
        return wrapper
    return decorator

# ============== CONTEXT MANAGERS ==============

@contextmanager
def log_context(**kwargs):
    """Context manager do tymczasowego ustawienia kontekstu logowania"""
    logger = logging.getLogger()
    old_context = {}
    
    # Zapisz i ustaw nowy kontekst
    for key, value in kwargs.items():
        if hasattr(logger, key):
            old_context[key] = getattr(logger, key)
        setattr(logger, key, value)
    
    try:
        yield
    finally:
        # Przywróć stary kontekst
        for key in kwargs:
            if key in old_context:
                setattr(logger, key, old_context[key])
            else:
                delattr(logger, key)

@contextmanager
def log_operation(operation_name: str, **metadata):
    """Context manager do logowania operacji z czasem wykonania"""
    logger = get_logger(__name__)
    
    # Log rozpoczęcia
    logger.info(f"Starting operation: {operation_name}", extra=metadata)
    start_time = time.time()
    
    try:
        yield
        
        # Log sukcesu
        duration = time.time() - start_time
        logger.info(
            f"Completed operation: {operation_name}",
            extra={**metadata, 'duration': duration}
        )
        
    except Exception as e:
        # Log błędu
        duration = time.time() - start_time
        logger.error(
            f"Failed operation: {operation_name} - {str(e)}",
            exc_info=True,
            extra={**metadata, 'duration': duration}
        )
        raise

# ============== KONFIGURACJA LOGGERÓW ==============

def setup_logging():
    """Konfiguruje system logowania"""
    
    # Usuń istniejące handlery
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Ustaw poziom główny
    root_logger.setLevel(getattr(logging, config.logging.level.upper()))
    
    # ===== HANDLER KONSOLI =====
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Użyj kolorowego formattera dla konsoli
    if sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = StructuredFormatter()
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # ===== HANDLER PLIKU (ROTACJA) =====
    log_file = config.logs_dir / "aidevs3_rag.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    # ===== HANDLER JSON (DLA ANALIZY) =====
    json_file = config.logs_dir / "aidevs3_rag.json"
    json_handler = logging.handlers.RotatingFileHandler(
        json_file,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(json_handler)
    
    # ===== HANDLER BŁĘDÓW =====
    error_file = config.logs_dir / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    error_handler.addFilter(LevelFilter("ERROR", "CRITICAL"))
    root_logger.addHandler(error_handler)
    
    # ===== HANDLER METRYK =====
    metrics_handler = MetricsHandler()
    metrics_handler.setLevel(logging.INFO)
    root_logger.addHandler(metrics_handler)
    
    # ===== KONFIGURACJA LOGGERÓW MODUŁÓW =====
    for module, level in config.logging.module_levels.items():
        module_logger = logging.getLogger(module)
        module_logger.setLevel(getattr(logging, level.upper()))
    
    # Specjalne ustawienia dla zewnętrznych bibliotek
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Dodaj rate limiting dla niektórych loggerów
    rate_limit_filter = RateLimitFilter(rate=10, per=60)
    logging.getLogger("database").addFilter(rate_limit_filter)
    logging.getLogger("extractors").addFilter(rate_limit_filter)
    
    # Log informacji o konfiguracji
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={config.logging.level}, log_dir={config.logs_dir}")

# ============== FUNKCJE POMOCNICZE ==============

def get_logger(name: str) -> Union[logging.Logger, ContextLogger]:
    """Zwraca logger dla modułu"""
    base_logger = logging.getLogger(name)
    
    # Zwróć ContextLogger dla lepszego śledzenia
    if config.system.debug:
        return ContextLogger(base_logger)
    
    return base_logger

def log_api_call(
    service: str,
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict] = None,
    response_status: Optional[int] = None,
    duration: Optional[float] = None,
    error: Optional[str] = None
):
    """Loguje wywołanie API"""
    logger = get_logger("api_calls")
    
    log_data = {
        'service': service,
        'endpoint': endpoint,
        'method': method,
        'params': params or {},
        'response_status': response_status,
        'duration': duration,
        'error': error
    }
    
    if error:
        logger.error(f"API call failed: {service} {endpoint}", extra=log_data)
    else:
        logger.info(f"API call: {service} {endpoint}", extra=log_data)

def log_database_query(
    database: str,
    query: str,
    params: Optional[Union[tuple, dict]] = None,
    duration: Optional[float] = None,
    rows_affected: Optional[int] = None,
    error: Optional[str] = None
):
    """Loguje zapytanie do bazy danych"""
    logger = get_logger(f"database.{database}")
    
    # Skróć długie zapytania
    if len(query) > 200:
        query_preview = query[:200] + "..."
    else:
        query_preview = query
    
    log_data = {
        'database': database,
        'query': query_preview,
        'params': params,
        'duration': duration,
        'rows_affected': rows_affected,
        'error': error
    }
    
    if error:
        logger.error(f"Database query failed: {database}", extra=log_data)
    else:
        logger.debug(f"Database query: {database}", extra=log_data)

def log_document_processing(
    document_id: str,
    task_id: str,
    action: str,
    status: str,
    duration: Optional[float] = None,
    metadata: Optional[Dict] = None
):
    """Loguje przetwarzanie dokumentu"""
    logger = get_logger("processors")
    
    log_data = {
        'document_id': document_id,
        'task_id': task_id,
        'action': action,
        'status': status,
        'duration': duration,
        'metadata': metadata or {}
    }
    
    if status == "error":
        logger.error(f"Document processing failed: {action}", extra=log_data)
    else:
        logger.info(f"Document processing: {action} - {status}", extra=log_data)

def log_agent_action(
    agent_name: str,
    action: str,
    question: Optional[str] = None,
    result: Optional[str] = None,
    confidence: Optional[float] = None,
    duration: Optional[float] = None
):
    """Loguje akcję agenta"""
    logger = get_logger(f"agents.{agent_name}")
    
    log_data = {
        'agent': agent_name,
        'action': action,
        'question': question,
        'result_preview': result[:100] if result else None,
        'confidence': confidence,
        'duration': duration
    }
    
    logger.info(f"Agent action: {agent_name} - {action}", extra=log_data)

# ============== ANALIZA LOGÓW ==============

@dataclass
class LogAnalytics:
    """Klasa do analizy logów"""
    
    @staticmethod
    def parse_log_file(log_file: Path, format: str = "json") -> list:
        """Parsuje plik logów"""
        logs = []
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    if format == "json":
                        logs.append(json.loads(line))
                    else:
                        # Parsowanie logów tekstowych
                        # TODO: Implementacja parsowania
                        pass
                except Exception:
                    continue
        
        return logs
    
    @staticmethod
    def analyze_performance(logs: list) -> Dict[str, Any]:
        """Analizuje wydajność na podstawie logów"""
        operations = defaultdict(list)
        
        for log in logs:
            if 'duration' in log:
                operation = log.get('function', 'unknown')
                operations[operation].append(log['duration'])
        
        # Oblicz statystyki
        stats = {}
        for operation, durations in operations.items():
            if durations:
                stats[operation] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'average': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        
        return stats
    
    @staticmethod
    def find_errors(logs: list) -> list:
        """Znajduje błędy w logach"""
        errors = []
        
        for log in logs:
            if log.get('level') in ['ERROR', 'CRITICAL']:
                errors.append({
                    'timestamp': log['timestamp'],
                    'message': log['message'],
                    'exception': log.get('exception'),
                    'context': {
                        k: v for k, v in log.items() 
                        if k not in ['timestamp', 'message', 'exception']
                    }
                })
        
        return errors

# ============== EKSPORT ==============

__all__ = [
    'setup_logging',
    'get_logger',
    'log_execution',
    'log_async_execution',
    'log_context',
    'log_operation',
    'log_api_call',
    'log_database_query',
    'log_document_processing',
    'log_agent_action',
    'ContextLogger',
    'LogAnalytics',
    'ColoredFormatter',
    'JSONFormatter',
    'StructuredFormatter',
    'DatabaseHandler',
    'MetricsHandler',
    'TaskFilter',
    'LevelFilter',
    'RateLimitFilter'
]

# ============== INICJALIZACJA ==============

# Automatyczna konfiguracja przy imporcie
if os.getenv('AUTO_SETUP_LOGGING', 'true').lower() == 'true':
    setup_logging()
