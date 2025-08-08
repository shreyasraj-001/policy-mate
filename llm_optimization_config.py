"""
Optimization configuration for LLM processing
"""

# API Configuration
OPTIMIZATION_CONFIG = {
    # Connection Settings
    "connection_pool_size": 100,
    "connections_per_host": 20,
    "dns_cache_ttl": 300,
    "keepalive_timeout": 30,
    
    # Timeout Settings (in seconds)
    "total_timeout": 20,      # Reduced from 30s
    "connect_timeout": 5,     # Connection timeout
    "socket_read_timeout": 15, # Socket read timeout
    
    # Batch Processing
    "default_batch_size": 8,   # Increased from 3
    "max_concurrent_requests": 16,  # Semaphore limit
    "batch_delay": 0.5,       # Reduced from 1.0s
    
    # Model Settings
    "max_tokens": 300,        # Increased slightly
    "temperature": 0.1,       # Low for consistency
    "top_p": 0.9,
    
    # Performance Flags
    "enable_connection_pooling": True,
    "enable_dns_caching": True,
    "enable_compression": True,
    "enable_session_warming": True,
    
    # Model Selection (ordered by speed)
    "fast_models": [
        "google/gemini-2.0-flash-thinking-exp",  # Fastest experimental
        "google/gemini-2.0-flash-001",           # Fast production
        "anthropic/claude-3-haiku-20240307",     # Fast alternative
        "openai/gpt-4.1-nano"                    # Compact option
    ],
    
    # Fallback Models
    "fallback_models": [
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3-haiku-20240307"
    ]
}

# Environment-specific overrides
ENVIRONMENT_OVERRIDES = {
    "development": {
        "total_timeout": 30,
        "batch_delay": 1.0,
        "default_batch_size": 3
    },
    "production": {
        "total_timeout": 15,
        "batch_delay": 0.3,
        "default_batch_size": 10
    },
    "testing": {
        "total_timeout": 10,
        "batch_delay": 0.1,
        "default_batch_size": 5
    }
}

def get_config(environment: str = "production"):
    """Get configuration for specific environment"""
    config = OPTIMIZATION_CONFIG.copy()
    if environment in ENVIRONMENT_OVERRIDES:
        config.update(ENVIRONMENT_OVERRIDES[environment])
    return config
