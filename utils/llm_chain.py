def classify_intent_keywords(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["cover", "coverage", "covered"]):
        return "coverage"
    if any(w in q for w in ["exclusion", "not covered", "excluded"]):
        return "exclusions"
    if any(w in q for w in ["limit", "maximum", "cap"]):
        return "limits"
    if any(w in q for w in ["premium", "payment", "pay", "installment"]):
        return "premium"
    if any(w in q for w in ["claim", "settlement", "reimbursement"]):
        return "claim"
    if any(w in q for w in ["renewal", "renew"]):
        return "renewal"
    if any(w in q for w in ["grace period", "late payment"]):
        return "grace_period"
    return "general"
def screen_chunks_by_keywords(chunks, question, intent, min_k=5):
    """
    Filter chunks by intent-specific keywords. If not enough, fallback to top-k.
    """
    intent_keywords = {
        "coverage": ["cover", "coverage", "covered"],
        "exclusions": ["exclusion", "not covered", "excluded"],
        "limits": ["limit", "maximum", "cap", "sum insured", "amount", "rupees", "lakh", "lakhs", "rs."],
        "premium": ["premium", "payment", "pay", "installment", "amount"],
        "claim": ["claim", "settlement", "reimbursement", "file a claim"],
        "renewal": ["renewal", "renew", "expiry", "expire"],
        "grace_period": ["grace period", "late payment", "days after", "within", "after due date"],
        "general": []
    }
    keywords = intent_keywords.get(intent, [])
    def chunk_text(chunk):
        if hasattr(chunk, 'page_content'):
            return chunk.page_content.lower()
        return str(chunk).lower()
    if not keywords:
        return chunks[:min_k]
    filtered = [c for c in chunks if any(kw in chunk_text(c) for kw in keywords)]
    if len(filtered) >= min_k:
        return filtered[:min_k]
    # If not enough, pad with top-k
    seen = set(id(c) for c in filtered)
    for c in chunks:
        if id(c) not in seen:
            filtered.append(c)
        if len(filtered) >= min_k:
            break
    return filtered[:min_k]
#llm_chain.py
"""
LLM Chain utility for processing queries with batch support using LangChain
"""

import asyncio
import os
import time
from typing import List, Dict, Any, Optional
import aiohttp
import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging
import json

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# OpenRouter configuration with optimized settings
#paid api key

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
# OPENROUTER_MODEL = "google/gemini-2.0-flash-001"


# Free API key for testing
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Use faster models for better response time
# OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"  # Faster experimental model
# OPENROUTER_MODEL = "anthropic/claude-3-haiku-20240307"  # Fast alternative

OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set. Please set it in your environment or .env file.")

# Optimized system prompt (shorter for faster processing)
SYSTEM_PROMPT = """You are an expert insurance policy AI assistant. Use ONLY the provided clauses (each clause is a discrete point from the document) to answer questions. 
If information isn't in the clauses, state: 'Information not available in document.
' Do NOT use any outside knowledge. Be concise, accurate, and professional."""

# Global session for connection pooling
_session: Optional[aiohttp.ClientSession] = None

async def get_session() -> aiohttp.ClientSession:
    """Get or create a global aiohttp session with optimized settings"""
    global _session
    if _session is None or _session.closed:
        # Optimized connector settings for faster requests
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=20,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,  # Keep connections alive
            enable_cleanup_closed=True
        )
        
        # Optimized timeout settings
        timeout = aiohttp.ClientTimeout(
            total=20,  # Reduced from 30s
            connect=5,   # Connection timeout
            sock_read=15  # Socket read timeout
        )
        
        _session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Policy RAG System"
            }
        )
    return _session

async def cleanup_session():
    """Clean up the global session"""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None

async def call_openrouter_api_async(prompt: str) -> str:
    """
    Optimized async function to call OpenRouter API with connection pooling
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        str: The response from the LLM
    """
    api_start = time.time()
    
    try:
        session = await get_session()
        
        # Optimized payload for faster processing
        data = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 75,  # Increased slightly for better answers
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False,  # Explicitly disable streaming for batch processing
        }
        
        request_start = time.time()
        async with session.post(OPENROUTER_API_URL, json=data) as response:
            request_time = time.time() - request_start
            
            logger.debug(f"🌐 Async API call took {request_time:.3f}s - Status: {response.status}")
            
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API Error {response.status}: {error_text}")
                return f"Error: API returned status {response.status}"
            
            parse_start = time.time()
            result = await response.json()
            parse_time = time.time() - parse_start
            
            # Extract the generated text from the response
            try:
                extract_start = time.time()
                answer = result["choices"][0]["message"]["content"].strip()
                extract_time = time.time() - extract_start
                
                api_total = time.time() - api_start
                logger.debug(f"🔧 Async API breakdown - Request: {request_time:.3f}s, Parse: {parse_time:.3f}s, Extract: {extract_time:.3f}s, Total: {api_total:.3f}s")
                
                return answer
            except (KeyError, IndexError) as e:
                api_total = time.time() - api_start
                logger.error(f"❌ Error extracting response after {api_total:.3f}s: {e}")
                logger.error(f"Full response: {result}")
                return f"Error: Could not parse response - {str(result)}"
                
    except asyncio.TimeoutError:
        api_total = time.time() - api_start
        logger.error(f"❌ Request timeout after {api_total:.3f}s")
        return "Error: Request timeout"
    except Exception as e:
        api_total = time.time() - api_start
        logger.error(f"❌ Unexpected error after {api_total:.3f}s: {e}")
        return f"Error: {str(e)}"

def call_openrouter_api(prompt: str) -> str:
    """
    Synchronous wrapper for backward compatibility
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        str: The response from the LLM
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an async context, create a new task
            return asyncio.create_task(call_openrouter_api_async(prompt))
        else:
            # Run in new event loop
            return loop.run_until_complete(call_openrouter_api_async(prompt))
    except Exception:
        # Fallback to thread pool execution
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, call_openrouter_api_async(prompt))
            return future.result()

async def process_chunk_with_llm_async(prompt: str) -> str:
    """
    Optimized asynchronous LLM processing with direct async API calls
    
    Args:
        prompt (str): The prompt to process
        
    Returns:
        str: The LLM response
    """
    async_start = time.time()
    
    try:
        # Direct async call without thread pool overhead
        response = await call_openrouter_api_async(prompt)
        async_total = time.time() - async_start
        logger.debug(f"🔄 Optimized async LLM processing completed in {async_total:.3f}s")
        return response
    except Exception as e:
        async_total = time.time() - async_start
        logger.error(f"❌ Error in optimized async LLM processing after {async_total:.3f}s: {e}")
        return f"Error: {str(e)}"

async def process_batches(tasks: List[asyncio.Task], batch_size: int = 5) -> List[str]:
    """
    Optimized batch processing with adaptive concurrency and connection pooling
    
    Args:
        tasks (List[asyncio.Task]): List of async tasks to process
        batch_size (int): Number of tasks to process concurrently (increased default)
        
    Returns:
        List[str]: List of responses
    """
    batch_start = time.time()
    results = []
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    
    logger.info(f"📊 Processing {len(tasks)} tasks in {total_batches} batches of size {batch_size}")
    
    # Process all batches with optimized concurrency
    semaphore = asyncio.Semaphore(batch_size * 2)  # Allow more concurrent connections
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await task
    
    for i in range(0, len(tasks), batch_size):
        batch_num = (i // batch_size) + 1
        batch = tasks[i:i + batch_size]
        
        batch_iteration_start = time.time()
        logger.info(f"🔄 Processing batch {batch_num}/{total_batches} with {len(batch)} tasks")
        
        try:
            # Process batch with semaphore control
            gather_start = time.time()
            semaphore_tasks = [process_with_semaphore(task) for task in batch]
            batch_results = await asyncio.gather(*semaphore_tasks, return_exceptions=True)
            gather_time = time.time() - gather_start
            
            # Handle results and exceptions
            process_start = time.time()
            processed_results = []
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_msg = f"Error in task {i + j + 1}: {str(result)}"
                    logger.error(error_msg)
                    processed_results.append(error_msg)
                else:
                    processed_results.append(result)
            
            process_time = time.time() - process_start
            batch_iteration_time = time.time() - batch_iteration_start
            
            logger.info(f"✅ Batch {batch_num} completed in {batch_iteration_time:.3f}s (gather: {gather_time:.3f}s, process: {process_time:.3f}s)")
            
            results.extend(processed_results)
            
            # Reduced delay between batches for faster processing
            if i + batch_size < len(tasks):
                delay = 0.5  # Reduced from 1.0s
                logger.debug(f"⏱️ Waiting {delay}s before next batch...")
                await asyncio.sleep(delay)
                
        except Exception as e:
            batch_iteration_time = time.time() - batch_iteration_start
            error_msg = f"Batch {batch_num} failed after {batch_iteration_time:.3f}s: {str(e)}"
            logger.error(error_msg)
            results.extend([error_msg] * len(batch))
    
    total_time = time.time() - batch_start
    logger.info(f"🏁 Completed processing all {len(tasks)} tasks in {total_time:.3f}s")
    return results

class LangChainLLMProcessor:
    """
    Optimized LangChain-based LLM processor with async capabilities
    """
    
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["document", "question"],
            template="Context: {document}\n\nQuestion: {question}\n\nAnswer:"  # Simplified template for faster processing
        )
    
    async def _format_and_call_llm_async(self, input_data: Dict[str, Any]) -> str:
        """Format prompt and call LLM asynchronously"""
        formatted_prompt = self.prompt_template.format(**input_data)
        return await call_openrouter_api_async(formatted_prompt)
    
    def _format_and_call_llm(self, formatted_prompt: str) -> str:
        """Synchronous format prompt and call LLM (backward compatibility)"""
        return call_openrouter_api(formatted_prompt.text)

    async def batch_process(self, documents: List[str], questions: List[str]) -> List[str]:
        """
        Optimized batch processing with improved concurrency and connection pooling
        
        Args:
            documents (List[str]): List of document contexts
            questions (List[str]): List of questions to answer
            
        Returns:
            List[str]: List of answers
        """
        langchain_start = time.time()
        
        # Prepare inputs for batch processing
        prep_start = time.time()
        document_context = " ".join(documents) if isinstance(documents, list) else documents
        
        # For each question, classify intent and screen chunks by keywords
        if isinstance(document_context, str):
            raw_chunks = [c.strip() for c in document_context.split('\n\n') if c.strip()]
        else:
            raw_chunks = document_context
        inputs = []
        for question in questions:
            # Classify intent (simple keyword-based for now)
            intent = classify_intent_keywords(question)
            top_chunks = screen_chunks_by_keywords(raw_chunks, question, intent, min_k=5)
            context = '\n\n'.join(chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in top_chunks)
            inputs.append({"document": context, "question": question})
        prep_time = time.time() - prep_start
        
        try:
            logger.info(f"🔗 Starting optimized batch processing for {len(inputs)} questions")
            logger.debug(f"📋 Input preparation took {prep_time:.3f}s")
            
            # Create all tasks at once for better parallelization
            task_creation_start = time.time()
            tasks = [
                asyncio.create_task(self._format_and_call_llm_async(input_data))
                for input_data in inputs
            ]
            task_creation_time = time.time() - task_creation_start
            
            # Use optimized batch processing with higher concurrency
            processing_start = time.time()
            batch_size = 8  # Increased batch size for better throughput
            results = await process_batches(tasks, batch_size)
            processing_time = time.time() - processing_start
            
            total_time = time.time() - langchain_start
            
            logger.info(f"🏁 Optimized batch processing completed in {total_time:.3f}s")
            logger.info(f"   📊 Preparation: {prep_time:.3f}s, Task creation: {task_creation_time:.3f}s, Processing: {processing_time:.3f}s")
            logger.info(f"   ⚡ Average per question: {total_time/len(questions):.3f}s")
            
            return results
            
        except Exception as e:
            total_time = time.time() - langchain_start
            logger.error(f"❌ Error in optimized batch processing after {total_time:.3f}s: {e}")
            return [f"Error: {str(e)}"] * len(questions)
        finally:
            # Clean up session if this is the last batch
            pass

    async def _process_single_async(self, input_data: dict) -> str:
        """Process a single input asynchronously with optimized timing"""
        single_start = time.time()
        
        try:
            # Direct async processing without format overhead
            result = await self._format_and_call_llm_async(input_data)
            
            single_total = time.time() - single_start
            logger.debug(f"🔧 Optimized single processing completed in {single_total:.3f}s")
            
            return result
        except Exception as e:
            single_total = time.time() - single_start
            logger.error(f"❌ Error processing single question after {single_total:.3f}s: {str(e)}")
            return f"Error processing question: {str(e)}"

# Global instance with optimized settings
llm_processor = LangChainLLMProcessor()

# Utility functions for performance optimization
async def warm_up_session():
    """Warm up the session by making a test request"""
    try:
        await call_openrouter_api_async("Test connection")
        logger.info("🔥 Session warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ Session warm-up failed: {e}")

async def cleanup_resources():
    """Clean up all resources before shutdown"""
    await cleanup_session()
    logger.info("🧹 Resources cleaned up")

# Context manager for session lifecycle
class OptimizedLLMContext:
    """Context manager for optimized LLM processing with automatic cleanup"""
    
    async def __aenter__(self):
        await warm_up_session()
        return llm_processor
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await cleanup_resources()

# Fast single query function
async def quick_query(question: str, context: str) -> str:
    """
    Optimized single query processing
    
    Args:
        question (str): The question to ask
        context (str): The document context
        
    Returns:
        str: The answer
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    return await call_openrouter_api_async(prompt)

# Batch query function with automatic optimization
async def batch_query(questions: List[str], context: str, batch_size: int = 8) -> List[str]:
    """
    Optimized batch query processing
    
    Args:
        questions (List[str]): List of questions
        context (str): The document context
        batch_size (int): Batch size for processing
        
    Returns:
        List[str]: List of answers
    """
    async with OptimizedLLMContext() as processor:
        return await processor.batch_process([context], questions)
