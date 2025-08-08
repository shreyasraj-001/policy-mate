"""
Test script to compare optimized vs original LLM processing performance
"""

import asyncio
import time
import logging
from utils.llm_chain import (
    OptimizedLLMContext, 
    quick_query, 
    batch_query,
    llm_processor
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_single_query_performance():
    """Test single query performance"""
    context = "This is a test insurance policy document with basic coverage information."
    question = "What type of coverage is provided?"
    
    # Test optimized single query
    start_time = time.time()
    answer = await quick_query(question, context)
    end_time = time.time()
    
    logger.info(f"🚀 Optimized single query took: {end_time - start_time:.3f}s")
    logger.info(f"📝 Answer: {answer[:100]}...")
    
    return end_time - start_time

async def test_batch_query_performance():
    """Test batch query performance"""
    context = "This is a test insurance policy document with comprehensive coverage including medical, dental, and vision benefits."
    
    questions = [
        "What types of coverage are included?",
        "Are dental benefits covered?",
        "What about vision coverage?",
        "Is medical coverage included?",
        "Are there any exclusions?"
    ]
    
    # Test optimized batch processing
    start_time = time.time()
    answers = await batch_query(questions, context, batch_size=8)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / len(questions)
    
    logger.info(f"🚀 Optimized batch query took: {total_time:.3f}s total")
    logger.info(f"⚡ Average per question: {avg_time:.3f}s")
    logger.info(f"📊 Processed {len(questions)} questions")
    
    for i, answer in enumerate(answers):
        logger.info(f"   Q{i+1}: {answer[:50]}...")
    
    return total_time, avg_time

async def test_context_manager():
    """Test the optimized context manager"""
    context = "Test policy document for performance testing."
    questions = ["What is this document about?", "Is this a test?"]
    
    start_time = time.time()
    async with OptimizedLLMContext() as processor:
        results = await processor.batch_process([context], questions)
    end_time = time.time()
    
    logger.info(f"🔧 Context manager test took: {end_time - start_time:.3f}s")
    logger.info(f"📝 Results: {len(results)} answers processed")
    
    return end_time - start_time

async def main():
    """Run all performance tests"""
    logger.info("🏁 Starting LLM performance tests...")
    
    try:
        # Test single query
        logger.info("\n" + "="*50)
        logger.info("🔍 Testing Single Query Performance")
        single_time = await test_single_query_performance()
        
        # Test batch query
        logger.info("\n" + "="*50)
        logger.info("📊 Testing Batch Query Performance")
        batch_time, avg_time = await test_batch_query_performance()
        
        # Test context manager
        logger.info("\n" + "="*50)
        logger.info("🔧 Testing Context Manager")
        context_time = await test_context_manager()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📈 PERFORMANCE SUMMARY")
        logger.info(f"   Single Query: {single_time:.3f}s")
        logger.info(f"   Batch Average: {avg_time:.3f}s per question")
        logger.info(f"   Context Manager: {context_time:.3f}s")
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
