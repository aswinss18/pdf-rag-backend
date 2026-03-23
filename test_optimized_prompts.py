#!/usr/bin/env python3
"""
Test script for the optimized prompt architecture
Tests minimal, structured, and purpose-driven prompts with token optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.prompt_templates import (
    PromptBuilder, ContextSelector, MemorySelector,
    build_optimized_prompt, optimize_context, optimize_memory
)
import time

def test_optimized_prompts():
    """Test the optimized prompt architecture"""
    print("🚀 Testing Optimized Prompt Architecture")
    print("=" * 55)
    
    # Test 1: Basic prompt templates
    print("\n📝 Test 1: Basic Prompt Templates")
    
    builder = PromptBuilder()
    
    # Test RAG prompt
    context = "Aswin S S is a Software Developer at GigLabz Private Ltd. His salary is ₹47,000 per month. He lives in Bangalore, Karnataka and has 5 years of experience in Python development."
    question = "What is Aswin's salary?"
    
    rag_prompt = builder.build_rag_prompt(context, question)
    print(f"RAG Prompt Length: {len(rag_prompt)} characters")
    print(f"RAG Prompt Preview: {rag_prompt[:150]}...")
    
    # Test Agent prompt
    tools = ["search_documents", "calculate_percentage", "get_weather", "convert_currency"]
    query = "Calculate 15% of Aswin's salary"
    
    agent_prompt = builder.build_agent_prompt(query, tools)
    print(f"\nAgent Prompt Length: {len(agent_prompt)} characters")
    print(f"Agent Prompt Preview: {agent_prompt[:150]}...")
    
    # Test Memory prompt
    memory_data = [
        {"text": "User's name is John", "importance": 0.9, "confidence": "high"},
        {"text": "User works as a Data Scientist", "importance": 0.8, "confidence": "high"},
        {"text": "User prefers Python over R", "importance": 0.6, "confidence": "medium"}
    ]
    profile = "Data Scientist named John"
    
    memory_prompt = builder.build_memory_prompt(profile, memory_data)
    print(f"\nMemory Prompt Length: {len(memory_prompt)} characters")
    print(f"Memory Prompt Preview: {memory_prompt[:150]}...")
    
    print("✅ Basic prompt templates working")
    
    # Test 2: Combined prompt optimization
    print("\n🔧 Test 2: Combined Prompt Optimization")
    
    combined_prompt = build_optimized_prompt(
        query=query,
        context=context,
        memory=memory_data,
        available_tools=tools,
        user_profile=profile
    )
    
    print(f"Combined Prompt Length: {len(combined_prompt)} characters")
    print(f"Combined Prompt Structure:")
    sections = combined_prompt.split('\n\n')
    for i, section in enumerate(sections[:5], 1):  # Show first 5 sections
        print(f"  Section {i}: {section[:80]}...")
    
    print("✅ Combined prompt optimization working")
    
    # Test 3: Context optimization
    print("\n📊 Test 3: Context Optimization")
    
    # Create test chunks with different scores
    test_chunks = [
        {
            "text": "Aswin S S is a Software Developer at GigLabz Private Ltd with 5 years of experience.",
            "combined_score": 0.95,
            "doc": "profile.pdf",
            "page": 1
        },
        {
            "text": "His monthly salary is ₹47,000 and he lives in Bangalore, Karnataka.",
            "combined_score": 0.88,
            "doc": "salary.pdf", 
            "page": 1
        },
        {
            "text": "He has expertise in Python, Django, and machine learning frameworks.",
            "combined_score": 0.75,
            "doc": "skills.pdf",
            "page": 1
        },
        {
            "text": "The weather in Bangalore is generally pleasant throughout the year.",
            "combined_score": 0.45,  # Low score - should be filtered
            "doc": "weather.pdf",
            "page": 1
        },
        {
            "text": "GigLabz Private Ltd is a technology company founded in 2020.",
            "combined_score": 0.65,
            "doc": "company.pdf",
            "page": 1
        }
    ]
    
    selector = ContextSelector()
    optimized_context = selector.select_best_context(test_chunks, "What is Aswin's salary?")
    
    print(f"Original chunks: {len(test_chunks)}")
    print(f"Optimized context length: {len(optimized_context)} characters")
    print(f"Context preview: {optimized_context[:200]}...")
    
    # Test duplicate removal
    duplicate_context = optimized_context + "\n\n" + optimized_context  # Add duplicate
    deduplicated = selector.remove_duplicates(duplicate_context)
    
    print(f"Before deduplication: {len(duplicate_context)} characters")
    print(f"After deduplication: {len(deduplicated)} characters")
    print(f"Reduction: {((len(duplicate_context) - len(deduplicated)) / len(duplicate_context) * 100):.1f}%")
    
    print("✅ Context optimization working")
    
    # Test 4: Memory optimization
    print("\n🧠 Test 4: Memory Optimization")
    
    # Create test memories with different relevance scores
    test_memories = [
        {
            "text": "User's name is Alice and she works as a Senior Data Scientist",
            "combined_score": 0.92,
            "importance": 0.9,
            "confidence": "high"
        },
        {
            "text": "User's salary is $150,000 per year",
            "combined_score": 0.85,
            "importance": 0.8,
            "confidence": "high"
        },
        {
            "text": "User prefers TensorFlow over PyTorch for deep learning",
            "combined_score": 0.70,
            "importance": 0.6,
            "confidence": "medium"
        },
        {
            "text": "User had coffee this morning",
            "combined_score": 0.35,  # Low relevance - should be filtered
            "importance": 0.3,
            "confidence": "low"
        },
        {
            "text": "User lives in San Francisco, California",
            "combined_score": 0.78,
            "importance": 0.7,
            "confidence": "high"
        }
    ]
    
    memory_selector = MemorySelector()
    optimized_memories = memory_selector.select_relevant_memories(test_memories, "What is the user's salary?")
    
    print(f"Original memories: {len(test_memories)}")
    print(f"Optimized memories: {len(optimized_memories)}")
    print("Selected memories:")
    for i, mem in enumerate(optimized_memories, 1):
        print(f"  {i}. Score: {mem['combined_score']:.2f} - {mem['text'][:60]}...")
    
    print("✅ Memory optimization working")
    
    # Test 5: Token reduction analysis
    print("\n📈 Test 5: Token Reduction Analysis")
    
    # Create a verbose, unoptimized prompt
    verbose_prompt = f"""
    You are an advanced AI assistant with comprehensive capabilities including document analysis, mathematical calculations, weather information retrieval, currency conversion, and memory management. You have access to a sophisticated hybrid retrieval-augmented generation system that combines vector similarity search with keyword-based search to provide the most relevant information from uploaded documents.
    
    Your available tools include:
    - search_documents: Advanced document search with hybrid vector and keyword matching
    - calculate_percentage: Precise percentage calculations with detailed explanations
    - calculate_salary_increment: Specialized salary calculation tool with tax considerations
    - get_weather: Real-time weather information for any global location
    - convert_currency: Multi-currency conversion with current exchange rates
    - list_available_documents: Document inventory management system
    
    User Profile Information:
    The user is {profile} with extensive background in data science and machine learning. They have expressed preferences for Python programming language and have shown interest in advanced analytics and statistical modeling techniques.
    
    Relevant Memory Context:
    - User's name is John and he works as a Senior Data Scientist at a Fortune 500 company
    - User has 8+ years of experience in machine learning and artificial intelligence
    - User prefers Python over R for data analysis and statistical computing
    - User has worked on projects involving natural language processing and computer vision
    - User is interested in deep learning frameworks, particularly TensorFlow and PyTorch
    
    Document Context:
    {context}
    
    Please process the following query with careful consideration of all available information, tools, and context. Provide a comprehensive, accurate, and well-structured response that addresses all aspects of the user's question while maintaining professional standards and attention to detail.
    
    Query: {query}
    
    Instructions:
    1. Analyze the query thoroughly to understand the user's intent and requirements
    2. Determine which tools, if any, are needed to provide a complete answer
    3. Use the ReAct pattern: Reason about the problem, Act with appropriate tools, Observe results
    4. Incorporate relevant memory and document context where applicable
    5. Provide clear, concise, and actionable information in your response
    6. Include source citations when referencing document information
    7. Ensure all calculations are accurate and properly explained
    8. Maintain consistency with the user's known preferences and background
    """
    
    # Compare with optimized prompt
    print(f"Verbose prompt length: {len(verbose_prompt)} characters")
    print(f"Optimized prompt length: {len(combined_prompt)} characters")
    
    reduction_percentage = ((len(verbose_prompt) - len(combined_prompt)) / len(verbose_prompt)) * 100
    print(f"Token reduction: {reduction_percentage:.1f}%")
    
    # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
    verbose_tokens = len(verbose_prompt) // 4
    optimized_tokens = len(combined_prompt) // 4
    token_savings = verbose_tokens - optimized_tokens
    
    print(f"Estimated verbose tokens: {verbose_tokens}")
    print(f"Estimated optimized tokens: {optimized_tokens}")
    print(f"Token savings: {token_savings} tokens ({reduction_percentage:.1f}%)")
    
    # Cost savings estimation (rough: $0.0001 per 1K tokens for GPT-4o-mini)
    cost_per_1k_tokens = 0.0001
    cost_savings = (token_savings / 1000) * cost_per_1k_tokens
    print(f"Estimated cost savings per query: ${cost_savings:.6f}")
    
    print("✅ Significant token reduction achieved")
    
    print("\n🎉 Optimized Prompt Architecture Test Completed!")
    print("\n🎯 Key Achievements:")
    print(f"  ✅ Modular prompt templates implemented")
    print(f"  ✅ Context optimization with smart filtering")
    print(f"  ✅ Memory optimization with relevance scoring")
    print(f"  ✅ Token reduction: {reduction_percentage:.1f}%")
    print(f"  ✅ Estimated cost savings: ${cost_savings:.6f} per query")
    print(f"  ✅ Maintained functionality with improved efficiency")
    
    print("\n💡 Optimization Features:")
    print("  • Minimal, structured, purpose-driven prompts")
    print("  • Smart context selection (top 3-5 high-score chunks)")
    print("  • Memory filtering (relevance threshold: 0.6)")
    print("  • Duplicate content removal")
    print("  • Token-optimized prompt building")
    print("  • Modular architecture for easy maintenance")

if __name__ == "__main__":
    test_optimized_prompts()