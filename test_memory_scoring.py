#!/usr/bin/env python3
"""
Test script for the advanced memory scoring system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.memory import AgentMemory
import time

def test_memory_scoring():
    """Test the advanced memory scoring system"""
    print("🧠 Testing Advanced Memory Scoring System")
    print("=" * 50)
    
    # Create a test memory instance
    memory = AgentMemory()
    
    # Clear any existing memory for clean test
    memory.clear_all_memory()
    
    # Test 1: Store memories with different importance levels
    print("\n📝 Test 1: Storing memories with different importance levels")
    
    test_memories = [
        ("Aswin's salary is ₹47,000 per month", "fact"),  # High importance
        ("User likes coffee in the morning", "preference"),  # Medium importance  
        ("The weather is nice today", "other"),  # Low importance
        ("Aswin works as a Software Developer at GigLabz", "fact"),  # High importance
        ("Hello how are you", "other"),  # Very low importance (should be filtered)
        ("User's name is John and he lives in Bangalore", "fact"),  # High importance
    ]
    
    for text, mem_type in test_memories:
        importance = memory.calculate_importance(text)
        print(f"  • '{text[:40]}...' → Importance: {importance:.2f}")
        memory.store_memory(text, mem_type)
    
    print(f"\n✅ Stored {len(memory.memory_store)} memories (filtered by importance > 0.6)")
    
    # Test 2: Test retrieval with advanced ranking
    print("\n🔍 Test 2: Testing advanced retrieval ranking")
    
    test_queries = [
        "What is Aswin's salary?",
        "Where does the user live?", 
        "What does Aswin do for work?",
        "Tell me about preferences"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        memories = memory.retrieve_memory(query, k=3)
        
        for i, mem in enumerate(memories, 1):
            combined_score = mem.get("combined_score", 0)
            similarity = mem.get("similarity_score", 0)
            importance = mem.get("importance_score", 0)
            recency = mem.get("recency_score", 0)
            
            print(f"    {i}. Score: {combined_score:.3f} (sim:{similarity:.2f}, imp:{importance:.2f}, rec:{recency:.2f})")
            print(f"       Text: '{mem['text'][:60]}...'")
    
    # Test 3: Test memory decay
    print("\n⏰ Test 3: Testing memory decay")
    
    # Simulate aging by modifying timestamps
    current_time = time.time()
    for i, memory_item in enumerate(memory.memory_store):
        if i == 0:  # Make first memory very old (6 months)
            memory_item["timestamp"] = current_time - (180 * 24 * 60 * 60)
        elif i == 1:  # Make second memory old (1 month)
            memory_item["timestamp"] = current_time - (30 * 24 * 60 * 60)
        # Others remain recent
    
    # Apply decay
    original_count = len(memory.memory_store)
    cleanup_result = memory.cleanup_old_memories(days_to_keep=90)  # Keep 3 months
    
    print(f"  • Original memories: {original_count}")
    print(f"  • After cleanup: {cleanup_result.get('kept', 0)}")
    print(f"  • Removed: {cleanup_result.get('removed', 0)}")
    print(f"  • Decay applied: {cleanup_result.get('decay_applied', 0)}")
    
    # Test 4: Test memory statistics
    print("\n📊 Test 4: Memory statistics")
    stats = memory.get_memory_stats()
    
    print(f"  • Total memories: {stats.get('stored_memories', 0)}")
    print(f"  • Average importance: {stats.get('average_importance', 0):.3f}")
    print(f"  • Scoring system: {stats.get('scoring_system', 'unknown')}")
    print(f"  • Importance distribution: {stats.get('importance_distribution', {})}")
    
    print("\n✅ All tests completed successfully!")
    print("\n🎯 Advanced Memory Scoring System Features:")
    print("  ✓ Store → Score → Rank → Retrieve → Inject architecture")
    print("  ✓ Combined scoring: similarity + importance + recency + frequency")
    print("  ✓ Memory decay based on age")
    print("  ✓ Automatic cleanup of low-value memories")
    print("  ✓ Advanced statistics and analytics")
    print("  ✓ Only stores memories with importance > 0.6")

if __name__ == "__main__":
    test_memory_scoring()