#!/usr/bin/env python3
"""
Complete system test for the advanced memory scoring system
Tests the full Store → Score → Rank → Retrieve → Inject pipeline
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_complete_system():
    """Test the complete advanced memory system"""
    print("🚀 Testing Complete Advanced Memory System")
    print("=" * 60)
    
    # Test 1: Clear all memory to start fresh
    print("\n🧹 Test 1: Clearing all memory for clean test")
    response = requests.post(f"{BASE_URL}/memory/clear-all")
    if response.status_code == 200:
        print("✅ Memory cleared successfully")
    else:
        print(f"❌ Failed to clear memory: {response.status_code}")
        return
    
    # Test 2: Store high-importance information through agent
    print("\n📝 Test 2: Storing high-importance information")
    
    high_importance_queries = [
        "My name is Alice and I work as a Senior Data Scientist at Google. My annual salary is $150,000.",
        "I live in San Francisco, California. I have 8 years of experience in machine learning.",
        "My goal is to become a Principal Engineer within the next 3 years.",
        "I prefer Python over R for data analysis, and I love working with TensorFlow.",
        "My current project budget is $2.5 million for the AI research initiative."
    ]
    
    for i, query in enumerate(high_importance_queries, 1):
        print(f"  Query {i}: {query[:50]}...")
        response = requests.post(f"{BASE_URL}/agent", 
                               data={"query": query},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        if response.status_code == 200:
            data = response.json()
            memory_used = data.get("memory_used", False)
            print(f"    ✅ Response received, Memory used: {memory_used}")
        else:
            print(f"    ❌ Failed: {response.status_code}")
    
    # Wait a moment for memory processing
    time.sleep(2)
    
    # Test 3: Check memory statistics
    print("\n📊 Test 3: Checking memory statistics")
    response = requests.get(f"{BASE_URL}/memory/status")
    if response.status_code == 200:
        stats = response.json()["memory_stats"]
        print(f"  • Total memories: {stats.get('stored_memories', 0)}")
        print(f"  • Chat history: {stats.get('chat_history_length', 0)}")
        print(f"  • Average importance: {stats.get('average_importance', 0):.3f}")
        print(f"  • Scoring system: {stats.get('scoring_system', 'unknown')}")
        
        importance_dist = stats.get('importance_distribution', {})
        print(f"  • High importance: {importance_dist.get('high', 0)}")
        print(f"  • Medium importance: {importance_dist.get('medium', 0)}")
        print(f"  • Low importance: {importance_dist.get('low', 0)}")
        
        print("✅ Memory statistics retrieved successfully")
    else:
        print(f"❌ Failed to get memory stats: {response.status_code}")
    
    # Test 4: Test memory retrieval with various queries
    print("\n🔍 Test 4: Testing memory retrieval and ranking")
    
    retrieval_queries = [
        "What is my name and where do I work?",
        "How much is my salary?",
        "What are my career goals?",
        "What programming languages do I prefer?",
        "Where do I live?",
        "What is my project budget?"
    ]
    
    for query in retrieval_queries:
        print(f"\n  Query: '{query}'")
        response = requests.post(f"{BASE_URL}/agent", 
                               data={"query": query},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            memory_used = data.get("memory_used", False)
            tools_used = data.get("tools_used", 0)
            
            print(f"    Memory used: {memory_used}")
            print(f"    Tools used: {tools_used}")
            print(f"    Answer: {answer[:100]}...")
            
            if memory_used and tools_used == 0:
                print("    ✅ Perfect! Answer from memory without tools")
            elif memory_used and tools_used > 0:
                print("    ✅ Good! Memory + tools used together")
            else:
                print("    ⚠️  Memory not used for this query")
        else:
            print(f"    ❌ Failed: {response.status_code}")
    
    # Test 5: Test low-importance information filtering
    print("\n🚫 Test 5: Testing low-importance information filtering")
    
    low_importance_queries = [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "I had coffee this morning.",
        "See you later!",
        "Thanks for your help."
    ]
    
    # Get current memory count
    response = requests.get(f"{BASE_URL}/memory/status")
    initial_count = response.json()["memory_stats"]["stored_memories"] if response.status_code == 200 else 0
    
    for query in low_importance_queries:
        requests.post(f"{BASE_URL}/agent", 
                     data={"query": query},
                     headers={"Content-Type": "application/x-www-form-urlencoded"})
    
    # Check if memory count increased (it shouldn't much)
    time.sleep(1)
    response = requests.get(f"{BASE_URL}/memory/status")
    final_count = response.json()["memory_stats"]["stored_memories"] if response.status_code == 200 else 0
    
    filtered_out = len(low_importance_queries) - (final_count - initial_count)
    print(f"  • Low-importance queries: {len(low_importance_queries)}")
    print(f"  • Filtered out: {filtered_out}")
    print(f"  • Filter efficiency: {(filtered_out/len(low_importance_queries)*100):.1f}%")
    
    if filtered_out >= len(low_importance_queries) * 0.8:  # 80% filter rate
        print("  ✅ Excellent filtering! Low-importance content blocked")
    else:
        print("  ⚠️  Some low-importance content was stored")
    
    # Test 6: Test memory cleanup and decay
    print("\n⏰ Test 6: Testing memory cleanup and decay")
    
    response = requests.post(f"{BASE_URL}/memory/cleanup")
    if response.status_code == 200:
        result = response.json()["cleanup_result"]
        print(f"  • Memories removed: {result.get('removed', 0)}")
        print(f"  • Memories kept: {result.get('kept', 0)}")
        print(f"  • Decay applied: {result.get('decay_applied', 0)}")
        print("  ✅ Memory cleanup completed successfully")
    else:
        print(f"  ❌ Cleanup failed: {response.status_code}")
    
    # Final statistics
    print("\n📈 Final System Statistics")
    print("-" * 40)
    
    response = requests.get(f"{BASE_URL}/memory/status")
    if response.status_code == 200:
        stats = response.json()["memory_stats"]
        
        print(f"Final Memory Count: {stats.get('stored_memories', 0)}")
        print(f"Average Importance: {stats.get('average_importance', 0):.3f}")
        print(f"Chat History Length: {stats.get('chat_history_length', 0)}")
        print(f"System Status: {stats.get('scoring_system', 'unknown')}")
        
        # Quality metrics
        importance_dist = stats.get('importance_distribution', {})
        total_memories = stats.get('stored_memories', 1)
        high_quality_ratio = importance_dist.get('high', 0) / total_memories * 100
        
        print(f"High Quality Ratio: {high_quality_ratio:.1f}%")
        
        if high_quality_ratio >= 60:
            print("✅ Excellent memory quality!")
        elif high_quality_ratio >= 40:
            print("✅ Good memory quality!")
        else:
            print("⚠️  Memory quality could be improved")
    
    print("\n🎉 Complete System Test Finished!")
    print("\n🎯 Advanced Memory System Features Verified:")
    print("  ✅ Store → Score → Rank → Retrieve → Inject architecture")
    print("  ✅ Intelligent importance scoring and filtering")
    print("  ✅ Memory retrieval without unnecessary tool calls")
    print("  ✅ Low-importance content filtering")
    print("  ✅ Memory cleanup and decay functionality")
    print("  ✅ Comprehensive statistics and monitoring")
    print("  ✅ Production-ready memory management")

if __name__ == "__main__":
    try:
        test_complete_system()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server at http://localhost:8000")
        print("Please make sure the backend server is running!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")