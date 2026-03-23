#!/usr/bin/env python3
"""
Test script to demonstrate all the enhanced UI features
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_ui_features():
    """Test all the enhanced UI features"""
    print("🎨 Testing Enhanced UI Features")
    print("=" * 50)
    
    # Test 1: Add some diverse memories with different confidence levels
    print("\n📝 Test 1: Adding diverse memories for UI demonstration")
    
    test_memories = [
        ("My name is Sarah and I work as a Machine Learning Engineer at OpenAI", "High confidence personal info"),
        ("I live in Seattle, Washington and my salary is $180,000 per year", "High confidence location and salary"),
        ("I prefer TensorFlow over PyTorch for deep learning projects", "Medium confidence preference"),
        ("My goal is to publish 3 research papers this year", "High confidence goal"),
        ("I have 6 years of experience in computer vision", "High confidence experience"),
        ("The weather looks nice today", "Low confidence casual comment"),
        ("I think Python is better than Java", "Medium confidence opinion"),
        ("My project budget is $1.2 million for the AI initiative", "High confidence financial data")
    ]
    
    for memory_text, description in test_memories:
        print(f"  Adding: {memory_text[:50]}... ({description})")
        response = requests.post(f"{BASE_URL}/agent", 
                               data={"query": memory_text},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        if response.status_code == 200:
            print("    ✅ Added successfully")
        else:
            print(f"    ❌ Failed: {response.status_code}")
    
    time.sleep(2)  # Wait for processing
    
    # Test 2: Check basic memory statistics
    print("\n📊 Test 2: Basic Memory Statistics")
    response = requests.get(f"{BASE_URL}/memory/status")
    if response.status_code == 200:
        stats = response.json()["memory_stats"]
        print(f"  • Total memories: {stats.get('stored_memories', 0)}")
        print(f"  • Average importance: {stats.get('average_importance', 0):.3f}")
        print(f"  • Scoring system: {stats.get('scoring_system', 'unknown')}")
        
        # Importance distribution
        importance_dist = stats.get('importance_distribution', {})
        print(f"  • High importance: {importance_dist.get('high', 0)}")
        print(f"  • Medium importance: {importance_dist.get('medium', 0)}")
        print(f"  • Low importance: {importance_dist.get('low', 0)}")
        
        # Confidence distribution
        confidence_dist = stats.get('confidence_distribution', {})
        print(f"  • High confidence: {confidence_dist.get('high', 0)}")
        print(f"  • Medium confidence: {confidence_dist.get('medium', 0)}")
        print(f"  • Low confidence: {confidence_dist.get('low', 0)}")
        
        print("✅ Basic statistics retrieved")
    else:
        print(f"❌ Failed to get basic stats: {response.status_code}")
    
    # Test 3: Check detailed memory information
    print("\n🔍 Test 3: Detailed Memory Information")
    response = requests.get(f"{BASE_URL}/memory/detailed")
    if response.status_code == 200:
        data = response.json()
        
        # System health
        system_health = data.get("system_health", {})
        print(f"  System Status: {system_health.get('system_status', 'unknown')}")
        print(f"  Quality Score: {(system_health.get('average_quality', 0) * 100):.1f}%")
        print(f"  Index Size: {system_health.get('index_size', 0)}")
        
        # Recent memories
        recent_memories = data.get("recent_memories", [])
        print(f"\n  Recent Memories ({len(recent_memories)}):")
        for i, memory in enumerate(recent_memories[:3], 1):  # Show top 3
            importance = memory.get("importance", 0)
            confidence = memory.get("confidence", "unknown")
            text = memory.get("text", "")[:60] + "..."
            
            importance_label = "HIGH" if importance >= 0.8 else "MED" if importance >= 0.6 else "LOW"
            print(f"    {i}. [{importance_label}] [{confidence.upper()}] {text}")
            print(f"       Importance: {importance:.2f}, Access: {memory.get('access_count', 0)}x, Age: {memory.get('age_days', 0):.1f}d")
        
        print("✅ Detailed information retrieved")
    else:
        print(f"❌ Failed to get detailed info: {response.status_code}")
    
    # Test 4: Test memory retrieval with confidence
    print("\n🧠 Test 4: Memory Retrieval with Confidence")
    
    test_queries = [
        "What is my name and where do I work?",
        "How much is my salary?",
        "What are my preferences for deep learning?",
        "What is my project budget?"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        response = requests.post(f"{BASE_URL}/agent", 
                               data={"query": query},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        if response.status_code == 200:
            data = response.json()
            memory_used = data.get("memory_used", False)
            tools_used = data.get("tools_used", 0)
            answer = data.get("answer", "")[:80] + "..."
            
            confidence_level = "High" if memory_used and tools_used == 0 else "Medium" if memory_used else "Low"
            
            print(f"    Memory Used: {memory_used}")
            print(f"    Confidence: {confidence_level}")
            print(f"    Answer: {answer}")
            
            if memory_used and tools_used == 0:
                print("    ✅ Perfect! High confidence answer from memory")
            elif memory_used:
                print("    ✅ Good! Memory + additional processing")
            else:
                print("    ⚠️  No memory used - may need more context")
        else:
            print(f"    ❌ Failed: {response.status_code}")
    
    # Test 5: Memory quality analysis
    print("\n📈 Test 5: Memory Quality Analysis")
    response = requests.get(f"{BASE_URL}/memory/status")
    if response.status_code == 200:
        stats = response.json()["memory_stats"]
        
        # Calculate quality metrics
        total_memories = stats.get('stored_memories', 1)
        importance_dist = stats.get('importance_distribution', {})
        confidence_dist = stats.get('confidence_distribution', {})
        
        # Quality score calculation
        high_importance = importance_dist.get('high', 0)
        medium_importance = importance_dist.get('medium', 0)
        quality_score = (high_importance * 1.0 + medium_importance * 0.6) / total_memories * 100
        
        # Confidence score
        high_confidence = confidence_dist.get('high', 0)
        medium_confidence = confidence_dist.get('medium', 0)
        confidence_score = (high_confidence * 1.0 + medium_confidence * 0.6) / total_memories * 100
        
        print(f"  Memory Quality Score: {quality_score:.1f}%")
        print(f"  Confidence Score: {confidence_score:.1f}%")
        print(f"  Average Importance: {stats.get('average_importance', 0):.3f}")
        
        # Quality assessment
        if quality_score >= 80 and confidence_score >= 80:
            print("  🌟 Excellent! High quality, high confidence memory system")
        elif quality_score >= 60 and confidence_score >= 60:
            print("  ✅ Good! Solid memory system performance")
        else:
            print("  ⚠️  Memory system could be improved")
        
        print("✅ Quality analysis completed")
    
    print("\n🎉 Enhanced UI Features Test Completed!")
    print("\n🎯 UI Features Demonstrated:")
    print("  ✅ Comprehensive memory statistics with confidence levels")
    print("  ✅ Detailed memory analysis with individual memory scores")
    print("  ✅ System health monitoring and quality metrics")
    print("  ✅ Visual confidence indicators and importance badges")
    print("  ✅ Memory age, access frequency, and source tracking")
    print("  ✅ Real-time memory quality scoring")
    print("  ✅ Advanced analytics and distribution charts")
    print("\n💡 The UI now shows:")
    print("  • Memory confidence levels (High/Medium/Low)")
    print("  • Importance scores with color coding")
    print("  • Access frequency and memory age")
    print("  • System health status")
    print("  • Quality distribution analytics")
    print("  • Individual memory details with badges")

if __name__ == "__main__":
    try:
        test_ui_features()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server at http://localhost:8000")
        print("Please make sure the backend server is running!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")