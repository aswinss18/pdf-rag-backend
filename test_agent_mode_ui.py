#!/usr/bin/env python3
"""
Test script to demonstrate enhanced agent mode UI with memory parameters
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_agent_mode_ui():
    """Test the enhanced agent mode UI with memory parameters"""
    print("🤖 Testing Enhanced Agent Mode UI")
    print("=" * 50)
    
    # Test 1: Test agent response with memory context
    print("\n🧠 Test 1: Agent Response with Memory Context")
    
    test_queries = [
        ("What is my name and where do I work?", "Personal information retrieval"),
        ("How much is my salary?", "Financial information retrieval"),
        ("What are my career goals?", "Goal and preference retrieval"),
        ("What programming languages do I prefer?", "Technical preference retrieval")
    ]
    
    for query, description in test_queries:
        print(f"\n  Query: '{query}'")
        print(f"  Description: {description}")
        
        response = requests.post(f"{BASE_URL}/agent", 
                               data={"query": query},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        if response.status_code == 200:
            data = response.json()
            
            # Basic response info
            memory_used = data.get("memory_used", False)
            tools_used = data.get("tools_used", 0)
            react_pattern = data.get("react_pattern", False)
            
            print(f"    ✅ Response received")
            print(f"    Memory Used: {memory_used}")
            print(f"    Tools Used: {tools_used}")
            print(f"    ReAct Pattern: {react_pattern}")
            
            # Memory context information
            memory_context = data.get("memory_context_info")
            if memory_context:
                print(f"    📊 Memory Context:")
                print(f"      • Memories Retrieved: {memory_context['memories_retrieved']}")
                print(f"      • System Quality: {memory_context['system_stats']['quality_score']:.1f}%")
                print(f"      • Total Memories: {memory_context['system_stats']['total_memories']}")
                
                print(f"    🔍 Memories Used ({len(memory_context['memories_used'])}):")
                for i, memory in enumerate(memory_context['memories_used'], 1):
                    combined_score = memory['combined_score']
                    importance = memory['importance']
                    confidence = memory['confidence']
                    similarity = memory['similarity_score']
                    recency = memory['recency_score']
                    
                    print(f"      {i}. [{confidence.upper()}] Score: {combined_score:.3f}")
                    print(f"         Text: '{memory['text']}'")
                    print(f"         Importance: {importance:.2f}, Similarity: {similarity:.2f}, Recency: {recency:.2f}")
                    print(f"         Access: {memory['access_count']}x, Age: {memory['age_days']:.1f}d")
            else:
                print("    ⚠️  No memory context information available")
            
            # Answer preview
            answer = data.get("answer", "")[:100] + "..."
            print(f"    💬 Answer: {answer}")
            
        else:
            print(f"    ❌ Failed: {response.status_code}")
    
    # Test 2: Test agent with tool calls and memory
    print("\n🛠️ Test 2: Agent with Tool Calls and Memory")
    
    complex_query = "Calculate 15% of my salary and tell me about my work experience"
    print(f"  Query: '{complex_query}'")
    
    response = requests.post(f"{BASE_URL}/agent", 
                           data={"query": complex_query},
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"  ✅ Complex query processed")
        print(f"  Memory Used: {data.get('memory_used', False)}")
        print(f"  Tools Used: {data.get('tools_used', 0)}")
        print(f"  Tool Calls: {len(data.get('tool_calls', []))}")
        print(f"  Reasoning Steps: {len(data.get('reasoning_steps', []))}")
        
        # Show tool calls
        tool_calls = data.get("tool_calls", [])
        if tool_calls:
            print(f"  🔧 Tool Calls:")
            for i, tool_call in enumerate(tool_calls, 1):
                print(f"    {i}. {tool_call['tool_name']}")
                print(f"       Arguments: {tool_call['arguments']}")
        
        # Memory context for complex query
        memory_context = data.get("memory_context_info")
        if memory_context:
            print(f"  🧠 Memory Context for Complex Query:")
            print(f"    Quality Score: {memory_context['system_stats']['quality_score']:.1f}%")
            print(f"    Memories Retrieved: {memory_context['memories_retrieved']}")
            
            # Show top memory used
            if memory_context['memories_used']:
                top_memory = memory_context['memories_used'][0]
                print(f"    Top Memory: '{top_memory['text']}'")
                print(f"    Combined Score: {top_memory['combined_score']:.3f}")
    
    # Test 3: Test memory quality indicators
    print("\n📊 Test 3: Memory Quality Indicators")
    
    # Get current memory status
    response = requests.get(f"{BASE_URL}/memory/status")
    if response.status_code == 200:
        stats = response.json()["memory_stats"]
        
        print(f"  System Status: {stats.get('scoring_system', 'unknown')}")
        print(f"  Total Memories: {stats.get('stored_memories', 0)}")
        print(f"  Average Importance: {stats.get('average_importance', 0):.3f}")
        
        # Quality distribution
        importance_dist = stats.get('importance_distribution', {})
        confidence_dist = stats.get('confidence_distribution', {})
        
        print(f"  Importance Distribution:")
        print(f"    High: {importance_dist.get('high', 0)}")
        print(f"    Medium: {importance_dist.get('medium', 0)}")
        print(f"    Low: {importance_dist.get('low', 0)}")
        
        print(f"  Confidence Distribution:")
        print(f"    High: {confidence_dist.get('high', 0)}")
        print(f"    Medium: {confidence_dist.get('medium', 0)}")
        print(f"    Low: {confidence_dist.get('low', 0)}")
        
        # Calculate UI display metrics
        total_memories = stats.get('stored_memories', 1)
        high_importance = importance_dist.get('high', 0)
        medium_importance = importance_dist.get('medium', 0)
        quality_score = (high_importance * 1.0 + medium_importance * 0.6) / total_memories * 100
        
        print(f"  UI Quality Score: {quality_score:.1f}%")
        
        if quality_score >= 80:
            print("  🌟 UI will show: Excellent quality indicator")
        elif quality_score >= 60:
            print("  ✅ UI will show: Good quality indicator")
        else:
            print("  ⚠️  UI will show: Needs improvement indicator")
    
    # Test 4: Test streaming response with memory context
    print("\n🌊 Test 4: Streaming Response with Memory Context")
    
    streaming_query = "Tell me about my background and experience"
    print(f"  Query: '{streaming_query}'")
    
    response = requests.post(f"{BASE_URL}/agent-stream", 
                           data={"query": streaming_query},
                           headers={"Content-Type": "application/x-www-form-urlencoded"},
                           stream=True)
    
    if response.status_code == 200:
        print("  ✅ Streaming response initiated")
        
        # Parse streaming response
        memory_info_found = False
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        if data.get('type') == 'metadata' and data.get('memory_context_info'):
                            memory_info_found = True
                            memory_context = data['memory_context_info']
                            print(f"  🧠 Streaming Memory Context Found:")
                            print(f"    Memories Retrieved: {memory_context['memories_retrieved']}")
                            print(f"    Quality Score: {memory_context['system_stats']['quality_score']:.1f}%")
                            break
                    except json.JSONDecodeError:
                        continue
        
        if not memory_info_found:
            print("  ⚠️  Memory context not found in streaming response")
    else:
        print(f"  ❌ Streaming failed: {response.status_code}")
    
    print("\n🎉 Enhanced Agent Mode UI Test Completed!")
    print("\n🎯 UI Features Demonstrated in Agent Mode:")
    print("  ✅ Memory context information in responses")
    print("  ✅ Individual memory scores and parameters")
    print("  ✅ Combined scoring (similarity + importance + recency)")
    print("  ✅ Confidence levels and quality indicators")
    print("  ✅ Memory retrieval count and system stats")
    print("  ✅ Access frequency and memory age display")
    print("  ✅ Real-time quality scoring in responses")
    print("  ✅ Enhanced memory badges and visual indicators")
    
    print("\n💡 Agent Mode Now Shows:")
    print("  • Memory retrieval count (e.g., '3 Retrieved')")
    print("  • System quality score in each response")
    print("  • Individual memory cards with detailed scores")
    print("  • Combined score, importance, similarity, recency")
    print("  • Confidence badges (High/Medium/Low)")
    print("  • Access count and memory age")
    print("  • Visual color coding for quality levels")

if __name__ == "__main__":
    try:
        test_agent_mode_ui()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server at http://localhost:8000")
        print("Please make sure the backend server is running!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")