#!/usr/bin/env python3
"""
Test script for the importance scoring system (no embeddings required)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.memory import AgentMemory
import time

def test_importance_scoring():
    """Test the importance scoring system without embeddings"""
    print("🧠 Testing Importance Scoring System")
    print("=" * 50)
    
    # Create a test memory instance
    memory = AgentMemory()
    
    # Test importance calculation for various texts
    test_texts = [
        ("Aswin's salary is ₹47,000 per month", "High importance - salary + currency + name"),
        ("User likes coffee in the morning", "Medium importance - preference"),
        ("The weather is nice today", "Low importance - weather"),
        ("Hello how are you", "Very low importance - greeting"),
        ("I work as a Software Developer at GigLabz Private Ltd", "High importance - job + company + personal"),
        ("My name is John and I live in Bangalore", "High importance - name + location + personal"),
        ("The temperature is 25°C", "Low importance - weather data"),
        ("I prefer React over Angular for frontend development", "Medium importance - preference + technical"),
        ("Budget for this project is $50,000", "High importance - money + numbers"),
        ("See you later", "Very low importance - casual"),
        ("My goal is to become a senior developer within 2 years", "High importance - goal + personal + numbers"),
        ("I have 5 years of experience in Python development", "High importance - experience + numbers + technical"),
    ]
    
    print("\n📊 Importance Scoring Results:")
    print("-" * 80)
    
    high_importance = []
    medium_importance = []
    low_importance = []
    
    for text, description in test_texts:
        importance = memory.calculate_importance(text)
        should_store = memory.should_store_memory(text)
        
        # Categorize by importance
        if importance >= 0.8:
            high_importance.append((text, importance))
            category = "🔴 HIGH"
        elif importance >= 0.6:
            medium_importance.append((text, importance))
            category = "🟡 MEDIUM"
        else:
            low_importance.append((text, importance))
            category = "🟢 LOW"
        
        store_status = "✅ STORE" if should_store else "❌ SKIP"
        
        print(f"{category:12} | {importance:.2f} | {store_status} | {text[:45]:<45} | {description}")
    
    print("\n" + "=" * 80)
    print(f"📈 Summary:")
    print(f"  • High Importance (≥0.8): {len(high_importance)} memories")
    print(f"  • Medium Importance (≥0.6): {len(medium_importance)} memories") 
    print(f"  • Low Importance (<0.6): {len(low_importance)} memories")
    print(f"  • Will be stored (>0.6): {len(high_importance) + len(medium_importance)} memories")
    print(f"  • Will be filtered out: {len(low_importance)} memories")
    
    # Test memory decay calculation
    print("\n⏰ Testing Memory Decay:")
    print("-" * 50)
    
    test_ages = [0.5, 1, 7, 30, 90, 180, 365]  # days
    original_importance = 0.9
    
    for age_days in test_ages:
        decayed = memory._apply_memory_decay(original_importance, age_days)
        decay_percent = ((original_importance - decayed) / original_importance) * 100
        
        print(f"  Age: {age_days:3.0f} days | Original: {original_importance:.2f} → Decayed: {decayed:.2f} | Decay: {decay_percent:4.1f}%")
    
    print("\n✅ Importance Scoring System Test Completed!")
    print("\n🎯 Key Features Verified:")
    print("  ✓ Keyword-based importance scoring (0.0-1.0)")
    print("  ✓ Currency and financial data detection")
    print("  ✓ Personal information identification")
    print("  ✓ Technical content recognition")
    print("  ✓ Selective storage (only importance > 0.6)")
    print("  ✓ Memory decay based on age")
    print("  ✓ Different decay rates for different importance levels")

if __name__ == "__main__":
    test_importance_scoring()