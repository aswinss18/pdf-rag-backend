#!/usr/bin/env python3
"""
Test script to verify JSON serialization of all data structures
"""

import json
import numpy as np
from core.vector_store import search
from core.hybrid_search import hybrid_search, get_hybrid_search_stats
from core.multi_document_context import group_chunks_by_document, analyze_document_distribution
from core.embeddings import get_embedding

def test_json_serialization():
    """Test that all data structures can be JSON serialized"""
    
    print("🧪 Testing JSON Serialization...")
    
    # Test basic data structures
    test_data = {
        "similarity_score": 0.85,
        "hybrid_score": 0.92,
        "vector_score": 0.88,
        "keyword_score": 0.75,
        "cosine_similarity": 0.91,
        "combined_score": 0.89,
        "coverage_percentage": 15.5,
        "avg_hybrid_score": 0.87,
        "distance": 0.23
    }
    
    try:
        json_str = json.dumps(test_data)
        print("✅ Basic data structures serialize correctly")
    except Exception as e:
        print(f"❌ Basic serialization failed: {e}")
        return False
    
    # Test with numpy types (should fail before our fixes)
    numpy_data = {
        "numpy_float32": np.float32(0.85),
        "numpy_float64": np.float64(0.92),
        "python_float": float(0.88)
    }
    
    try:
        json_str = json.dumps(numpy_data)
        print("❌ Numpy types should not serialize directly")
        return False
    except TypeError:
        print("✅ Numpy types correctly fail serialization (as expected)")
    
    # Test converted numpy types
    converted_data = {
        "converted_float32": float(np.float32(0.85)),
        "converted_float64": float(np.float64(0.92)),
        "python_float": float(0.88)
    }
    
    try:
        json_str = json.dumps(converted_data)
        print("✅ Converted numpy types serialize correctly")
    except Exception as e:
        print(f"❌ Converted numpy serialization failed: {e}")
        return False
    
    # Test hybrid search stats structure
    mock_results = [
        {
            "hybrid_score": 0.95,
            "search_types": ["vector", "keyword"]
        },
        {
            "hybrid_score": 0.87,
            "search_types": ["vector"]
        },
        {
            "hybrid_score": 0.82,
            "search_types": ["keyword"]
        }
    ]
    
    try:
        stats = get_hybrid_search_stats(mock_results)
        json_str = json.dumps(stats)
        print("✅ Hybrid search stats serialize correctly")
    except Exception as e:
        print(f"❌ Hybrid stats serialization failed: {e}")
        return False
    
    # Test document analysis structure
    mock_chunks = [
        {"doc": "doc1.pdf", "page": 1, "text": "test", "hybrid_score": 0.9},
        {"doc": "doc2.pdf", "page": 2, "text": "test", "hybrid_score": 0.8}
    ]
    
    try:
        grouped = group_chunks_by_document(mock_chunks)
        analysis = analyze_document_distribution(grouped)
        json_str = json.dumps(analysis)
        print("✅ Document analysis serializes correctly")
    except Exception as e:
        print(f"❌ Document analysis serialization failed: {e}")
        return False
    
    print("\n🎉 All JSON serialization tests passed!")
    return True

if __name__ == "__main__":
    success = test_json_serialization()
    if success:
        print("\n✅ JSON serialization is working correctly")
        print("The streaming API should now work without TypeError")
    else:
        print("\n❌ JSON serialization issues remain")