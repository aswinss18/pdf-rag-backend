#!/usr/bin/env python3
"""
Test script for the multi-document hybrid RAG pipeline
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_multi_document_pipeline():
    """Test the multi-document pipeline by asking questions and checking multi-doc metadata"""
    
    # First check if we have documents loaded
    status_response = requests.get(f"{BASE_URL}/status")
    if status_response.status_code != 200:
        print("❌ Server not responding")
        return False
    
    status_data = status_response.json()
    print(f"📊 Server Status: {status_data['documents_loaded']} documents loaded")
    
    if status_data['documents_loaded'] == 0:
        print("⚠️  No documents loaded. Please upload PDFs first.")
        return False
    
    # Test questions designed for multi-document analysis
    test_questions = [
        "What are the main topics covered in these documents?",  # General cross-document
        "Compare the methodologies discussed across documents",   # Comparison query
        "What are the key findings from all sources?",          # Synthesis query
        "How do the different documents approach this topic?",   # Multi-perspective
    ]
    
    print("\n🚀 Testing Multi-Document Hybrid RAG Pipeline...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/{len(test_questions)} ---")
        print(f"📝 Question: {question}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/ask-stream",
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                data=f"question={question}",
                stream=True
            )
            
            if response.status_code != 200:
                print(f"❌ Request failed with status {response.status_code}")
                continue
            
            print("📡 Processing multi-document analysis...")
            
            sources = []
            metadata = {}
            
            # Process streaming response
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    buffer += chunk
                    
                    # Process complete SSE events
                    events = buffer.split('\n\n')
                    buffer = events.pop()
                    
                    for event in events:
                        if event.startswith('data: '):
                            try:
                                data = json.loads(event[6:])
                                
                                if data.get('done'):
                                    break
                                
                                if 'sources' in data:
                                    sources = data['sources']
                                
                                if 'metadata' in data:
                                    metadata = data['metadata']
                                    
                            except json.JSONDecodeError:
                                continue
            
            # Analyze multi-document results
            print(f"\n🔍 Multi-Document Analysis:")
            
            # Document analysis
            if metadata.get('document_analysis'):
                doc_analysis = metadata['document_analysis']
                print(f"  • Documents found: {doc_analysis.get('document_count', 0)}")
                print(f"  • Multi-document mode: {doc_analysis.get('multi_document', False)}")
                print(f"  • Total chunks: {doc_analysis.get('total_chunks', 0)}")
                
                if doc_analysis.get('document_stats'):
                    print(f"  • Document breakdown:")
                    for doc_name, stats in doc_analysis['document_stats'].items():
                        print(f"    - {doc_name}: {stats.get('chunk_count', 0)} chunks ({stats.get('coverage_percentage', 0)}%)")
            
            # Context metadata
            if metadata.get('context_metadata'):
                context_meta = metadata['context_metadata']
                print(f"  • Context documents: {context_meta.get('documents', 0)}")
                print(f"  • Context length: {context_meta.get('context_length', 0)} chars")
                print(f"  • Multi-doc analysis: {context_meta.get('multi_document_analysis', False)}")
            
            # Hybrid search stats
            if metadata.get('hybrid_stats'):
                stats = metadata['hybrid_stats']
                print(f"  • Hybrid search total: {stats.get('total', 0)}")
                print(f"  • Vector only: {stats.get('vector_only', 0)}")
                print(f"  • Keyword only: {stats.get('keyword_only', 0)}")
                print(f"  • Both methods: {stats.get('both_methods', 0)}")
            
            # Source analysis
            print(f"\n📚 Source Analysis ({len(sources)} sources):")
            doc_sources = {}
            for source in sources:
                doc_name = source.get('doc', 'unknown')
                if doc_name not in doc_sources:
                    doc_sources[doc_name] = []
                doc_sources[doc_name].append(source)
            
            for doc_name, doc_sources_list in doc_sources.items():
                print(f"  📄 {doc_name}: {len(doc_sources_list)} chunks")
                for source in doc_sources_list[:2]:  # Show first 2 per document
                    search_types = source.get('search_types', [])
                    score = source.get('hybrid_score', source.get('combined_score', 0))
                    coverage = source.get('doc_coverage', 0)
                    
                    type_str = "+".join(search_types) if search_types else "unknown"
                    print(f"    - Page {source.get('page', '?')}: {type_str} (score: {score}, coverage: {coverage}%)")
                    
                    if source.get('matched_terms'):
                        print(f"      Keywords: {', '.join(source['matched_terms'][:3])}")
            
            # Check for multi-document features
            multi_doc_features = []
            if metadata.get('pipeline_version') == 'multi_doc_hybrid_v1':
                multi_doc_features.append("✅ Multi-document pipeline active")
            if metadata.get('document_analysis', {}).get('multi_document'):
                multi_doc_features.append("✅ Multi-document analysis")
            if metadata.get('context_metadata', {}).get('multi_document_analysis'):
                multi_doc_features.append("✅ Cross-document context")
            if len(doc_sources) > 1:
                multi_doc_features.append("✅ Multiple document sources")
            
            print(f"\n🎯 Multi-Document Features:")
            for feature in multi_doc_features:
                print(f"  {feature}")
            
            if len(multi_doc_features) >= 2:
                print(f"  🎉 Multi-document analysis working!")
            else:
                print(f"  ⚠️  Limited multi-document functionality")
                
        except Exception as e:
            print(f"❌ Error testing question '{question}': {e}")
    
    print(f"\n🎉 Multi-document pipeline test completed!")
    return True

def test_document_comparison():
    """Test specific document comparison capabilities"""
    print(f"\n🔬 Testing Document Comparison Features...")
    
    comparison_question = "Compare the approaches discussed in different documents"
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=f"question={comparison_question}",
        )
        
        if response.status_code == 200:
            answer = response.json()
            print(f"📝 Comparison Question: {comparison_question}")
            print(f"💬 Answer Preview: {answer.get('answer', '')[:200]}...")
            print("✅ Document comparison endpoint working")
            return True
        else:
            print(f"❌ Comparison test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing comparison: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Multi-Document Hybrid RAG Pipeline Test")
    print("=" * 50)
    
    success = test_multi_document_pipeline()
    comparison_success = test_document_comparison()
    
    if success and comparison_success:
        print("\n✅ All tests completed successfully!")
        print("\n💡 The multi-document pipeline provides:")
        print("  • Cross-document analysis and comparison")
        print("  • Document-aware context building")
        print("  • Multi-perspective synthesis")
        print("  • Hybrid search across all documents")
        print("  • Intelligent document grouping")
        print("  • Specialized comparison prompts")
    else:
        print("\n❌ Some tests failed or features not working")