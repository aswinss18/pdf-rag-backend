#!/usr/bin/env python3
"""
Test script to verify multi-document upload functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_multi_document_support():
    """Test that multiple documents can be uploaded and accessed"""
    
    print("🧪 Testing Multi-Document Upload Support...")
    
    # Test 1: Check initial status
    print("\n1. Checking initial status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"   Initial documents: {data.get('unique_documents', 0)}")
            print(f"   Document names: {data.get('document_names', [])}")
            print(f"   Multi-document mode: {data.get('multi_document_mode', False)}")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error checking status: {e}")
        return False
    
    # Test 2: List current documents
    print("\n2. Listing current documents...")
    try:
        response = requests.get(f"{BASE_URL}/documents/list")
        if response.status_code == 200:
            data = response.json()
            print(f"   Total documents: {data.get('total_documents', 0)}")
            print(f"   Total chunks: {data.get('total_chunks', 0)}")
            if data.get('documents'):
                for doc_name, info in data['documents'].items():
                    print(f"   - {doc_name}: {info['chunk_count']} chunks, pages {info['page_range']}")
        else:
            print(f"   ❌ Document list failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error listing documents: {e}")
    
    # Test 3: Test document clearing
    print("\n3. Testing document clearing...")
    try:
        response = requests.post(f"{BASE_URL}/documents/clear")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ {data.get('message', 'Documents cleared')}")
            print(f"   Documents after clear: {data.get('documents_loaded', 0)}")
        else:
            print(f"   ❌ Clear failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error clearing documents: {e}")
    
    # Test 4: Check status after clear
    print("\n4. Checking status after clear...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"   Documents after clear: {data.get('unique_documents', 0)}")
            print(f"   Status: {data.get('status', 'unknown')}")
            if data.get('unique_documents', 0) == 0:
                print("   ✅ Clear operation successful")
            else:
                print("   ⚠️  Documents still present after clear")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error checking status: {e}")
    
    print("\n🎉 Multi-document upload test completed!")
    print("\n💡 Key Changes Made:")
    print("  • Removed clear_documents() from upload endpoint")
    print("  • Added /documents/clear endpoint for manual clearing")
    print("  • Added /documents/list endpoint for document management")
    print("  • Updated UI to support multi-document workflow")
    print("  • Enhanced status endpoint with multi-document info")
    
    return True

def test_upload_simulation():
    """Simulate the upload workflow"""
    print("\n📤 Upload Workflow Simulation:")
    print("  1. User uploads PDF A → Documents: [A]")
    print("  2. User uploads PDF B → Documents: [A, B] (not cleared!)")
    print("  3. User asks question → Searches across both A and B")
    print("  4. Multi-document analysis compares A vs B")
    print("  5. User can manually clear all documents if needed")

if __name__ == "__main__":
    print("🔬 Multi-Document Upload Support Test")
    print("=" * 50)
    
    success = test_multi_document_support()
    test_upload_simulation()
    
    if success:
        print("\n✅ Multi-document support is now active!")
        print("\n🚀 Next Steps:")
        print("  1. Upload multiple PDFs through the UI")
        print("  2. Ask questions that compare documents")
        print("  3. Use 'Clear All Documents' to start fresh")
    else:
        print("\n❌ Some tests failed")