"""Tests for Search Service."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "opensearch" in data


def test_search_basic():
    """Test basic search."""
    response = client.post(
        "/search",
        json={
            "query": "chest",
            "page": 1,
            "page_size": 20
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "results" in data
    assert "aggregations" in data


def test_search_with_filters():
    """Test search with faceted filters."""
    response = client.post(
        "/search",
        json={
            "query": "xray",
            "modalities": ["CT", "MRI"],
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "page": 1,
            "page_size": 20
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["results"], list)


def test_search_semantic():
    """Test semantic search."""
    response = client.post(
        "/search",
        json={
            "query": "lung abnormalities",
            "semantic_search": True,
            "page": 1,
            "page_size": 10
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_search_with_annotations_filter():
    """Test search filtering by annotations."""
    response = client.post(
        "/search",
        json={
            "query": "",
            "has_annotations": True,
            "annotation_labels": ["tumor", "nodule"],
            "page": 1,
            "page_size": 20
        }
    )
    assert response.status_code == 200


def test_search_with_predictions_filter():
    """Test search filtering by AI predictions."""
    response = client.post(
        "/search",
        json={
            "query": "",
            "prediction_models": ["chest-xray-classifier"],
            "page": 1,
            "page_size": 20
        }
    )
    assert response.status_code == 200


def test_get_facets():
    """Test getting available facets."""
    response = client.get("/facets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    # Check for expected facet keys
    expected_keys = ["modalities", "body_parts", "institutions"]
    for key in expected_keys:
        assert key in data


def test_pagination():
    """Test search pagination."""
    # Page 1
    response1 = client.post(
        "/search",
        json={
            "query": "",
            "page": 1,
            "page_size": 10
        }
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["page"] == 1
    assert data1["page_size"] == 10

    # Page 2
    response2 = client.post(
        "/search",
        json={
            "query": "",
            "page": 2,
            "page_size": 10
        }
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["page"] == 2


def test_reindex_request():
    """Test reindex endpoint."""
    response = client.post(
        "/reindex",
        json={
            "index_name": "studies",
            "batch_size": 1000,
            "full_reindex": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_index_single_study():
    """Test indexing a single study."""
    response = client.post("/index/study/test-study-id")
    assert response.status_code == 200


def test_save_query():
    """Test saving a search query."""
    response = client.post(
        "/queries/save",
        json={
            "name": "Test Query",
            "description": "A test saved query",
            "query": {
                "query": "chest xray",
                "modalities": ["X-Ray"],
                "page": 1,
                "page_size": 20
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Query"


def test_list_saved_queries():
    """Test listing saved queries."""
    response = client.get("/queries")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_export_csv():
    """Test CSV export."""
    response = client.post(
        "/export/csv",
        json={
            "query": "test",
            "page": 1,
            "page_size": 100
        }
    )
    assert response.status_code == 200


@pytest.mark.benchmark
def test_search_performance():
    """Benchmark search performance."""
    import time
    
    queries = [
        {"query": "chest", "semantic_search": False},
        {"query": "chest", "semantic_search": True},
        {"query": "brain tumor", "modalities": ["MRI"]},
        {"query": "", "has_annotations": True}
    ]
    
    results = []
    for query_data in queries:
        start = time.time()
        response = client.post("/search", json={**query_data, "page": 1, "page_size": 20})
        duration = time.time() - start
        
        assert response.status_code == 200
        data = response.json()
        results.append({
            "query": query_data.get("query", ""),
            "semantic": query_data.get("semantic_search", False),
            "duration_ms": duration * 1000,
            "took_ms": data.get("took_ms", 0),
            "results_count": len(data.get("results", []))
        })
    
    # Print benchmark results
    print("\n" + "="*60)
    print("SEARCH PERFORMANCE BENCHMARK")
    print("="*60)
    for result in results:
        print(f"\nQuery: '{result['query']}' (semantic={result['semantic']})")
        print(f"  Total duration: {result['duration_ms']:.2f}ms")
        print(f"  OpenSearch took: {result['took_ms']}ms")
        print(f"  Results returned: {result['results_count']}")
    print("="*60 + "\n")
    
    # Assert performance thresholds
    for result in results:
        assert result['duration_ms'] < 5000, f"Search took too long: {result['duration_ms']}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
