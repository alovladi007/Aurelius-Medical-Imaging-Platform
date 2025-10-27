"""Search Service - OpenSearch indexing and retrieval with semantic search."""
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuration
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/aurelius")

# Global instances
opensearch_client = None
semantic_model = None


class SearchConfig:
    """OpenSearch index configurations."""
    
    STUDIES_INDEX = "studies"
    ANNOTATIONS_INDEX = "annotations"
    SLIDES_INDEX = "slides"
    
    STUDIES_MAPPING = {
        "mappings": {
            "properties": {
                "study_id": {"type": "keyword"},
                "study_instance_uid": {"type": "keyword"},
                "patient_id": {"type": "keyword"},
                "accession_number": {"type": "keyword"},
                "study_date": {"type": "date"},
                "study_description": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "modality": {"type": "keyword"},
                "body_part": {"type": "keyword"},
                "referring_physician": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "institution": {"type": "keyword"},
                "number_of_series": {"type": "integer"},
                "number_of_instances": {"type": "integer"},
                "annotations": {
                    "type": "nested",
                    "properties": {
                        "label": {"type": "keyword"},
                        "confidence": {"type": "float"}
                    }
                },
                "predictions": {
                    "type": "nested",
                    "properties": {
                        "model_name": {"type": "keyword"},
                        "prediction": {"type": "keyword"},
                        "confidence": {"type": "float"}
                    }
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,  # all-MiniLM-L6-v2 dimension
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                },
                "indexed_at": {"type": "date"},
                "updated_at": {"type": "date"}
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True,
                "knn.algo_param.ef_search": 100
            },
            "analysis": {
                "analyzer": {
                    "medical_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"]
                    }
                }
            }
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize OpenSearch and semantic model on startup."""
    global opensearch_client, semantic_model
    
    print("🔍 Initializing Search Service...")
    
    # Initialize OpenSearch client
    opensearch_client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    
    # Wait for OpenSearch to be ready
    for i in range(30):
        try:
            if opensearch_client.ping():
                print("✅ OpenSearch connected")
                break
        except Exception:
            if i == 29:
                print("❌ OpenSearch connection failed")
            await asyncio.sleep(1)
    
    # Create indexes if they don't exist
    try:
        if not opensearch_client.indices.exists(index=SearchConfig.STUDIES_INDEX):
            opensearch_client.indices.create(
                index=SearchConfig.STUDIES_INDEX,
                body=SearchConfig.STUDIES_MAPPING
            )
            print(f"✅ Created index: {SearchConfig.STUDIES_INDEX}")
    except Exception as e:
        print(f"⚠️ Index creation: {e}")
    
    # Load semantic search model
    print("🤖 Loading sentence transformer model...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Semantic model loaded")
    
    yield
    
    # Cleanup
    print("👋 Shutting down Search Service...")


app = FastAPI(
    title="Aurelius Search Service",
    version="1.0.0",
    description="Search and retrieval service with semantic search",
    lifespan=lifespan
)


# ============================================================================
# MODELS
# ============================================================================

class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., description="Search query text")
    modalities: Optional[List[str]] = Field(None, description="Filter by modalities")
    body_parts: Optional[List[str]] = Field(None, description="Filter by body parts")
    date_from: Optional[date] = Field(None, description="Filter from date")
    date_to: Optional[date] = Field(None, description="Filter to date")
    institutions: Optional[List[str]] = Field(None, description="Filter by institutions")
    has_annotations: Optional[bool] = Field(None, description="Filter by annotation presence")
    annotation_labels: Optional[List[str]] = Field(None, description="Filter by annotation labels")
    prediction_models: Optional[List[str]] = Field(None, description="Filter by model predictions")
    semantic_search: bool = Field(False, description="Enable semantic search")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Results per page")


class SearchResult(BaseModel):
    """Search result item."""
    study_id: str
    study_instance_uid: str
    study_date: Optional[date]
    study_description: Optional[str]
    modality: Optional[str]
    body_part: Optional[str]
    number_of_series: int
    number_of_instances: int
    score: float
    highlights: Dict[str, List[str]] = {}


class SearchResponse(BaseModel):
    """Search response."""
    total: int
    page: int
    page_size: int
    took_ms: int
    results: List[SearchResult]
    aggregations: Dict[str, Any] = {}


class SavedQuery(BaseModel):
    """Saved search query."""
    name: str
    description: Optional[str]
    query: SearchQuery


class ReindexRequest(BaseModel):
    """Reindex request."""
    index_name: str
    batch_size: int = 1000
    full_reindex: bool = False


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check."""
    try:
        es_health = opensearch_client.cluster.health()
        return {
            "status": "healthy",
            "service": "search-svc",
            "opensearch": es_health["status"],
            "semantic_model": "loaded" if semantic_model else "not loaded"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.post("/search", response_model=SearchResponse)
async def search_studies(query: SearchQuery):
    """
    Search studies with faceted filtering and optional semantic search.
    
    Supports:
    - Full-text search
    - Faceted filtering (modality, body part, date range, etc.)
    - Semantic search with sentence embeddings
    - Aggregations for faceted navigation
    """
    # Build OpenSearch query
    must_clauses = []
    filter_clauses = []
    
    # Text search
    if query.query:
        if query.semantic_search and semantic_model:
            # Semantic search using vector similarity
            query_embedding = semantic_model.encode(query.query).tolist()
            must_clauses.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            })
        else:
            # Traditional keyword search
            must_clauses.append({
                "multi_match": {
                    "query": query.query,
                    "fields": [
                        "study_description^3",
                        "referring_physician^2",
                        "body_part^2",
                        "modality"
                    ],
                    "type": "best_fields",
                    "operator": "or"
                }
            })
    
    # Faceted filters
    if query.modalities:
        filter_clauses.append({"terms": {"modality": query.modalities}})
    
    if query.body_parts:
        filter_clauses.append({"terms": {"body_part": query.body_parts}})
    
    if query.institutions:
        filter_clauses.append({"terms": {"institution": query.institutions}})
    
    if query.date_from or query.date_to:
        date_range = {}
        if query.date_from:
            date_range["gte"] = query.date_from.isoformat()
        if query.date_to:
            date_range["lte"] = query.date_to.isoformat()
        filter_clauses.append({"range": {"study_date": date_range}})
    
    if query.has_annotations is not None:
        if query.has_annotations:
            filter_clauses.append({"exists": {"field": "annotations"}})
        else:
            filter_clauses.append({
                "bool": {"must_not": {"exists": {"field": "annotations"}}}
            })
    
    if query.annotation_labels:
        filter_clauses.append({
            "nested": {
                "path": "annotations",
                "query": {"terms": {"annotations.label": query.annotation_labels}}
            }
        })
    
    if query.prediction_models:
        filter_clauses.append({
            "nested": {
                "path": "predictions",
                "query": {"terms": {"predictions.model_name": query.prediction_models}}
            }
        })
    
    # Build complete query
    opensearch_query = {
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}],
                "filter": filter_clauses
            }
        },
        "from": (query.page - 1) * query.page_size,
        "size": query.page_size,
        "highlight": {
            "fields": {
                "study_description": {},
                "referring_physician": {}
            }
        },
        "aggs": {
            "modalities": {"terms": {"field": "modality", "size": 20}},
            "body_parts": {"terms": {"field": "body_part", "size": 50}},
            "institutions": {"terms": {"field": "institution", "size": 20}},
            "date_histogram": {
                "date_histogram": {
                    "field": "study_date",
                    "calendar_interval": "month"
                }
            }
        }
    }
    
    # Execute search
    try:
        response = opensearch_client.search(
            index=SearchConfig.STUDIES_INDEX,
            body=opensearch_query
        )
        
        # Parse results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(SearchResult(
                study_id=source["study_id"],
                study_instance_uid=source["study_instance_uid"],
                study_date=source.get("study_date"),
                study_description=source.get("study_description"),
                modality=source.get("modality"),
                body_part=source.get("body_part"),
                number_of_series=source["number_of_series"],
                number_of_instances=source["number_of_instances"],
                score=hit["_score"],
                highlights=hit.get("highlight", {})
            ))
        
        # Parse aggregations
        aggregations = {}
        if "aggregations" in response:
            for agg_name, agg_data in response["aggregations"].items():
                if "buckets" in agg_data:
                    aggregations[agg_name] = [
                        {"key": bucket["key"], "count": bucket["doc_count"]}
                        for bucket in agg_data["buckets"]
                    ]
        
        return SearchResponse(
            total=response["hits"]["total"]["value"],
            page=query.page,
            page_size=query.page_size,
            took_ms=response["took"],
            results=results,
            aggregations=aggregations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/facets")
async def get_facets():
    """Get available facets for filtering."""
    try:
        # Get all unique values for facets
        aggs_query = {
            "size": 0,
            "aggs": {
                "modalities": {"terms": {"field": "modality", "size": 50}},
                "body_parts": {"terms": {"field": "body_part", "size": 100}},
                "institutions": {"terms": {"field": "institution", "size": 50}},
                "annotation_labels": {
                    "nested": {"path": "annotations"},
                    "aggs": {
                        "labels": {"terms": {"field": "annotations.label", "size": 100}}
                    }
                },
                "prediction_models": {
                    "nested": {"path": "predictions"},
                    "aggs": {
                        "models": {"terms": {"field": "predictions.model_name", "size": 50}}
                    }
                }
            }
        }
        
        response = opensearch_client.search(
            index=SearchConfig.STUDIES_INDEX,
            body=aggs_query
        )
        
        facets = {}
        for agg_name, agg_data in response["aggregations"].items():
            if agg_name in ["annotation_labels", "prediction_models"]:
                # Handle nested aggregations
                nested_agg = agg_data[list(agg_data.keys())[1]]  # Get nested agg
                facets[agg_name] = [
                    bucket["key"] for bucket in nested_agg["buckets"]
                ]
            else:
                facets[agg_name] = [
                    bucket["key"] for bucket in agg_data["buckets"]
                ]
        
        return facets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get facets: {str(e)}")


# ============================================================================
# INDEXING ENDPOINTS
# ============================================================================

@app.post("/reindex")
async def reindex_studies(request: ReindexRequest):
    """
    Reindex studies from database to OpenSearch.
    
    This can be run as a background job using Celery in production.
    """
    # TODO: Implement actual database query and indexing
    # For now, return a job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Reindexing {request.index_name} with batch_size={request.batch_size}"
    }


@app.post("/index/study/{study_id}")
async def index_study(study_id: str):
    """Index or update a single study."""
    # TODO: Implement single study indexing
    return {"message": f"Study {study_id} queued for indexing"}


@app.delete("/index/study/{study_id}")
async def delete_study_from_index(study_id: str):
    """Remove study from search index."""
    try:
        response = opensearch_client.delete(
            index=SearchConfig.STUDIES_INDEX,
            id=study_id
        )
        return {"message": f"Study {study_id} deleted from index"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Study not found in index: {str(e)}")


# ============================================================================
# SAVED QUERIES
# ============================================================================

@app.post("/queries/save")
async def save_query(saved_query: SavedQuery):
    """Save a search query for later use."""
    # TODO: Store in database
    return {"message": "Query saved", "name": saved_query.name}


@app.get("/queries")
async def list_saved_queries():
    """List all saved queries."""
    # TODO: Load from database
    return []


@app.delete("/queries/{query_name}")
async def delete_saved_query(query_name: str):
    """Delete a saved query."""
    # TODO: Delete from database
    return {"message": f"Query {query_name} deleted"}


# ============================================================================
# EXPORT
# ============================================================================

@app.post("/export/csv")
async def export_search_results_csv(query: SearchQuery):
    """Export search results to CSV."""
    # TODO: Implement CSV export
    return {"message": "CSV export queued"}


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    uvicorn.run(app, host="0.0.0.0", port=8004)
