"""Reindexing script for OpenSearch.

This script reads data from PostgreSQL and indexes it into OpenSearch.
Can be run standalone or as a Celery task.
"""
import os
import sys
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/aurelius")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))


class StudyReindexer:
    """Reindex studies from PostgreSQL to OpenSearch."""
    
    def __init__(self):
        self.opensearch = OpenSearch(
            hosts=[OPENSEARCH_URL],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_conn = None
    
    def connect_db(self):
        """Connect to PostgreSQL."""
        self.db_conn = psycopg2.connect(DATABASE_URL)
        print(f"✅ Connected to PostgreSQL")
    
    def fetch_studies_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Fetch a batch of studies from database."""
        query = """
        SELECT 
            s.id as study_id,
            s.study_instance_uid,
            s.patient_id,
            s.accession_number,
            s.study_date,
            s.study_description,
            s.modality,
            s.referring_physician,
            s.number_of_series,
            s.number_of_instances,
            s.metadata,
            s.created_at,
            s.updated_at,
            p.first_name,
            p.last_name,
            o.name as institution
        FROM studies s
        LEFT JOIN patients p ON s.patient_id = p.id
        LEFT JOIN organizations o ON s.organization_id = o.id
        WHERE s.deleted_at IS NULL
        ORDER BY s.created_at
        LIMIT %s OFFSET %s
        """
        
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (limit, offset))
            return cursor.fetchall()
    
    def fetch_study_annotations(self, study_id: str) -> List[Dict[str, Any]]:
        """Fetch annotations for a study."""
        query = """
        SELECT label, properties
        FROM annotations
        WHERE target_id = %s AND target_type = 'study'
        AND deleted_at IS NULL
        """
        
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (study_id,))
            return cursor.fetchall()
    
    def fetch_study_predictions(self, study_id: str) -> List[Dict[str, Any]]:
        """Fetch AI predictions for a study."""
        query = """
        SELECT 
            m.model_name,
            p.results,
            p.confidence
        FROM predictions p
        JOIN ml_models m ON p.model_id = m.id
        WHERE p.target_id = %s AND p.target_type = 'study'
        ORDER BY p.created_at DESC
        """
        
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (study_id,))
            return cursor.fetchall()
    
    def prepare_document(self, study: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a study document for indexing."""
        # Fetch related data
        annotations = self.fetch_study_annotations(study["study_id"])
        predictions = self.fetch_study_predictions(study["study_id"])
        
        # Extract body part from metadata (DICOM tag 0018,0015)
        body_part = None
        if study.get("metadata"):
            body_part = study["metadata"].get("BodyPartExamined")
        
        # Create text for embedding
        text_for_embedding = f"{study.get('study_description', '')} {study.get('modality', '')} {body_part or ''}"
        
        # Generate embedding
        embedding = self.semantic_model.encode(text_for_embedding).tolist()
        
        # Prepare document
        doc = {
            "_index": "studies",
            "_id": study["study_id"],
            "_source": {
                "study_id": study["study_id"],
                "study_instance_uid": study["study_instance_uid"],
                "patient_id": study["patient_id"],
                "accession_number": study.get("accession_number"),
                "study_date": study.get("study_date").isoformat() if study.get("study_date") else None,
                "study_description": study.get("study_description"),
                "modality": study.get("modality"),
                "body_part": body_part,
                "referring_physician": study.get("referring_physician"),
                "institution": study.get("institution"),
                "number_of_series": study.get("number_of_series", 0),
                "number_of_instances": study.get("number_of_instances", 0),
                "annotations": [
                    {
                        "label": ann["label"],
                        "confidence": ann.get("properties", {}).get("confidence", 1.0)
                    }
                    for ann in annotations
                ],
                "predictions": [
                    {
                        "model_name": pred["model_name"],
                        "prediction": pred["results"].get("class", "unknown"),
                        "confidence": pred.get("confidence", 0.0)
                    }
                    for pred in predictions
                ],
                "embedding": embedding,
                "indexed_at": datetime.utcnow().isoformat(),
                "updated_at": study.get("updated_at").isoformat() if study.get("updated_at") else None
            }
        }
        
        return doc
    
    def reindex(self, full_reindex: bool = False):
        """Perform reindexing."""
        print(f"\n{'='*60}")
        print(f"REINDEXING STUDIES")
        print(f"{'='*60}")
        print(f"Mode: {'FULL' if full_reindex else 'INCREMENTAL'}")
        print(f"Batch size: {BATCH_SIZE}")
        print()
        
        self.connect_db()
        
        # Get total count
        with self.db_conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM studies WHERE deleted_at IS NULL")
            total_studies = cursor.fetchone()[0]
        
        print(f"Total studies to index: {total_studies}")
        
        # Create index if it doesn't exist
        if not self.opensearch.indices.exists(index="studies"):
            print("Creating studies index...")
            from app.main import SearchConfig
            self.opensearch.indices.create(
                index="studies",
                body=SearchConfig.STUDIES_MAPPING
            )
            print("✅ Index created")
        
        # Reindex in batches
        offset = 0
        indexed_count = 0
        failed_count = 0
        
        while offset < total_studies:
            print(f"\nProcessing batch: {offset}-{offset + BATCH_SIZE} ({int(offset/total_studies*100)}%)")
            
            # Fetch batch
            studies = self.fetch_studies_batch(offset, BATCH_SIZE)
            
            if not studies:
                break
            
            # Prepare documents
            documents = []
            for study in studies:
                try:
                    doc = self.prepare_document(study)
                    documents.append(doc)
                except Exception as e:
                    print(f"❌ Failed to prepare study {study['study_id']}: {e}")
                    failed_count += 1
            
            # Bulk index
            try:
                success, failed = helpers.bulk(
                    self.opensearch,
                    documents,
                    chunk_size=BATCH_SIZE,
                    raise_on_error=False
                )
                indexed_count += success
                failed_count += len(failed)
                
                if failed:
                    print(f"⚠️ {len(failed)} documents failed to index")
                
                print(f"✅ Indexed {success} documents")
                
            except Exception as e:
                print(f"❌ Batch indexing failed: {e}")
                failed_count += len(documents)
            
            offset += BATCH_SIZE
        
        # Summary
        print(f"\n{'='*60}")
        print(f"REINDEXING COMPLETE")
        print(f"{'='*60}")
        print(f"Total: {total_studies}")
        print(f"Indexed: {indexed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {int(indexed_count/total_studies*100)}%")
        print(f"{'='*60}\n")
        
        self.db_conn.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reindex studies to OpenSearch")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Perform full reindex (default: incremental)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for indexing (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Override batch size if provided
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
    
    # Run reindexing
    reindexer = StudyReindexer()
    reindexer.reindex(full_reindex=args.full)


if __name__ == "__main__":
    main()
