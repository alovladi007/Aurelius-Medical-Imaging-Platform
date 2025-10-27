"""Locust load testing script for Aurelius API.

This script simulates realistic user behavior patterns for load testing.

Usage:
    # Basic load test
    locust -f load_test.py --host=http://localhost:8000

    # Headless mode with specific users/spawn rate
    locust -f load_test.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless

    # Distributed load test (master)
    locust -f load_test.py --host=http://localhost:8000 --master

    # Distributed load test (worker)
    locust -f load_test.py --host=http://localhost:8000 --worker --master-host=<master-ip>
"""
import random
import json
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask


class APIUser(HttpUser):
    """Base user class with authentication."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    def on_start(self):
        """Called when a user starts. Perform login."""
        # Login to get JWT token
        response = self.client.post(
            "/auth/login",
            json={
                "username": "doctor",
                "password": "doctor123"
            },
            name="/auth/login"
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            print(f"Login failed: {response.status_code}")
            raise RescheduleTask()


class ClinicianUser(APIUser):
    """Simulates a clinician browsing studies and viewing images."""
    
    weight = 5  # 50% of users are clinicians
    
    @task(10)
    def list_studies(self):
        """List studies with pagination."""
        page = random.randint(1, 10)
        self.client.get(
            f"/studies?page={page}&page_size=20",
            headers=self.headers,
            name="/studies (list)"
        )
    
    @task(5)
    def search_studies(self):
        """Search for studies."""
        queries = ["chest", "brain", "abdomen", "spine", "lung"]
        query = random.choice(queries)
        
        self.client.post(
            "/search",
            json={
                "query": query,
                "page": 1,
                "page_size": 20
            },
            headers=self.headers,
            name="/search (query)"
        )
    
    @task(3)
    def search_with_filters(self):
        """Search with faceted filters."""
        self.client.post(
            "/search",
            json={
                "query": "xray",
                "modalities": ["X-Ray", "CT"],
                "date_from": "2024-01-01",
                "page": 1,
                "page_size": 20
            },
            headers=self.headers,
            name="/search (filtered)"
        )
    
    @task(2)
    def get_study_details(self):
        """Get details of a specific study."""
        # In a real test, you'd have actual study IDs
        study_id = f"study-{random.randint(1, 1000)}"
        self.client.get(
            f"/studies/{study_id}",
            headers=self.headers,
            name="/studies/{id}",
            catch_response=True
        )
    
    @task(1)
    def check_worklists(self):
        """Check assigned worklists."""
        self.client.get(
            "/worklists",
            headers=self.headers,
            name="/worklists"
        )


class ResearcherUser(APIUser):
    """Simulates a researcher querying data and running ML models."""
    
    weight = 3  # 30% of users are researchers
    
    @task(5)
    def semantic_search(self):
        """Perform semantic search."""
        queries = [
            "lung nodules with high density",
            "brain lesions in frontal lobe",
            "cardiac abnormalities",
            "liver masses with contrast enhancement"
        ]
        query = random.choice(queries)
        
        self.client.post(
            "/search",
            json={
                "query": query,
                "semantic_search": True,
                "page": 1,
                "page_size": 20
            },
            headers=self.headers,
            name="/search (semantic)"
        )
    
    @task(3)
    def search_with_annotations(self):
        """Search for annotated studies."""
        self.client.post(
            "/search",
            json={
                "query": "",
                "has_annotations": True,
                "annotation_labels": ["tumor", "nodule"],
                "page": 1,
                "page_size": 50
            },
            headers=self.headers,
            name="/search (annotations)"
        )
    
    @task(2)
    def list_ml_models(self):
        """List available ML models."""
        self.client.get(
            "/ml/models",
            headers=self.headers,
            name="/ml/models"
        )
    
    @task(1)
    def run_ml_prediction(self):
        """Trigger ML inference."""
        self.client.post(
            "/ml/predict",
            json={
                "model_name": "chest-xray-classifier",
                "model_version": "latest",
                "input_data": {
                    "study_id": f"study-{random.randint(1, 1000)}"
                }
            },
            headers=self.headers,
            name="/ml/predict",
            catch_response=True
        )


class AdminUser(APIUser):
    """Simulates an admin performing system tasks."""
    
    weight = 1  # 10% of users are admins
    
    @task(3)
    def check_system_health(self):
        """Check system health."""
        self.client.get("/health", name="/health")
    
    @task(2)
    def get_metrics(self):
        """Get Prometheus metrics."""
        self.client.get("/metrics", name="/metrics")
    
    @task(1)
    def check_job_status(self):
        """Check background job status."""
        job_id = f"job-{random.randint(1, 100)}"
        self.client.get(
            f"/jobs/{job_id}",
            headers=self.headers,
            name="/jobs/{id}",
            catch_response=True
        )


class RadiologyWorkflowUser(APIUser):
    """Simulates a complete radiology workflow."""
    
    weight = 2  # 20% of users follow full workflow
    
    @task
    def radiology_workflow(self):
        """Complete radiology workflow: search -> view -> annotate -> report."""
        # Step 1: Get worklist
        worklist_response = self.client.get(
            "/worklists",
            headers=self.headers,
            name="workflow: get worklist"
        )
        
        # Step 2: Search for studies
        search_response = self.client.post(
            "/search",
            json={
                "query": "chest",
                "modalities": ["X-Ray"],
                "page": 1,
                "page_size": 5
            },
            headers=self.headers,
            name="workflow: search"
        )
        
        # Step 3: View study details (simulate)
        self.client.get(
            f"/studies/study-{random.randint(1, 100)}",
            headers=self.headers,
            name="workflow: view study",
            catch_response=True
        )
        
        # Step 4: Request AI assistance
        self.client.post(
            "/ml/predict",
            json={
                "model_name": "chest-xray-classifier",
                "input_data": {"study_id": "study-123"}
            },
            headers=self.headers,
            name="workflow: AI prediction",
            catch_response=True
        )
        
        # Step 5: Update worklist item (simulate completion)
        self.client.patch(
            f"/worklists/items/item-{random.randint(1, 100)}",
            json={"status": "completed"},
            headers=self.headers,
            name="workflow: complete item",
            catch_response=True
        )


# Custom event handlers for detailed reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("="*60)
    print("LOAD TEST STARTED")
    print("="*60)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'Unknown'}")
    print("="*60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("="*60)
    print("LOAD TEST COMPLETED")
    print("="*60)
    
    stats = environment.stats
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {stats.total.min_response_time:.2f}ms")
    print(f"Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")
    print(f"Failure Rate: {stats.total.fail_ratio * 100:.2f}%")
    
    print("\n--- Top 10 Slowest Endpoints ---")
    sorted_stats = sorted(
        environment.stats.entries.values(),
        key=lambda x: x.avg_response_time,
        reverse=True
    )
    for stat in sorted_stats[:10]:
        print(f"{stat.name}: {stat.avg_response_time:.2f}ms (p95: {stat.get_response_time_percentile(0.95):.2f}ms)")
    
    print("="*60)


if __name__ == "__main__":
    import os
    os.system("locust -f load_test.py --host=http://localhost:8000")
