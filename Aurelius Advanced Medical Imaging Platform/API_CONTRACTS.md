# API Contracts

## Overview

This document defines all API contracts for the Aurelius Medical Imaging Platform, including REST APIs (OpenAPI/Swagger) and gRPC service definitions.

## REST APIs

### API Gateway

**Base URL**: `http://localhost:8000`

**OpenAPI Spec**: http://localhost:8000/openapi.json

#### Authentication Endpoints

```yaml
POST /auth/login
  Request:
    username: string
    password: string
  Response:
    access_token: string
    refresh_token: string
    expires_in: integer

POST /auth/refresh
  Request:
    refresh_token: string
  Response:
    access_token: string
    refresh_token: string

GET /auth/me
  Response:
    sub: string
    username: string
    email: string
    roles: string[]
```

#### Study Management

```yaml
GET /studies
  Query Parameters:
    page: integer (default: 1)
    page_size: integer (default: 20, max: 100)
    patient_id: string (optional)
    modality: string (optional)
    date_from: date (optional)
    date_to: date (optional)
    search: string (optional)
  Response:
    total: integer
    page: integer
    page_size: integer
    studies: Study[]

GET /studies/{study_id}
  Path Parameters:
    study_id: UUID
  Response:
    Study object with full details
```

#### Imaging Endpoints

```yaml
POST /imaging/upload
  Content-Type: multipart/form-data
  Request:
    file: binary
  Response:
    job_id: UUID
    status: string
    message: string

GET /imaging/jobs/{job_id}
  Response:
    job_id: UUID
    status: string (pending|running|completed|failed)
    progress: integer (0-100)
    result: object
```

#### ML Endpoints

```yaml
POST /ml/predict
  Request:
    model_name: string
    model_version: string
    input_data: object
  Response:
    prediction_id: UUID
    results: object
    confidence: float
    inference_time_ms: integer

GET /ml/models
  Response:
    models: ModelInfo[]
```

### Imaging Service

**Base URL**: `http://localhost:8001`

Full API documentation available at: http://localhost:8001/docs

### ML Service

**Base URL**: `http://localhost:8002`

Full API documentation available at: http://localhost:8002/docs

## gRPC Services

### Imaging Service Proto

```protobuf
syntax = "proto3";

package aurelius.imaging;

service ImagingService {
  rpc IngestDICOM(DICOMIngestRequest) returns (IngestResponse);
  rpc QueryStudies(StudyQueryRequest) returns (StudyQueryResponse);
  rpc GetInstance(InstanceRequest) returns (InstanceResponse);
}

message DICOMIngestRequest {
  bytes dicom_data = 1;
  map<string, string> metadata = 2;
}

message IngestResponse {
  string job_id = 1;
  string status = 2;
}

message StudyQueryRequest {
  string patient_id = 1;
  string modality = 2;
  string date_range_start = 3;
  string date_range_end = 4;
}

message StudyQueryResponse {
  repeated Study studies = 1;
}

message Study {
  string study_instance_uid = 1;
  string patient_id = 2;
  string study_date = 3;
  string modality = 4;
  int32 number_of_series = 5;
}
```

### ML Service Proto

```protobuf
syntax = "proto3";

package aurelius.ml;

service MLService {
  rpc Predict(PredictionRequest) returns (PredictionResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
}

message PredictionRequest {
  string model_name = 1;
  string model_version = 2;
  bytes input_data = 3;
  map<string, string> parameters = 4;
}

message PredictionResponse {
  string prediction_id = 1;
  bytes results = 2;
  float confidence = 3;
  int32 inference_time_ms = 4;
}
```

## Data Models

### Common Types

```typescript
// UUID type
type UUID = string;

// ISO 8601 datetime
type DateTime = string;

// ISO 8601 date
type Date = string;

// Modality types
type Modality = 'CT' | 'MRI' | 'X-Ray' | 'US' | 'PET' | 'SPECT' | 'MG' | 'CR' | 'DX' | 'NM' | 'OT';

// Study status
type StudyStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';

// User roles
type Role = 'admin' | 'clinician' | 'researcher' | 'radiologist' | 'pathologist' | 'student' | 'ml-engineer';
```

### Study Model

```typescript
interface Study {
  id: UUID;
  study_instance_uid: string;
  patient_id: UUID;
  accession_number?: string;
  study_date?: Date;
  study_time?: string;
  study_description?: string;
  modality?: Modality;
  referring_physician?: string;
  number_of_series: number;
  number_of_instances: number;
  storage_location?: string;
  storage_size_bytes?: number;
  created_at: DateTime;
  updated_at: DateTime;
}
```

### Patient Model

```typescript
interface Patient {
  id: UUID;
  patient_id: string;
  patient_id_hash?: string;
  organization_id: UUID;
  first_name?: string;
  last_name?: string;
  date_of_birth?: Date;
  sex?: 'M' | 'F' | 'O' | 'U';
  is_deidentified: boolean;
  created_at: DateTime;
  updated_at: DateTime;
}
```

### Prediction Model

```typescript
interface Prediction {
  id: UUID;
  model_id: UUID;
  target_id: UUID;
  target_type: string;
  user_id: UUID;
  prediction_type: string;
  results: Record<string, any>;
  confidence?: number;
  inference_time_ms: number;
  created_at: DateTime;
}
```

## Error Responses

All APIs follow a consistent error response format:

```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "request_id": "uuid",
  "timestamp": "2025-01-27T10:00:00Z"
}
```

### Common HTTP Status Codes

- `200 OK`: Successful GET/PUT/PATCH
- `201 Created`: Successful POST
- `204 No Content`: Successful DELETE
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (e.g., duplicate)
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

- **Default**: 60 requests per minute per user
- **Burst**: Up to 100 requests allowed in 10 seconds
- **Headers**:
  - `X-RateLimit-Limit`: Total allowed
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Unix timestamp when limit resets

## Pagination

All list endpoints support pagination:

```
GET /api/resource?page=1&page_size=20
```

Response includes pagination metadata:

```json
{
  "total": 150,
  "page": 1,
  "page_size": 20,
  "items": [...]
}
```

## Versioning

APIs are versioned via URL path:

- Current: `/v1/endpoint`
- Next: `/v2/endpoint` (when breaking changes are needed)

## Authentication

All API requests (except `/auth/login`) require JWT bearer token:

```
Authorization: Bearer <access_token>
```

Tokens are obtained via the `/auth/login` endpoint and are valid for 30 minutes. Use `/auth/refresh` to get a new token without re-authenticating.

## WebSocket Endpoints

### Real-time Updates

```
ws://localhost:8000/ws
```

Messages:

```json
{
  "type": "study_update",
  "data": {
    "study_id": "uuid",
    "status": "completed"
  }
}
```

## DICOMweb Endpoints

### QIDO-RS (Query)

```
GET /imaging/dicomweb/studies?PatientID=123456
```

### WADO-RS (Retrieve)

```
GET /imaging/dicomweb/studies/{study_uid}
GET /imaging/dicomweb/studies/{study_uid}/series/{series_uid}
GET /imaging/dicomweb/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}
```

### STOW-RS (Store)

```
POST /imaging/dicomweb/studies
Content-Type: multipart/related
```

---

*For full OpenAPI specs, visit the `/docs` endpoint of each service.*

*For gRPC definitions, see `apps/gateway/proto/` directory.*
