# Data Model

## Entity Relationship Diagram

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│Organizations │       │    Users     │       │   Patients   │
│──────────────│       │──────────────│       │──────────────│
│ id (PK)      │◄──────│ id (PK)      │       │ id (PK)      │
│ name         │   │   │ keycloak_id  │       │ patient_id   │
│ type         │   │   │ organization ├──────►│ organization │
│ email        │   │   │ username     │   │   │ first_name   │
│ address      │   │   │ email        │   │   │ last_name    │
└──────────────┘   │   │ roles[]      │   │   │ dob          │
                   │   └──────────────┘   │   │ sex          │
                   │           │           │   │ is_deidentif │
                   │           │           │   └──────┬───────┘
                   │           │           │          │
                   │           │           │          │
                   │           │           │   ┌──────▼───────┐
                   │           │           │   │   Studies    │
                   │           │           │   │──────────────│
                   │           │           └───┤ id (PK)      │
                   │           │               │ study_uid    │
                   │           │               │ patient_id   │
                   │           │               │ organization │
                   │           │               │ accession_no │
                   │           │               │ study_date   │
                   │           │               │ modality     │
                   │           │               │ num_series   │
                   │           │               └──────┬───────┘
                   │           │                      │
                   │           │               ┌──────▼───────┐
                   │           │               │    Series    │
                   │           │               │──────────────│
                   │           │               │ id (PK)      │
                   │           │               │ series_uid   │
                   │           │               │ study_id (FK)│
                   │           │               │ series_num   │
                   │           │               │ modality     │
                   │           │               │ num_instances│
                   │           │               └──────┬───────┘
                   │           │                      │
                   │           │               ┌──────▼───────┐
                   │           │               │  Instances   │
                   │           │               │──────────────│
                   │           │               │ id (PK)      │
                   │           │               │ sop_uid      │
                   │           │               │ series_id(FK)│
                   │           │               │ instance_num │
                   │           │               │ storage_path │
                   │           │               └──────────────┘
                   │           │
                   │           │               ┌──────────────┐
                   │           │               │  Annotations │
                   │           │               │──────────────│
                   │           └───────────────┤ id (PK)      │
                   │                           │ user_id (FK) │
                   │                           │ target_id    │
                   │                           │ target_type  │
                   │                           │ label        │
                   │                           │ coordinates  │
                   │                           └──────────────┘
                   │
                   │                           ┌──────────────┐
                   │                           │  ML Models   │
                   │                           │──────────────│
                   │                           │ id (PK)      │
                   │                           │ model_name   │
                   │                           │ model_version│
                   │                           │ model_type   │
                   │                           │ framework    │
                   │                           │ mlflow_run_id│
                   │                           └──────┬───────┘
                   │                                  │
                   │                           ┌──────▼───────┐
                   │                           │ Predictions  │
                   │                           │──────────────│
                   └───────────────────────────┤ id (PK)      │
                                               │ model_id (FK)│
                                               │ user_id (FK) │
                                               │ target_id    │
                                               │ results      │
                                               │ confidence   │
                                               └──────────────┘
```

## Core Tables

### organizations

Healthcare organizations, research institutions, or hospital departments.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| name | VARCHAR(255) | Organization name |
| type | VARCHAR(50) | 'hospital', 'clinic', 'lab', 'university', 'research' |
| email | VARCHAR(255) | Contact email |
| phone | VARCHAR(50) | Contact phone |
| address | JSONB | Structured address |
| settings | JSONB | Organization-specific settings |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |
| deleted_at | TIMESTAMPTZ | Soft delete timestamp |

**Indexes**: name

### users

System users linked to Keycloak identities.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| keycloak_id | VARCHAR(255) | Keycloak user ID (unique) |
| organization_id | UUID | FK to organizations |
| username | VARCHAR(255) | Username (unique) |
| email | VARCHAR(255) | Email (unique) |
| first_name | VARCHAR(255) | First name |
| last_name | VARCHAR(255) | Last name |
| roles | VARCHAR(50)[] | User roles |
| preferences | JSONB | User preferences |
| last_login_at | TIMESTAMPTZ | Last login time |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |

**Indexes**: keycloak_id, email, organization_id

### patients

Patient records with PHI and de-identification support.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| patient_id | VARCHAR(255) | Medical Record Number (MRN) - unique |
| patient_id_hash | VARCHAR(64) | SHA-256 hash for matching |
| organization_id | UUID | FK to organizations |
| first_name | VARCHAR(255) | First name |
| last_name | VARCHAR(255) | Last name |
| date_of_birth | DATE | Date of birth |
| sex | VARCHAR(10) | 'M', 'F', 'O', 'U' |
| metadata | JSONB | Additional patient data |
| is_deidentified | BOOLEAN | De-identification flag |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |
| deleted_at | TIMESTAMPTZ | Soft delete timestamp |

**Indexes**: patient_id, patient_id_hash, organization_id

**Triggers**: Auto-generates patient_id_hash from patient_id

### studies

DICOM studies.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| study_instance_uid | VARCHAR(255) | DICOM Study Instance UID (unique) |
| patient_id | UUID | FK to patients |
| organization_id | UUID | FK to organizations |
| accession_number | VARCHAR(255) | Accession number |
| study_date | DATE | Study date |
| study_time | TIME | Study time |
| study_description | TEXT | Study description |
| modality | VARCHAR(50) | Primary modality |
| referring_physician | VARCHAR(255) | Referring physician name |
| number_of_series | INTEGER | Number of series |
| number_of_instances | INTEGER | Total instances |
| orthanc_id | VARCHAR(255) | Orthanc internal ID |
| storage_location | VARCHAR(500) | MinIO path |
| storage_size_bytes | BIGINT | Total size |
| metadata | JSONB | DICOM tags |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |
| deleted_at | TIMESTAMPTZ | Soft delete timestamp |

**Indexes**: study_instance_uid, patient_id, organization_id, study_date, modality, accession_number

### series

DICOM series within studies.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| series_instance_uid | VARCHAR(255) | DICOM Series Instance UID (unique) |
| study_id | UUID | FK to studies |
| series_number | INTEGER | Series number |
| series_description | TEXT | Series description |
| modality | VARCHAR(50) | Modality |
| body_part_examined | VARCHAR(255) | Body part |
| number_of_instances | INTEGER | Instance count |
| orthanc_id | VARCHAR(255) | Orthanc internal ID |
| metadata | JSONB | DICOM tags |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |

**Indexes**: series_instance_uid, study_id, modality

### instances

Individual DICOM instances.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| sop_instance_uid | VARCHAR(255) | SOP Instance UID (unique) |
| series_id | UUID | FK to series |
| instance_number | INTEGER | Instance number |
| acquisition_date | DATE | Acquisition date |
| acquisition_time | TIME | Acquisition time |
| orthanc_id | VARCHAR(255) | Orthanc internal ID |
| storage_path | VARCHAR(500) | Storage path |
| file_size_bytes | BIGINT | File size |
| metadata | JSONB | DICOM tags |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |

**Indexes**: sop_instance_uid, series_id

### slides

Whole slide imaging (WSI) data.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| slide_id | VARCHAR(255) | Slide identifier (unique) |
| patient_id | UUID | FK to patients |
| organization_id | UUID | FK to organizations |
| specimen_type | VARCHAR(100) | Tissue type |
| stain | VARCHAR(100) | Staining method |
| magnification | VARCHAR(50) | Scanner magnification |
| scan_date | TIMESTAMPTZ | Scan timestamp |
| scanner_model | VARCHAR(255) | Scanner model |
| width_pixels | INTEGER | Image width |
| height_pixels | INTEGER | Image height |
| mpp_x | FLOAT | Microns per pixel (X) |
| mpp_y | FLOAT | Microns per pixel (Y) |
| tile_size | INTEGER | Tile size (default 256) |
| pyramid_levels | INTEGER | Number of pyramid levels |
| storage_path | VARCHAR(500) | Storage path |
| file_size_bytes | BIGINT | File size |
| file_format | VARCHAR(50) | File format (SVS, NDPI, etc.) |
| metadata | JSONB | Additional metadata |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |
| deleted_at | TIMESTAMPTZ | Soft delete timestamp |

**Indexes**: slide_id, patient_id, organization_id, stain

### annotations

Image annotations and measurements.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| annotation_type | VARCHAR(50) | 'dicom', 'wsi', 'signal', 'general' |
| target_id | UUID | Referenced entity ID |
| target_type | VARCHAR(50) | Referenced entity type |
| user_id | UUID | FK to users |
| label | VARCHAR(255) | Annotation label |
| coordinates | JSONB | Spatial coordinates |
| properties | JSONB | Additional properties |
| version | INTEGER | Version number |
| parent_id | UUID | FK to parent annotation (versioning) |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |
| deleted_at | TIMESTAMPTZ | Soft delete timestamp |

**Indexes**: (target_id, target_type), user_id, annotation_type

### ml_models

Machine learning model registry.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| model_name | VARCHAR(255) | Model name |
| model_version | VARCHAR(50) | Version string |
| model_type | VARCHAR(100) | 'classification', 'segmentation', 'detection', 'regression', 'generation' |
| framework | VARCHAR(50) | Framework (PyTorch, TensorFlow, etc.) |
| mlflow_run_id | VARCHAR(255) | MLflow run ID |
| storage_path | VARCHAR(500) | Model artifact path |
| input_schema | JSONB | Expected input format |
| output_schema | JSONB | Output format |
| metrics | JSONB | Performance metrics |
| status | VARCHAR(50) | 'active', 'deprecated', 'archived' |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |

**Indexes**: model_name, model_type, (model_name, model_version) unique

### predictions

ML inference results.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| model_id | UUID | FK to ml_models |
| target_id | UUID | Target entity ID |
| target_type | VARCHAR(50) | Target entity type |
| user_id | UUID | FK to users |
| prediction_type | VARCHAR(50) | Prediction category |
| results | JSONB | Prediction results |
| confidence | FLOAT | Confidence score |
| inference_time_ms | INTEGER | Inference duration |
| created_at | TIMESTAMPTZ | Creation timestamp |

**Indexes**: model_id, (target_id, target_type), created_at DESC

### audit_log (TimescaleDB Hypertable)

Append-only audit trail.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| event_time | TIMESTAMPTZ | Event timestamp (partition key) |
| user_id | UUID | User ID |
| keycloak_id | VARCHAR(255) | Keycloak ID |
| organization_id | UUID | Organization ID |
| action | VARCHAR(100) | Action performed |
| resource_type | VARCHAR(100) | Resource type |
| resource_id | UUID | Resource ID |
| ip_address | INET | Client IP |
| user_agent | TEXT | User agent string |
| details | JSONB | Additional details |
| phi_accessed | BOOLEAN | PHI access flag |

**Indexes**: (user_id, event_time DESC), (action, event_time DESC), (resource_type, resource_id, event_time DESC), (phi_accessed, event_time DESC) WHERE phi_accessed = TRUE

**Partitioning**: By event_time (monthly chunks)

## JSONB Schemas

### Patient Metadata

```json
{
  "mrn": "string",
  "external_ids": ["string"],
  "insurance": {
    "provider": "string",
    "policy_number": "string"
  },
  "contact": {
    "phone": "string",
    "email": "string",
    "address": {
      "street": "string",
      "city": "string",
      "state": "string",
      "zip": "string"
    }
  }
}
```

### Study Metadata (DICOM Tags)

```json
{
  "StudyDescription": "string",
  "ReferringPhysicianName": "string",
  "StudyDate": "YYYYMMDD",
  "StudyTime": "HHMMSS",
  "Modality": "string",
  "InstitutionName": "string",
  "ProtocolName": "string"
}
```

### Annotation Coordinates

```json
{
  "type": "polygon",
  "points": [
    {"x": 100, "y": 200},
    {"x": 150, "y": 250}
  ],
  "bounding_box": {
    "x": 100,
    "y": 200,
    "width": 50,
    "height": 50
  }
}
```

### Prediction Results

```json
{
  "class": "normal",
  "probability": 0.95,
  "scores": {
    "normal": 0.95,
    "abnormal": 0.05
  },
  "segmentation": {
    "mask_url": "s3://...",
    "overlay_url": "s3://..."
  }
}
```

## Time-Series Data (TimescaleDB)

### recordings

Signal recordings (ECG, EEG, etc.).

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| recording_type | VARCHAR(50) | Signal type |
| patient_id | UUID | FK to patients |
| organization_id | UUID | FK to organizations |
| recording_date | TIMESTAMPTZ | Recording timestamp (partition key) |
| duration_seconds | FLOAT | Total duration |
| sampling_rate | INTEGER | Samples per second |
| number_of_channels | INTEGER | Channel count |
| storage_path | VARCHAR(500) | Storage path |
| metadata | JSONB | Additional metadata |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |

**Hypertable**: Partitioned by recording_date

### signal_segments

Compressed signal data.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Composite primary key part |
| recording_id | UUID | FK to recordings |
| channel_index | INTEGER | Channel number |
| segment_start | TIMESTAMPTZ | Segment start (partition key) |
| segment_end | TIMESTAMPTZ | Segment end |
| data_blob | BYTEA | Compressed signal data |
| annotations | JSONB | Segment annotations |

**Hypertable**: Partitioned by segment_start
**Primary Key**: (recording_id, segment_start, channel_index)

## Relationships

- **One-to-Many**:
  - organizations → users
  - organizations → patients
  - patients → studies
  - studies → series
  - series → instances
  - users → annotations
  - ml_models → predictions

- **Many-to-One**:
  - studies → patients
  - predictions → models
  - annotations → users

- **Polymorphic**:
  - annotations → (studies, slides, recordings)
  - predictions → (studies, slides, instances)

## Indexes and Performance

### Full-Text Search Indexes

```sql
CREATE INDEX idx_studies_description_fts ON studies USING gin(to_tsvector('english', study_description));
CREATE INDEX idx_patients_name_fts ON patients USING gin(to_tsvector('english', first_name || ' ' || last_name));
```

### Composite Indexes

```sql
CREATE INDEX idx_studies_patient_date ON studies(patient_id, study_date DESC);
CREATE INDEX idx_predictions_model_created ON predictions(model_id, created_at DESC);
```

### Partial Indexes

```sql
CREATE INDEX idx_audit_phi ON audit_log(event_time DESC) WHERE phi_accessed = TRUE;
CREATE INDEX idx_active_models ON ml_models(model_name) WHERE status = 'active';
```

## Data Retention

| Table | Retention | Policy |
|-------|-----------|--------|
| audit_log | 7 years | HIPAA requirement |
| studies | Indefinite | Per institutional policy |
| predictions | 5 years | Archive to cold storage |
| jobs | 90 days | Delete after completion |

## Backup Strategy

- **Full Backup**: Daily at 2 AM
- **Incremental**: Every 6 hours
- **Point-in-Time Recovery**: 30 days
- **Cross-Region Replication**: Real-time for critical tables

---

*For schema migrations, see `/infra/docker/migrations/`*
*For SQL queries, see `/docs/QUERIES.md` (to be created)*
