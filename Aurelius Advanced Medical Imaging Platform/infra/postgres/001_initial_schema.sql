-- Aurelius Medical Imaging Platform
-- Initial Schema Migration
-- Version: 001
-- Date: 2025-01-27

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- USERS & ORGANIZATIONS
-- ============================================================================

CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('hospital', 'clinic', 'lab', 'university', 'research')),
    email VARCHAR(255),
    phone VARCHAR(50),
    address JSONB,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    keycloak_id VARCHAR(255) UNIQUE NOT NULL,
    organization_id UUID REFERENCES organizations(id),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    roles VARCHAR(50)[] DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_keycloak_id ON users(keycloak_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_organization ON users(organization_id);

-- ============================================================================
-- PATIENTS & STUDIES
-- ============================================================================

CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(255) UNIQUE NOT NULL,  -- Medical Record Number
    patient_id_hash VARCHAR(64) UNIQUE,  -- SHA-256 for de-identified matching
    organization_id UUID REFERENCES organizations(id),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    date_of_birth DATE,
    sex VARCHAR(10) CHECK (sex IN ('M', 'F', 'O', 'U')),
    metadata JSONB DEFAULT '{}',
    is_deidentified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_patients_patient_id ON patients(patient_id);
CREATE INDEX idx_patients_hash ON patients(patient_id_hash);
CREATE INDEX idx_patients_org ON patients(organization_id);

CREATE TABLE studies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_instance_uid VARCHAR(255) UNIQUE NOT NULL,
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id),
    accession_number VARCHAR(255),
    study_date DATE,
    study_time TIME,
    study_description TEXT,
    modality VARCHAR(50),
    referring_physician VARCHAR(255),
    number_of_series INTEGER DEFAULT 0,
    number_of_instances INTEGER DEFAULT 0,
    orthanc_id VARCHAR(255),
    storage_location VARCHAR(500),
    storage_size_bytes BIGINT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_studies_patient ON studies(patient_id);
CREATE INDEX idx_studies_uid ON studies(study_instance_uid);
CREATE INDEX idx_studies_org ON studies(organization_id);
CREATE INDEX idx_studies_date ON studies(study_date);
CREATE INDEX idx_studies_modality ON studies(modality);
CREATE INDEX idx_studies_accession ON studies(accession_number);

CREATE TABLE series (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_instance_uid VARCHAR(255) UNIQUE NOT NULL,
    study_id UUID REFERENCES studies(id) ON DELETE CASCADE,
    series_number INTEGER,
    series_description TEXT,
    modality VARCHAR(50),
    body_part_examined VARCHAR(255),
    number_of_instances INTEGER DEFAULT 0,
    orthanc_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_series_study ON series(study_id);
CREATE INDEX idx_series_uid ON series(series_instance_uid);
CREATE INDEX idx_series_modality ON series(modality);

CREATE TABLE instances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sop_instance_uid VARCHAR(255) UNIQUE NOT NULL,
    series_id UUID REFERENCES series(id) ON DELETE CASCADE,
    instance_number INTEGER,
    acquisition_date DATE,
    acquisition_time TIME,
    orthanc_id VARCHAR(255),
    storage_path VARCHAR(500),
    file_size_bytes BIGINT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_instances_series ON instances(series_id);
CREATE INDEX idx_instances_uid ON instances(sop_instance_uid);

-- ============================================================================
-- WHOLE SLIDE IMAGING (WSI)
-- ============================================================================

CREATE TABLE slides (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slide_id VARCHAR(255) UNIQUE NOT NULL,
    patient_id UUID REFERENCES patients(id),
    organization_id UUID REFERENCES organizations(id),
    specimen_type VARCHAR(100),
    stain VARCHAR(100),
    magnification VARCHAR(50),
    scan_date TIMESTAMP WITH TIME ZONE,
    scanner_model VARCHAR(255),
    width_pixels INTEGER,
    height_pixels INTEGER,
    mpp_x FLOAT,  -- microns per pixel
    mpp_y FLOAT,
    tile_size INTEGER DEFAULT 256,
    pyramid_levels INTEGER,
    storage_path VARCHAR(500),
    file_size_bytes BIGINT,
    file_format VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_slides_patient ON slides(patient_id);
CREATE INDEX idx_slides_org ON slides(organization_id);
CREATE INDEX idx_slides_stain ON slides(stain);

-- ============================================================================
-- GENERIC ASSETS (Non-DICOM files)
-- ============================================================================

CREATE TABLE assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_type VARCHAR(50) NOT NULL CHECK (asset_type IN ('image', 'video', 'signal', 'document', 'other')),
    patient_id UUID REFERENCES patients(id),
    organization_id UUID REFERENCES organizations(id),
    study_id UUID REFERENCES studies(id),
    file_name VARCHAR(500),
    file_type VARCHAR(100),
    mime_type VARCHAR(100),
    storage_path VARCHAR(500),
    file_size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_assets_type ON assets(asset_type);
CREATE INDEX idx_assets_patient ON assets(patient_id);
CREATE INDEX idx_assets_study ON assets(study_id);

-- ============================================================================
-- TIME-SERIES SIGNALS (ECG, EEG, etc.)
-- ============================================================================

CREATE TABLE recordings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_type VARCHAR(50) NOT NULL CHECK (recording_type IN ('ecg', 'eeg', 'emg', 'eog', 'pcg', 'other')),
    patient_id UUID REFERENCES patients(id),
    organization_id UUID REFERENCES organizations(id),
    recording_date TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT,
    sampling_rate INTEGER,
    number_of_channels INTEGER,
    storage_path VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('recordings', 'recording_date', if_not_exists => TRUE);

CREATE TABLE signal_segments (
    id UUID DEFAULT uuid_generate_v4(),
    recording_id UUID REFERENCES recordings(id) ON DELETE CASCADE,
    channel_index INTEGER,
    segment_start TIMESTAMP WITH TIME ZONE NOT NULL,
    segment_end TIMESTAMP WITH TIME ZONE,
    data_blob BYTEA,  -- Compressed signal data
    annotations JSONB DEFAULT '{}',
    PRIMARY KEY (recording_id, segment_start, channel_index)
);

SELECT create_hypertable('signal_segments', 'segment_start', if_not_exists => TRUE);

-- ============================================================================
-- ANNOTATIONS & MEASUREMENTS
-- ============================================================================

CREATE TABLE annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    annotation_type VARCHAR(50) NOT NULL CHECK (annotation_type IN ('dicom', 'wsi', 'signal', 'general')),
    target_id UUID NOT NULL,  -- References studies, slides, recordings, etc.
    target_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id),
    label VARCHAR(255),
    coordinates JSONB,  -- Spatial coordinates, bounding boxes, polygons
    properties JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1,
    parent_id UUID REFERENCES annotations(id),  -- For version history
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_annotations_target ON annotations(target_id, target_type);
CREATE INDEX idx_annotations_user ON annotations(user_id);
CREATE INDEX idx_annotations_type ON annotations(annotation_type);

-- ============================================================================
-- ML MODELS & PREDICTIONS
-- ============================================================================

CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL CHECK (model_type IN ('classification', 'segmentation', 'detection', 'regression', 'generation')),
    framework VARCHAR(50),
    mlflow_run_id VARCHAR(255),
    storage_path VARCHAR(500),
    input_schema JSONB,
    output_schema JSONB,
    metrics JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
);

CREATE INDEX idx_ml_models_name ON ml_models(model_name);
CREATE INDEX idx_ml_models_type ON ml_models(model_type);

CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ml_models(id),
    target_id UUID NOT NULL,
    target_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id),
    prediction_type VARCHAR(50),
    results JSONB NOT NULL,
    confidence FLOAT,
    inference_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_model ON predictions(model_id);
CREATE INDEX idx_predictions_target ON predictions(target_id, target_type);
CREATE INDEX idx_predictions_created ON predictions(created_at DESC);

-- ============================================================================
-- WORKLISTS & TASKS
-- ============================================================================

CREATE TABLE worklists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    worklist_type VARCHAR(50) CHECK (worklist_type IN ('radiology', 'pathology', 'research', 'qa', 'teaching')),
    organization_id UUID REFERENCES organizations(id),
    description TEXT,
    filters JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE worklist_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    worklist_id UUID REFERENCES worklists(id) ON DELETE CASCADE,
    study_id UUID REFERENCES studies(id),
    slide_id UUID REFERENCES slides(id),
    assigned_to UUID REFERENCES users(id),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),
    priority INTEGER DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    due_date TIMESTAMP WITH TIME ZONE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_worklist_items_worklist ON worklist_items(worklist_id);
CREATE INDEX idx_worklist_items_status ON worklist_items(status);
CREATE INDEX idx_worklist_items_assigned ON worklist_items(assigned_to);

-- ============================================================================
-- AUDIT LOG (Append-only)
-- ============================================================================

CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    user_id UUID,
    keycloak_id VARCHAR(255),
    organization_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    details JSONB DEFAULT '{}',
    phi_accessed BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('audit_log', 'event_time', if_not_exists => TRUE);
CREATE INDEX idx_audit_user ON audit_log(user_id, event_time DESC);
CREATE INDEX idx_audit_action ON audit_log(action, event_time DESC);
CREATE INDEX idx_audit_resource ON audit_log(resource_type, resource_id, event_time DESC);
CREATE INDEX idx_audit_phi ON audit_log(phi_accessed, event_time DESC) WHERE phi_accessed = TRUE;

-- ============================================================================
-- CONSENT & COMPLIANCE
-- ============================================================================

CREATE TABLE consent_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id),
    consent_type VARCHAR(100) NOT NULL CHECK (consent_type IN ('treatment', 'research', 'data_sharing', 'teaching', 'ai_training')),
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'withdrawn', 'expired')),
    granted_date DATE NOT NULL,
    expiry_date DATE,
    scope JSONB DEFAULT '{}',
    irb_protocol VARCHAR(255),
    document_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_consent_patient ON consent_records(patient_id);
CREATE INDEX idx_consent_type ON consent_records(consent_type, status);

-- ============================================================================
-- PROVENANCE & DATA LINEAGE
-- ============================================================================

CREATE TABLE provenance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    derived_id UUID NOT NULL,
    derived_type VARCHAR(50) NOT NULL,
    transformation VARCHAR(255),
    transformation_details JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_provenance_source ON provenance(source_id, source_type);
CREATE INDEX idx_provenance_derived ON provenance(derived_id, derived_type);

-- ============================================================================
-- JOBS & BACKGROUND TASKS
-- ============================================================================

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    user_id UUID REFERENCES users(id),
    input_params JSONB DEFAULT '{}',
    output_results JSONB,
    error_message TEXT,
    progress_percent INTEGER DEFAULT 0 CHECK (progress_percent BETWEEN 0 AND 100),
    celery_task_id VARCHAR(255),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_user ON jobs(user_id);
CREATE INDEX idx_jobs_type ON jobs(job_type);

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to all relevant tables
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON patients FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_studies_updated_at BEFORE UPDATE ON studies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_series_updated_at BEFORE UPDATE ON series FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_instances_updated_at BEFORE UPDATE ON instances FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_slides_updated_at BEFORE UPDATE ON slides FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_assets_updated_at BEFORE UPDATE ON assets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_annotations_updated_at BEFORE UPDATE ON annotations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_worklists_updated_at BEFORE UPDATE ON worklists FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_worklist_items_updated_at BEFORE UPDATE ON worklist_items FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_consent_records_updated_at BEFORE UPDATE ON consent_records FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Patient ID hashing function
CREATE OR REPLACE FUNCTION hash_patient_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.patient_id IS NOT NULL THEN
        NEW.patient_id_hash := encode(digest(NEW.patient_id, 'sha256'), 'hex');
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER hash_patient_id_trigger 
BEFORE INSERT OR UPDATE ON patients 
FOR EACH ROW 
WHEN (NEW.patient_id IS NOT NULL)
EXECUTE FUNCTION hash_patient_id();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default organization
INSERT INTO organizations (id, name, type, email) VALUES 
    ('00000000-0000-0000-0000-000000000001', 'Aurelius Medical Center', 'hospital', 'admin@aurelius.local');

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Aurelius database schema created successfully';
END $$;
