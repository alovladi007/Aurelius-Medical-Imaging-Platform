#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create additional databases
    CREATE DATABASE orthanc;
    CREATE DATABASE keycloak;
    CREATE DATABASE fhir;
    CREATE DATABASE mlflow;
    
    -- Create TimescaleDB extension in main database
    \c aurelius
    CREATE EXTENSION IF NOT EXISTS timescaledb;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_trgm";
    CREATE EXTENSION IF NOT EXISTS "btree_gin";
    
    -- Grant permissions
    GRANT ALL PRIVILEGES ON DATABASE orthanc TO postgres;
    GRANT ALL PRIVILEGES ON DATABASE keycloak TO postgres;
    GRANT ALL PRIVILEGES ON DATABASE fhir TO postgres;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;
    GRANT ALL PRIVILEGES ON DATABASE aurelius TO postgres;
    
    \echo 'Database initialization complete'
EOSQL
