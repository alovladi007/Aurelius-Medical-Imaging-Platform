# Aurelius Medical Imaging Platform - Security & Compliance

## Security Overview

The Aurelius platform implements defense-in-depth security principles with multiple layers of protection for Protected Health Information (PHI) and medical imaging data.

## Authentication & Authorization

### Identity Provider: Keycloak

- **Protocol**: OAuth 2.0 / OpenID Connect (OIDC)
- **Token Type**: JWT (JSON Web Tokens)
- **Token Expiry**: 30 minutes (configurable)
- **Refresh Token Expiry**: 24 hours (configurable)
- **Password Policy**:
  - Minimum 8 characters
  - Password history: 3
  - Max failed attempts: 5 (account lockout)
  - Lockout duration: 15 minutes

### Role-Based Access Control (RBAC)

| Role | Permissions | Use Case |
|------|-------------|----------|
| **admin** | Full system access, user management, configuration | System administrators |
| **clinician** | Patient data access, worklists, reports | Physicians, nurses |
| **radiologist** | DICOM studies, imaging worklists, reporting | Radiologists |
| **pathologist** | WSI slides, pathology worklists, annotations | Pathologists |
| **researcher** | De-identified data only, ML models, analysis | Research scientists |
| **ml-engineer** | Model training, deployment, evaluation | Data scientists |
| **student** | Educational access, limited data | Medical students |

### Authorization Flow

```
User → Frontend → API Gateway → Keycloak (validate JWT)
                              → Check roles
                              → Route to service
```

## Data Protection

### Encryption

#### At Rest
- **Database**: Transparent Data Encryption (TDE) in PostgreSQL
- **Object Storage**: MinIO server-side encryption (SSE)
- **Algorithm**: AES-256-GCM
- **Key Management**: HashiCorp Vault (production)

#### In Transit
- **External**: TLS 1.3 (HTTPS)
- **Internal**: TLS 1.2+ between services (production)
- **Certificate Management**: Let's Encrypt / cert-manager (K8s)

### PHI Zones

#### Zone 1: Identified PHI
- **Location**: Production database, MinIO `dicom-studies` bucket
- **Access**: Clinicians only with audit logging
- **Retention**: Per institutional policy

#### Zone 2: De-identified Data
- **Location**: `processed-data` bucket, research database
- **Access**: Researchers without PHI access
- **De-identification**: DICOM tags stripped per HIPAA Safe Harbor

#### Zone 3: Public/Teaching
- **Location**: Public datasets, educational materials
- **Access**: Students, unauthenticated users
- **Content**: Fully anonymized, no PHI

## De-identification

### DICOM De-identification

**Method**: HIPAA Safe Harbor + Expert Determination

**Tags Removed** (Partial List):
- Patient Name (0010,0010)
- Patient ID (0010,0020)
- Patient Birth Date (0010,0030)
- Patient Address (0010,1040)
- Institution Name (0008,0080)
- Referring Physician (0008,0090)
- Performing Physician (0008,1050)
- Accession Number (0008,0050)

**Tags Retained**:
- Study Date (shifted randomly)
- Modality (0008,0060)
- Body Part (0018,0015)
- Imaging parameters

**Reversible Mapping**:
- Original Patient ID → De-identified ID mapping stored in Vault
- Encrypted with organization-specific key
- Accessible only by authorized personnel
- Audit logged

### WSI De-identification

- Filename sanitization (remove patient names)
- Embedded label detection and masking (ML-based)
- Barcode removal
- Metadata stripping

## Audit Logging

### Audit Events

All PHI access is logged to append-only table:

```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id UUID,
    keycloak_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    phi_accessed BOOLEAN DEFAULT FALSE
);
```

**Logged Actions**:
- User login/logout
- Patient data access
- Study retrieval
- Image viewing
- Data download/export
- De-identification operations
- Consent modifications
- User role changes

**Retention**: 7 years (HIPAA requirement)

**Immutability**: TimescaleDB hypertable with no UPDATE/DELETE permissions

## Network Security

### Network Segmentation

```
┌─────────────────────────────────────┐
│         DMZ (Public)                │
│  Load Balancer, WAF, CDN            │
└────────────┬────────────────────────┘
             │ HTTPS only
┌────────────▼────────────────────────┐
│     Application Layer               │
│  Frontend, API Gateway              │
│  - TLS termination                  │
│  - Rate limiting                    │
│  - DDoS protection                  │
└────────────┬────────────────────────┘
             │ Internal network
┌────────────▼────────────────────────┐
│     Service Layer                   │
│  Imaging, ML, ETL services          │
│  - No external access               │
│  - mTLS between services            │
└────────────┬────────────────────────┘
             │ Internal network
┌────────────▼────────────────────────┐
│     Data Layer                      │
│  PostgreSQL, Redis, MinIO           │
│  - Encrypted at rest                │
│  - No external access               │
│  - Backup to separate network       │
└─────────────────────────────────────┘
```

### Firewall Rules

**Ingress (Production)**:
- Port 443 (HTTPS) → Load Balancer only
- Port 22 (SSH) → Bastion host only (IP whitelisted)
- All other ports blocked

**Egress**:
- Outbound HTTPS for updates
- Restricted to approved domains

### DDoS Protection

- CloudFlare / AWS Shield
- Rate limiting: 100 requests/minute per IP
- Adaptive throttling during attacks

## Access Control Policies (OPA)

### Policy Examples

#### Study Access Policy
```rego
package aurelius.study

default allow = false

# Clinicians can access their patients
allow {
    input.user.role == "clinician"
    input.resource.patient_id in input.user.assigned_patients
}

# Researchers can access de-identified data
allow {
    input.user.role == "researcher"
    input.resource.is_deidentified == true
}

# Admins can access everything
allow {
    input.user.role == "admin"
}
```

#### Data Export Policy
```rego
package aurelius.export

default allow = false

# Only allow export if consent granted
allow {
    input.resource.consent_status == "active"
    input.action == "export"
    input.user.role in ["clinician", "researcher"]
}

# No PHI export for researchers without explicit approval
deny {
    input.user.role == "researcher"
    input.resource.contains_phi == true
    not input.resource.has_irb_approval
}
```

## Consent Management

### Consent Types

| Type | Description | Required For |
|------|-------------|--------------|
| **Treatment** | Use for clinical care | Viewing patient studies |
| **Research** | Use in research studies | Including in datasets |
| **Data Sharing** | Share with external entities | Cross-institution collaboration |
| **Teaching** | Use for education | Including in teaching files |
| **AI Training** | Use for ML model training | Model development |

### Consent Lifecycle

1. **Grant**: Patient signs consent form → Stored in database + document
2. **Verify**: System checks consent before PHI access
3. **Revoke**: Patient can withdraw consent → Data access blocked
4. **Expire**: Auto-expiry based on consent duration

### Granular Consent

- Per-study consent
- Per-modality consent (e.g., allow CT, deny MRI)
- Time-limited consent
- Purpose-limited consent

## Vulnerability Management

### Security Scanning

**Tools**:
- **Container Scanning**: Trivy, Clair
- **Dependency Scanning**: Snyk, Dependabot
- **Code Analysis**: Bandit (Python), ESLint security plugin
- **SAST**: SonarQube
- **DAST**: OWASP ZAP

**Frequency**:
- Daily: Dependency checks
- Weekly: Container scans
- Monthly: Penetration testing
- Quarterly: Third-party security audit

### Patch Management

- **Critical Vulnerabilities**: 24-48 hour SLA
- **High Severity**: 7 day SLA
- **Medium/Low**: Next maintenance window

## Incident Response

### Breach Notification

**HIPAA Breach Notification Rule**:
1. **Discover breach** → Document immediately
2. **Assess scope** → Determine affected individuals
3. **Notify affected** → Within 60 days
4. **Notify HHS** → Within 60 days (if >500 individuals)
5. **Notify media** → If >500 individuals in same state

### Incident Response Plan

1. **Detection**: Automated alerts, user reports
2. **Containment**: Isolate affected systems
3. **Eradication**: Remove threat, patch vulnerability
4. **Recovery**: Restore from backup, verify integrity
5. **Lessons Learned**: Post-mortem, update procedures

### Forensics

- Audit logs preserved
- System snapshots taken
- Chain of custody maintained
- External forensics firm engaged if needed

## Secrets Management

### HashiCorp Vault

**Stored Secrets**:
- Database credentials
- API keys
- Encryption keys
- De-identification mapping keys
- Certificate private keys

**Access Policy**:
- Services authenticate via Kubernetes service accounts
- Time-limited tokens
- Audit logging of all secret access
- Automatic secret rotation

## Compliance

### HIPAA Compliance

**Technical Safeguards** (45 CFR § 164.312):
- [x] Access Control (Unique user IDs, emergency access)
- [x] Audit Controls (Audit logs, PHI access tracking)
- [x] Integrity (Data integrity checks, checksums)
- [x] Person/Entity Authentication (Keycloak OIDC)
- [x] Transmission Security (TLS 1.3)

**Administrative Safeguards** (45 CFR § 164.308):
- [x] Security Management Process
- [x] Workforce Security
- [ ] Information Access Management (partial)
- [ ] Security Awareness Training (to be implemented)
- [ ] Security Incident Procedures (documented)

**Physical Safeguards** (45 CFR § 164.310):
- [ ] Facility Access Controls (cloud provider responsibility)
- [ ] Workstation Security (to be documented)
- [ ] Device and Media Controls (to be implemented)

### GDPR Compliance (if applicable)

**Data Subject Rights**:
- Right to access: Export API
- Right to erasure: Soft delete + anonymization
- Right to rectification: Update APIs
- Right to data portability: Export in standard formats
- Right to object: Consent withdrawal

**Data Protection Principles**:
- Lawfulness, fairness, transparency: Consent tracking
- Purpose limitation: Purpose-specific consent
- Data minimization: De-identification
- Accuracy: Data validation
- Storage limitation: Retention policies
- Integrity and confidentiality: Encryption

### FDA Compliance (for AI/ML models)

**Software as a Medical Device (SaMD)**:
- Risk classification
- Quality management system
- Design controls
- Validation testing
- Post-market surveillance

## Security Checklist

### Development
- [ ] Threat modeling completed
- [ ] Security review of architecture
- [ ] Secure coding guidelines followed
- [ ] Dependencies up to date
- [ ] Secrets not in code
- [ ] Input validation on all endpoints
- [ ] Output encoding to prevent XSS
- [ ] SQL parameterization (no string concatenation)

### Deployment
- [ ] TLS certificates valid
- [ ] Firewall rules configured
- [ ] Secrets in Vault
- [ ] Backup encryption enabled
- [ ] Monitoring and alerting configured
- [ ] Incident response plan documented
- [ ] Security contacts identified

### Operations
- [ ] Audit logs reviewed weekly
- [ ] Vulnerability scans run monthly
- [ ] Access reviews quarterly
- [ ] Disaster recovery tests quarterly
- [ ] Security training annual
- [ ] Compliance audit annual

## Security Contacts

- **Security Team**: security@aurelius-medical.io
- **Report Vulnerability**: security-reports@aurelius-medical.io (GPG key available)
- **Incident Response**: incident-response@aurelius-medical.io (24/7)

## References

- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Controls](https://www.cisecurity.org/controls/)
