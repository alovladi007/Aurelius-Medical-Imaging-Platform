"""Add multi-tenancy tables

Revision ID: 014_add_multitenancy
Revises: 013_add_fhir_mappings
Create Date: 2025-01-27 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '014_add_multitenancy'
down_revision = '013_add_fhir_mappings'
branch_labels = None
depends_on = None


def upgrade():
    # Create tenant_tier enum
    tenant_tier_enum = postgresql.ENUM(
        'free', 'starter', 'professional', 'enterprise',
        name='tenanttier'
    )
    tenant_tier_enum.create(op.get_bind())
    
    # Create tenant_status enum
    tenant_status_enum = postgresql.ENUM(
        'active', 'suspended', 'trial', 'cancelled',
        name='tenantstatus'
    )
    tenant_status_enum.create(op.get_bind())
    
    # Create billing_cycle enum
    billing_cycle_enum = postgresql.ENUM(
        'monthly', 'annual',
        name='billingcycle'
    )
    billing_cycle_enum.create(op.get_bind())
    
    # Create tenant_user_role enum
    tenant_user_role_enum = postgresql.ENUM(
        'owner', 'admin', 'member', 'viewer',
        name='tenantuserrole'
    )
    tenant_user_role_enum.create(op.get_bind())
    
    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), unique=True, nullable=False),
        sa.Column('domain', sa.String(255), nullable=True),
        sa.Column('tier', tenant_tier_enum, nullable=False, server_default='free'),
        sa.Column('status', tenant_status_enum, nullable=False, server_default='trial'),
        sa.Column('billing_cycle', billing_cycle_enum, server_default='monthly'),
        sa.Column('quota_api_calls', sa.Integer, server_default='10000'),
        sa.Column('quota_storage_gb', sa.Integer, server_default='10'),
        sa.Column('quota_gpu_hours', sa.Integer, server_default='5'),
        sa.Column('quota_users', sa.Integer, server_default='5'),
        sa.Column('quota_studies', sa.Integer, server_default='1000'),
        sa.Column('stripe_customer_id', sa.String(255), unique=True, nullable=True),
        sa.Column('stripe_subscription_id', sa.String(255), unique=True, nullable=True),
        sa.Column('billing_email', sa.String(255), nullable=True),
        sa.Column('trial_ends_at', sa.DateTime, nullable=True),
        sa.Column('subscription_started_at', sa.DateTime, nullable=True),
        sa.Column('subscription_ends_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('now()')),
        sa.Column('settings', postgresql.JSON, server_default='{}'),
        sa.Column('metadata', postgresql.JSON, server_default='{}'),
    )
    
    # Create indexes
    op.create_index('idx_tenant_slug', 'tenants', ['slug'])
    op.create_index('idx_tenant_status', 'tenants', ['status'])
    op.create_index('idx_tenant_tier', 'tenants', ['tier'])
    
    # Create tenant_users table
    op.create_table(
        'tenant_users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', tenant_user_role_enum, nullable=False, server_default='member'),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('invited_at', sa.DateTime, nullable=False, server_default=sa.text('now()')),
        sa.Column('joined_at', sa.DateTime, nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('tenant_id', 'user_id', name='uq_tenant_user'),
    )
    
    # Create indexes
    op.create_index('idx_tenant_user_tenant', 'tenant_users', ['tenant_id'])
    op.create_index('idx_tenant_user_user', 'tenant_users', ['user_id'])
    
    # Create usage_records table
    op.create_table(
        'usage_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('period_start', sa.DateTime, nullable=False),
        sa.Column('period_end', sa.DateTime, nullable=False),
        sa.Column('api_calls', sa.Integer, server_default='0'),
        sa.Column('storage_gb', sa.Numeric(10, 2), server_default='0'),
        sa.Column('gpu_hours', sa.Numeric(10, 2), server_default='0'),
        sa.Column('active_users', sa.Integer, server_default='0'),
        sa.Column('total_studies', sa.Integer, server_default='0'),
        sa.Column('cost_api_calls', sa.Integer, server_default='0'),
        sa.Column('cost_storage', sa.Integer, server_default='0'),
        sa.Column('cost_gpu', sa.Integer, server_default='0'),
        sa.Column('cost_total', sa.Integer, server_default='0'),
        sa.Column('breakdown', postgresql.JSON, server_default='{}'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Create indexes
    op.create_index('idx_usage_tenant_period', 'usage_records', ['tenant_id', 'period_start', 'period_end'])
    op.create_index('idx_usage_period', 'usage_records', ['period_start', 'period_end'])
    
    # Create invoices table
    op.create_table(
        'invoices',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('invoice_number', sa.String(50), unique=True, nullable=False),
        sa.Column('amount_cents', sa.Integer, nullable=False),
        sa.Column('currency', sa.String(3), server_default='USD', nullable=False),
        sa.Column('status', sa.String(50), server_default='draft'),
        sa.Column('period_start', sa.DateTime, nullable=False),
        sa.Column('period_end', sa.DateTime, nullable=False),
        sa.Column('due_date', sa.DateTime, nullable=True),
        sa.Column('paid_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('now()')),
        sa.Column('stripe_invoice_id', sa.String(255), unique=True, nullable=True),
        sa.Column('stripe_payment_intent_id', sa.String(255), nullable=True),
        sa.Column('line_items', postgresql.JSON, server_default='[]'),
        sa.Column('metadata', postgresql.JSON, server_default='{}'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Create indexes
    op.create_index('idx_invoice_tenant', 'invoices', ['tenant_id'])
    op.create_index('idx_invoice_period', 'invoices', ['period_start', 'period_end'])
    op.create_index('idx_invoice_status', 'invoices', ['status'])
    
    # Create tenant_invitations table
    op.create_table(
        'tenant_invitations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('role', tenant_user_role_enum, nullable=False, server_default='member'),
        sa.Column('token', sa.String(255), unique=True, nullable=False),
        sa.Column('invited_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('accepted_at', sa.DateTime, nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['invited_by'], ['users.id']),
    )
    
    # Create indexes
    op.create_index('idx_invitation_tenant', 'tenant_invitations', ['tenant_id'])
    op.create_index('idx_invitation_email', 'tenant_invitations', ['email'])
    op.create_index('idx_invitation_token', 'tenant_invitations', ['token'])
    
    # Add tenant_id to existing tables for multi-tenancy
    # Studies
    op.add_column('studies', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_studies_tenant', 'studies', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')
    op.create_index('idx_studies_tenant', 'studies', ['tenant_id'])
    
    # Annotations
    op.add_column('annotations', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_annotations_tenant', 'annotations', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')
    op.create_index('idx_annotations_tenant', 'annotations', ['tenant_id'])
    
    # ML Models
    op.add_column('ml_models', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_ml_models_tenant', 'ml_models', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')
    op.create_index('idx_ml_models_tenant', 'ml_models', ['tenant_id'])
    
    # Worklists
    op.add_column('worklists', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_worklists_tenant', 'worklists', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')
    op.create_index('idx_worklists_tenant', 'worklists', ['tenant_id'])
    
    # Add tenant_id to users table
    op.add_column('users', sa.Column('default_tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_users_default_tenant', 'users', 'tenants', ['default_tenant_id'], ['id'], ondelete='SET NULL')


def downgrade():
    # Drop foreign keys and columns from existing tables
    op.drop_constraint('fk_users_default_tenant', 'users', type_='foreignkey')
    op.drop_column('users', 'default_tenant_id')
    
    op.drop_index('idx_worklists_tenant', 'worklists')
    op.drop_constraint('fk_worklists_tenant', 'worklists', type_='foreignkey')
    op.drop_column('worklists', 'tenant_id')
    
    op.drop_index('idx_ml_models_tenant', 'ml_models')
    op.drop_constraint('fk_ml_models_tenant', 'ml_models', type_='foreignkey')
    op.drop_column('ml_models', 'tenant_id')
    
    op.drop_index('idx_annotations_tenant', 'annotations')
    op.drop_constraint('fk_annotations_tenant', 'annotations', type_='foreignkey')
    op.drop_column('annotations', 'tenant_id')
    
    op.drop_index('idx_studies_tenant', 'studies')
    op.drop_constraint('fk_studies_tenant', 'studies', type_='foreignkey')
    op.drop_column('studies', 'tenant_id')
    
    # Drop tables
    op.drop_table('tenant_invitations')
    op.drop_table('invoices')
    op.drop_table('usage_records')
    op.drop_table('tenant_users')
    op.drop_table('tenants')
    
    # Drop enums
    postgresql.ENUM(name='tenantuserrole').drop(op.get_bind())
    postgresql.ENUM(name='billingcycle').drop(op.get_bind())
    postgresql.ENUM(name='tenantstatus').drop(op.get_bind())
    postgresql.ENUM(name='tenanttier').drop(op.get_bind())
