"""Stripe billing integration service.

This service handles Stripe payment processing, subscription management,
and webhook events.
"""
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import stripe
from sqlalchemy.orm import Session
import uuid

from app.models.tenants import (
    Tenant, Invoice, UsageRecord, TenantTier, BillingCycle,
    TIER_PRICING, USAGE_COSTS
)
from app.services.tenant_service import TenantService

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_placeholder")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_placeholder")


class BillingService:
    """Service for billing and payment processing."""
    
    def __init__(self, db: Session):
        self.db = db
        self.tenant_service = TenantService(db)
    
    # ============================================================================
    # CUSTOMER MANAGEMENT
    # ============================================================================
    
    def create_stripe_customer(
        self,
        tenant: Tenant,
        email: str,
        name: Optional[str] = None
    ) -> str:
        """
        Create a Stripe customer for a tenant.
        
        Returns:
            Stripe customer ID
        """
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name or tenant.name,
                metadata={
                    "tenant_id": str(tenant.id),
                    "tenant_slug": tenant.slug,
                }
            )
            
            # Update tenant with Stripe customer ID
            tenant.stripe_customer_id = customer.id
            tenant.billing_email = email
            self.db.commit()
            
            logger.info(f"Created Stripe customer {customer.id} for tenant {tenant.slug}")
            
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating customer: {e}")
            raise
    
    def get_or_create_customer(self, tenant: Tenant, email: str) -> str:
        """Get existing or create new Stripe customer."""
        if tenant.stripe_customer_id:
            return tenant.stripe_customer_id
        
        return self.create_stripe_customer(tenant, email)
    
    # ============================================================================
    # SUBSCRIPTION MANAGEMENT
    # ============================================================================
    
    def create_subscription(
        self,
        tenant: Tenant,
        tier: TenantTier,
        billing_cycle: BillingCycle,
        payment_method_id: str
    ) -> Dict[str, Any]:
        """
        Create a subscription for a tenant.
        
        Args:
            tenant: Tenant to subscribe
            tier: Subscription tier
            billing_cycle: Monthly or annual
            payment_method_id: Stripe payment method ID
        
        Returns:
            Subscription details
        """
        try:
            # Get or create customer
            customer_id = self.get_or_create_customer(
                tenant,
                tenant.billing_email or "billing@example.com"
            )
            
            # Attach payment method to customer
            stripe.PaymentMethod.attach(
                payment_method_id,
                customer=customer_id
            )
            
            # Set as default payment method
            stripe.Customer.modify(
                customer_id,
                invoice_settings={"default_payment_method": payment_method_id}
            )
            
            # Get price based on tier and cycle
            pricing = TIER_PRICING[tier]
            amount = pricing["monthly_price"] if billing_cycle == BillingCycle.MONTHLY else pricing["annual_price"]
            
            # Create Stripe price if needed (in production, use pre-created prices)
            price = stripe.Price.create(
                unit_amount=amount,
                currency="usd",
                recurring={
                    "interval": "month" if billing_cycle == BillingCycle.MONTHLY else "year"
                },
                product_data={
                    "name": f"Aurelius {tier.value.title()} Plan",
                    "metadata": {
                        "tier": tier.value,
                        "billing_cycle": billing_cycle.value
                    }
                }
            )
            
            # Create subscription
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price.id}],
                metadata={
                    "tenant_id": str(tenant.id),
                    "tenant_slug": tenant.slug,
                    "tier": tier.value,
                }
            )
            
            # Update tenant
            tenant.stripe_subscription_id = subscription.id
            tenant.tier = tier
            tenant.billing_cycle = billing_cycle
            tenant.subscription_started_at = datetime.utcnow()
            
            # Update quotas
            tenant_service = TenantService(self.db)
            tenant_service.upgrade_tenant(tenant.id, tier)
            
            self.db.commit()
            
            logger.info(f"Created subscription {subscription.id} for tenant {tenant.slug}")
            
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "current_period_start": subscription.current_period_start,
                "current_period_end": subscription.current_period_end,
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating subscription: {e}")
            raise
    
    def cancel_subscription(
        self,
        tenant: Tenant,
        immediate: bool = False
    ) -> Dict[str, Any]:
        """
        Cancel a tenant's subscription.
        
        Args:
            tenant: Tenant to cancel
            immediate: If True, cancel immediately; otherwise at period end
        
        Returns:
            Cancellation details
        """
        if not tenant.stripe_subscription_id:
            raise ValueError("Tenant has no active subscription")
        
        try:
            if immediate:
                subscription = stripe.Subscription.delete(tenant.stripe_subscription_id)
            else:
                subscription = stripe.Subscription.modify(
                    tenant.stripe_subscription_id,
                    cancel_at_period_end=True
                )
            
            logger.info(f"Cancelled subscription for tenant {tenant.slug} (immediate={immediate})")
            
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "cancel_at": subscription.cancel_at,
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error cancelling subscription: {e}")
            raise
    
    # ============================================================================
    # USAGE BILLING
    # ============================================================================
    
    def calculate_usage_costs(self, usage: UsageRecord) -> Dict[str, int]:
        """
        Calculate costs based on usage.
        
        Returns:
            Dict with cost_api_calls, cost_storage, cost_gpu, cost_total (in cents)
        """
        cost_api_calls = int(usage.api_calls * USAGE_COSTS["api_call"])
        cost_storage = int(float(usage.storage_gb) * USAGE_COSTS["storage_gb"])
        cost_gpu = int(float(usage.gpu_hours) * USAGE_COSTS["gpu_hour"])
        cost_total = cost_api_calls + cost_storage + cost_gpu
        
        return {
            "cost_api_calls": cost_api_calls,
            "cost_storage": cost_storage,
            "cost_gpu": cost_gpu,
            "cost_total": cost_total,
        }
    
    def create_usage_invoice(
        self,
        tenant: Tenant,
        period_start: datetime,
        period_end: datetime
    ) -> Invoice:
        """
        Create an invoice for a tenant's usage.
        
        This is called at the end of each billing period to charge for overages.
        """
        # Get usage for period
        usage = self.db.query(UsageRecord).filter(
            UsageRecord.tenant_id == tenant.id,
            UsageRecord.period_start >= period_start,
            UsageRecord.period_end <= period_end
        ).first()
        
        if not usage:
            logger.warning(f"No usage record found for tenant {tenant.slug} in period {period_start} - {period_end}")
            return None
        
        # Calculate costs
        costs = self.calculate_usage_costs(usage)
        
        # Update usage record with costs
        usage.cost_api_calls = costs["cost_api_calls"]
        usage.cost_storage = costs["cost_storage"]
        usage.cost_gpu = costs["cost_gpu"]
        usage.cost_total = costs["cost_total"]
        
        # Create invoice
        invoice_number = f"INV-{tenant.slug.upper()}-{period_start.strftime('%Y%m')}"
        
        invoice = Invoice(
            tenant_id=tenant.id,
            invoice_number=invoice_number,
            amount_cents=costs["cost_total"],
            period_start=period_start,
            period_end=period_end,
            due_date=period_end + timedelta(days=7),
            line_items=[
                {
                    "description": "API Calls",
                    "quantity": usage.api_calls,
                    "unit_price": USAGE_COSTS["api_call"],
                    "amount": costs["cost_api_calls"]
                },
                {
                    "description": "Storage (GB-months)",
                    "quantity": float(usage.storage_gb),
                    "unit_price": USAGE_COSTS["storage_gb"],
                    "amount": costs["cost_storage"]
                },
                {
                    "description": "GPU Hours",
                    "quantity": float(usage.gpu_hours),
                    "unit_price": USAGE_COSTS["gpu_hour"],
                    "amount": costs["cost_gpu"]
                },
            ]
        )
        
        self.db.add(invoice)
        self.db.commit()
        self.db.refresh(invoice)
        
        logger.info(f"Created usage invoice {invoice_number} for ${costs['cost_total']/100:.2f}")
        
        return invoice
    
    def charge_invoice(self, invoice: Invoice) -> bool:
        """
        Charge an invoice via Stripe.
        
        Returns:
            True if payment succeeded
        """
        tenant = invoice.tenant
        
        if not tenant.stripe_customer_id:
            logger.error(f"Tenant {tenant.slug} has no Stripe customer")
            return False
        
        try:
            # Create Stripe invoice
            stripe_invoice = stripe.Invoice.create(
                customer=tenant.stripe_customer_id,
                description=f"Usage for {invoice.period_start.strftime('%B %Y')}",
                metadata={
                    "invoice_id": str(invoice.id),
                    "invoice_number": invoice.invoice_number,
                    "tenant_id": str(tenant.id),
                }
            )
            
            # Add line items
            for item in invoice.line_items:
                stripe.InvoiceItem.create(
                    customer=tenant.stripe_customer_id,
                    invoice=stripe_invoice.id,
                    description=item["description"],
                    quantity=int(item["quantity"]),
                    unit_amount=int(item["unit_price"]),
                )
            
            # Finalize and pay
            stripe_invoice = stripe.Invoice.finalize_invoice(stripe_invoice.id)
            stripe_invoice = stripe.Invoice.pay(stripe_invoice.id)
            
            # Update invoice
            invoice.stripe_invoice_id = stripe_invoice.id
            invoice.stripe_payment_intent_id = stripe_invoice.payment_intent
            invoice.status = "paid"
            invoice.paid_at = datetime.utcnow()
            
            self.db.commit()
            
            logger.info(f"Charged invoice {invoice.invoice_number} successfully")
            
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error charging invoice: {e}")
            invoice.status = "failed"
            self.db.commit()
            return False
    
    # ============================================================================
    # WEBHOOK HANDLING
    # ============================================================================
    
    def handle_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """
        Handle Stripe webhook events.
        
        Args:
            payload: Raw webhook payload
            signature: Stripe signature header
        
        Returns:
            Event processing result
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, STRIPE_WEBHOOK_SECRET
            )
        except ValueError as e:
            logger.error(f"Invalid webhook payload: {e}")
            raise
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid webhook signature: {e}")
            raise
        
        # Handle the event
        event_type = event["type"]
        data = event["data"]["object"]
        
        logger.info(f"Received Stripe webhook: {event_type}")
        
        if event_type == "customer.subscription.updated":
            return self._handle_subscription_updated(data)
        
        elif event_type == "customer.subscription.deleted":
            return self._handle_subscription_deleted(data)
        
        elif event_type == "invoice.paid":
            return self._handle_invoice_paid(data)
        
        elif event_type == "invoice.payment_failed":
            return self._handle_invoice_payment_failed(data)
        
        else:
            logger.info(f"Unhandled webhook event: {event_type}")
            return {"status": "unhandled", "event_type": event_type}
    
    def _handle_subscription_updated(self, subscription_data: Dict) -> Dict:
        """Handle subscription.updated webhook."""
        subscription_id = subscription_data["id"]
        
        tenant = self.db.query(Tenant).filter(
            Tenant.stripe_subscription_id == subscription_id
        ).first()
        
        if not tenant:
            logger.warning(f"Tenant not found for subscription {subscription_id}")
            return {"status": "not_found"}
        
        # Update tenant subscription status
        status = subscription_data["status"]
        logger.info(f"Subscription {subscription_id} status: {status}")
        
        return {"status": "processed", "tenant_id": str(tenant.id)}
    
    def _handle_subscription_deleted(self, subscription_data: Dict) -> Dict:
        """Handle subscription.deleted webhook."""
        subscription_id = subscription_data["id"]
        
        tenant = self.db.query(Tenant).filter(
            Tenant.stripe_subscription_id == subscription_id
        ).first()
        
        if not tenant:
            return {"status": "not_found"}
        
        # Downgrade to free tier
        tenant_service = TenantService(self.db)
        tenant_service.upgrade_tenant(tenant.id, TenantTier.FREE)
        
        logger.info(f"Downgraded tenant {tenant.slug} to free tier after subscription cancellation")
        
        return {"status": "processed", "tenant_id": str(tenant.id)}
    
    def _handle_invoice_paid(self, invoice_data: Dict) -> Dict:
        """Handle invoice.paid webhook."""
        stripe_invoice_id = invoice_data["id"]
        
        invoice = self.db.query(Invoice).filter(
            Invoice.stripe_invoice_id == stripe_invoice_id
        ).first()
        
        if invoice:
            invoice.status = "paid"
            invoice.paid_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Marked invoice {invoice.invoice_number} as paid")
        
        return {"status": "processed"}
    
    def _handle_invoice_payment_failed(self, invoice_data: Dict) -> Dict:
        """Handle invoice.payment_failed webhook."""
        stripe_invoice_id = invoice_data["id"]
        
        invoice = self.db.query(Invoice).filter(
            Invoice.stripe_invoice_id == stripe_invoice_id
        ).first()
        
        if invoice:
            invoice.status = "failed"
            self.db.commit()
            
            logger.warning(f"Invoice {invoice.invoice_number} payment failed")
        
        return {"status": "processed"}
    
    # ============================================================================
    # REPORTING
    # ============================================================================
    
    def get_tenant_billing_summary(
        self,
        tenant_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Get billing summary for a tenant."""
        tenant = self.tenant_service.get_tenant(tenant_id)
        if not tenant:
            return None
        
        # Get current usage
        usage = self.tenant_service.get_current_usage(tenant_id)
        
        # Get recent invoices
        invoices = self.db.query(Invoice).filter(
            Invoice.tenant_id == tenant_id
        ).order_by(Invoice.created_at.desc()).limit(12).all()
        
        # Calculate costs
        current_costs = self.calculate_usage_costs(usage) if usage else {
            "cost_api_calls": 0,
            "cost_storage": 0,
            "cost_gpu": 0,
            "cost_total": 0,
        }
        
        return {
            "tenant": {
                "id": str(tenant.id),
                "name": tenant.name,
                "tier": tenant.tier,
                "status": tenant.status,
            },
            "subscription": {
                "stripe_customer_id": tenant.stripe_customer_id,
                "stripe_subscription_id": tenant.stripe_subscription_id,
                "billing_cycle": tenant.billing_cycle,
            },
            "current_usage": {
                "api_calls": usage.api_calls if usage else 0,
                "storage_gb": float(usage.storage_gb) if usage else 0,
                "gpu_hours": float(usage.gpu_hours) if usage else 0,
                "estimated_cost": current_costs["cost_total"] / 100,  # Convert to dollars
            },
            "quotas": {
                "api_calls": tenant.quota_api_calls,
                "storage_gb": tenant.quota_storage_gb,
                "gpu_hours": tenant.quota_gpu_hours,
            },
            "recent_invoices": [
                {
                    "invoice_number": inv.invoice_number,
                    "amount": inv.amount_cents / 100,
                    "status": inv.status,
                    "period": {
                        "start": inv.period_start.isoformat(),
                        "end": inv.period_end.isoformat(),
                    }
                }
                for inv in invoices
            ]
        }
