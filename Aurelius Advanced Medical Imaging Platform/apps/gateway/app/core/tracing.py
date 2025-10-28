"""OpenTelemetry tracing configuration for Aurelius services.

This module provides a unified tracing setup for all microservices.
"""
import os
import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

logger = logging.getLogger(__name__)


def setup_tracing(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False
):
    """
    Setup OpenTelemetry tracing for a service.
    
    Args:
        service_name: Name of the service (e.g., "gateway", "imaging-svc")
        otlp_endpoint: OTLP collector endpoint (default: from env OTEL_EXPORTER_OTLP_ENDPOINT)
        enable_console_export: Export traces to console for debugging
    """
    # Get configuration from environment
    if otlp_endpoint is None:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
    
    # Create resource
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        "service.version": os.getenv("VERSION", "1.0.0")
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add OTLP exporter
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True  # Use secure=True in production with TLS
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"✅ OpenTelemetry OTLP exporter configured: {otlp_endpoint}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to configure OTLP exporter: {e}")
    
    # Optionally add console exporter for debugging
    if enable_console_export:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("✅ OpenTelemetry console exporter enabled")
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    logger.info(f"✅ OpenTelemetry tracing initialized for {service_name}")
    
    return provider


def instrument_app(app, db_engine=None):
    """
    Instrument a FastAPI app and its dependencies with OpenTelemetry.
    
    Args:
        app: FastAPI application instance
        db_engine: SQLAlchemy engine (optional)
    """
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    logger.info("✅ FastAPI instrumented")
    
    # Instrument httpx (for HTTP client calls)
    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("✅ HTTPX instrumented")
    except Exception as e:
        logger.warning(f"⚠️ Failed to instrument HTTPX: {e}")
    
    # Instrument SQLAlchemy if engine provided
    if db_engine:
        try:
            SQLAlchemyInstrumentor().instrument(engine=db_engine)
            logger.info("✅ SQLAlchemy instrumented")
        except Exception as e:
            logger.warning(f"⚠️ Failed to instrument SQLAlchemy: {e}")
    
    # Instrument Redis
    try:
        RedisInstrumentor().instrument()
        logger.info("✅ Redis instrumented")
    except Exception as e:
        logger.warning(f"⚠️ Failed to instrument Redis: {e}")


def get_tracer(name: str):
    """Get a tracer for custom spans."""
    return trace.get_tracer(name)


# Custom span helpers
def trace_function(span_name: str):
    """Decorator to trace a function with a custom span."""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage in service:
"""
from app.core.tracing import setup_tracing, instrument_app

# In lifespan or startup:
setup_tracing("my-service")
instrument_app(app, db_engine=engine)

# For custom spans:
from app.core.tracing import trace_function, get_tracer

@trace_function("process_image")
async def process_image(image_id: str):
    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("validate_image"):
        # Validation logic
        pass
    
    with tracer.start_as_current_span("extract_metadata"):
        # Metadata extraction
        pass
    
    return result
"""
