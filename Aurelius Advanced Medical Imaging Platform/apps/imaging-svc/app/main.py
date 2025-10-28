"""Imaging Service - DICOM processing and image management."""
import os
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Aurelius Imaging Service",
    version="1.0.0",
    description="DICOM processing and medical image management"
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "imaging-svc",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "imaging-svc",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post("/process-dicom")
async def process_dicom():
    """Process DICOM files."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "DICOM processing endpoint - implementation in progress"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
