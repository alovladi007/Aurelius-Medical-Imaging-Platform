"""Search Service - Full-text search and indexing."""
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Aurelius Search Service",
    version="1.0.0",
    description="Full-text search and indexing service"
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "search-svc",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "search-svc",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/search")
async def search():
    """Search endpoint."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "Search endpoint - implementation in progress"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
