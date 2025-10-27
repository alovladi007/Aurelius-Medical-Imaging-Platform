.PHONY: help bootstrap install generate up down restart logs clean test lint format migrate seed e2e

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Aurelius Medical Imaging Platform$(NC)"
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

bootstrap: ## Complete setup from scratch (install + generate + up + migrate + seed)
	@echo "$(BLUE)🚀 Bootstrapping Aurelius platform...$(NC)"
	@$(MAKE) install
	@$(MAKE) generate
	@$(MAKE) up
	@sleep 10
	@$(MAKE) migrate
	@$(MAKE) seed
	@echo "$(GREEN)✅ Bootstrap complete! Visit http://localhost:3000$(NC)"

install: ## Install all dependencies (Python + Node)
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@# Install Python dependencies
	@cd apps/gateway && pip install -r requirements.txt
	@cd apps/imaging-svc && pip install -r requirements.txt
	@cd apps/ml-svc && pip install -r requirements.txt
	@cd apps/etl-svc && pip install -r requirements.txt
	@# Install Node dependencies
	@cd apps/frontend && pnpm install
	@cd packages/ui && pnpm install
	@cd packages/shared-types && pnpm install
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

generate: ## Generate API clients and gRPC stubs
	@echo "$(BLUE)🔧 Generating API clients...$(NC)"
	@# Generate TypeScript client from OpenAPI
	@docker run --rm -v $(PWD):/local openapitools/openapi-generator-cli generate \
		-i /local/apps/gateway/openapi.yaml \
		-g typescript-axios \
		-o /local/apps/frontend/src/api/generated || true
	@# Generate Python gRPC stubs
	@python -m grpc_tools.protoc \
		-I./apps/gateway/proto \
		--python_out=./apps/imaging-svc/proto_gen \
		--grpc_python_out=./apps/imaging-svc/proto_gen \
		./apps/gateway/proto/*.proto || true
	@echo "$(GREEN)✅ Clients generated$(NC)"

up: ## Start all services
	@echo "$(BLUE)🐳 Starting services...$(NC)"
	@docker compose -f infra/docker/compose.yaml up -d
	@echo "$(GREEN)✅ Services started$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:3000$(NC)"
	@echo "$(YELLOW)Gateway: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Keycloak: http://localhost:8080$(NC)"

down: ## Stop all services
	@echo "$(BLUE)🛑 Stopping services...$(NC)"
	@docker compose -f infra/docker/compose.yaml down
	@echo "$(GREEN)✅ Services stopped$(NC)"

restart: down up ## Restart all services

logs: ## Tail logs from all services
	@docker compose -f infra/docker/compose.yaml logs -f

logs-gateway: ## Tail gateway logs
	@docker compose -f infra/docker/compose.yaml logs -f gateway

logs-imaging: ## Tail imaging-svc logs
	@docker compose -f infra/docker/compose.yaml logs -f imaging-svc

logs-ml: ## Tail ml-svc logs
	@docker compose -f infra/docker/compose.yaml logs -f ml-svc

clean: ## Remove all containers, volumes, and generated files
	@echo "$(RED)🧹 Cleaning up...$(NC)"
	@docker compose -f infra/docker/compose.yaml down -v
	@rm -rf apps/frontend/.next
	@rm -rf apps/frontend/node_modules
	@rm -rf apps/frontend/src/api/generated
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

test: ## Run all tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@cd apps/gateway && pytest -v
	@cd apps/imaging-svc && pytest -v
	@cd apps/ml-svc && pytest -v
	@cd apps/frontend && pnpm test
	@echo "$(GREEN)✅ All tests passed$(NC)"

test-unit: ## Run unit tests only
	@cd apps/gateway && pytest -v -m "not integration"
	@cd apps/imaging-svc && pytest -v -m "not integration"
	@cd apps/ml-svc && pytest -v -m "not integration"

test-integration: ## Run integration tests
	@cd apps/gateway && pytest -v -m integration
	@cd apps/imaging-svc && pytest -v -m integration

e2e: ## Run end-to-end tests
	@echo "$(BLUE)🔄 Running E2E tests...$(NC)"
	@cd apps/frontend && pnpm test:e2e
	@echo "$(GREEN)✅ E2E tests passed$(NC)"

lint: ## Run linters
	@echo "$(BLUE)🔍 Linting code...$(NC)"
	@cd apps/gateway && ruff check .
	@cd apps/imaging-svc && ruff check .
	@cd apps/ml-svc && ruff check .
	@cd apps/frontend && pnpm lint
	@echo "$(GREEN)✅ Linting complete$(NC)"

format: ## Format code
	@echo "$(BLUE)✨ Formatting code...$(NC)"
	@cd apps/gateway && black . && ruff check --fix .
	@cd apps/imaging-svc && black . && ruff check --fix .
	@cd apps/ml-svc && black . && ruff check --fix .
	@cd apps/frontend && pnpm format
	@echo "$(GREEN)✅ Formatting complete$(NC)"

typecheck: ## Run type checking
	@echo "$(BLUE)📝 Type checking...$(NC)"
	@cd apps/gateway && mypy .
	@cd apps/imaging-svc && mypy .
	@cd apps/ml-svc && mypy .
	@cd apps/frontend && pnpm typecheck
	@echo "$(GREEN)✅ Type checking complete$(NC)"

migrate: ## Run database migrations
	@echo "$(BLUE)📊 Running migrations...$(NC)"
	@docker compose -f infra/docker/compose.yaml exec -T postgres psql -U postgres -d aurelius -f /migrations/001_initial_schema.sql || true
	@echo "$(GREEN)✅ Migrations complete$(NC)"

migrate-create: ## Create a new migration
	@read -p "Migration name: " name; \
	timestamp=$$(date +%s); \
	touch "infra/docker/migrations/$${timestamp}_$${name}.sql"

seed: ## Seed database with sample data
	@echo "$(BLUE)🌱 Seeding database...$(NC)"
	@python scripts/seed_data.py
	@echo "$(GREEN)✅ Database seeded$(NC)"

keycloak-import: ## Import Keycloak realm
	@docker compose -f infra/docker/compose.yaml exec -T keycloak \
		/opt/keycloak/bin/kc.sh import --file /tmp/aurelius-realm.json

dev-frontend: ## Start frontend in dev mode
	@cd apps/frontend && pnpm dev

dev-gateway: ## Start gateway in dev mode
	@cd apps/gateway && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-imaging: ## Start imaging service in dev mode
	@cd apps/imaging-svc && uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

dev-ml: ## Start ML service in dev mode
	@cd apps/ml-svc && uvicorn app.main:app --reload --host 0.0.0.0 --port 8002

dev-search: ## Start search service in dev mode
	@cd apps/search-svc && uvicorn app.main:app --reload --host 0.0.0.0 --port 8004

reindex: ## Reindex all studies to OpenSearch
	@cd apps/search-svc && python app/services/reindex.py --full

reindex-incremental: ## Incremental reindex (only new/updated)
	@cd apps/search-svc && python app/services/reindex.py

shell-opensearch: ## Query OpenSearch
	@curl -X GET "http://localhost:9200/_cat/indices?v"

shell-postgres: ## Open Postgres shell
	@docker compose -f infra/docker/compose.yaml exec postgres psql -U postgres -d aurelius

shell-redis: ## Open Redis shell
	@docker compose -f infra/docker/compose.yaml exec redis redis-cli

health: ## Check health of all services
	@echo "$(BLUE)🏥 Checking service health...$(NC)"
	@curl -f http://localhost:8000/health || echo "$(RED)Gateway unhealthy$(NC)"
	@curl -f http://localhost:8001/health || echo "$(RED)Imaging service unhealthy$(NC)"
	@curl -f http://localhost:8002/health || echo "$(RED)ML service unhealthy$(NC)"
	@curl -f http://localhost:8042/system || echo "$(RED)Orthanc unhealthy$(NC)"
	@echo "$(GREEN)✅ Health check complete$(NC)"

ps: ## Show running services
	@docker compose -f infra/docker/compose.yaml ps

stats: ## Show service resource usage
	@docker stats --no-stream

backup: ## Backup database
	@echo "$(BLUE)💾 Backing up database...$(NC)"
	@docker compose -f infra/docker/compose.yaml exec -T postgres pg_dump -U postgres aurelius > backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Backup complete$(NC)"

restore: ## Restore database from backup
	@read -p "Backup file: " file; \
	docker compose -f infra/docker/compose.yaml exec -T postgres psql -U postgres aurelius < $$file

docs: ## Generate API documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@cd apps/gateway && python -m app.generate_docs
	@echo "$(GREEN)✅ Documentation generated$(NC)"

security-scan: ## Run security scans
	@echo "$(BLUE)🔒 Running security scans...$(NC)"
	@docker run --rm -v $(PWD):/code aquasec/trivy fs /code
	@cd apps/gateway && bandit -r app/
	@cd apps/frontend && pnpm audit
	@echo "$(GREEN)✅ Security scan complete$(NC)"

perf-test: ## Run performance tests
	@echo "$(BLUE)⚡ Running performance tests...$(NC)"
	@k6 run scripts/load-test.js
	@echo "$(GREEN)✅ Performance test complete$(NC)"

.DEFAULT_GOAL := help
