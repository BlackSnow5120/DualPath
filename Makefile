.PHONY: install run build clean test help

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GORUN=$(GOCMD) run
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# App name
APP_NAME=dualpath
BINARY_UNIX=$(APP_NAME)_unix

help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(GOMOD) tidy
	$(GOMOD) download

run: ## Run the application
	$(GORUN) main.go

build: ## Build the application
	$(GOBUILD) -o $(BINARY_NAME) -v .

clean: ## Clean build files
	rm -f $(BINARY_NAME)
	go clean

test: ## Run tests
	$(GOTEST) -v ./...

docker-up: ## Start Docker services (PostgreSQL, Redis, Milvus)
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

migrate: ## Run database migrations
	$(GORUN) main.go migrate

deps: ## Download all dependencies
	$(GOGET) ./...

start: install run ## Install and run the application

# Docker targets
docker-build:
	docker build -t $(APP_NAME) .

docker-run:
	docker run --rm -p 8080:8080 --env-file .env $(APP_NAME)
