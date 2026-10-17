COMPOSE = docker compose
RUN = $(COMPOSE) run --rm app

.PHONY: build up all ingest eda train evaluate predict monitor test psql down clean

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d db

all: build up
	$(RUN) all

ingest:
	$(RUN) ingest

eda:
	$(RUN) eda

train:
	$(RUN) train

evaluate:
	$(RUN) evaluate

predict:
	$(RUN) predict

monitor:
	$(RUN) monitor

test:
	$(COMPOSE) run --rm --no-deps --entrypoint pytest app -q tests

psql:
	$(COMPOSE) exec db psql -U leads -d leads

down:
	$(COMPOSE) down

# also drops the database volume
clean:
	$(COMPOSE) down -v
	rm -rf artifacts/models
