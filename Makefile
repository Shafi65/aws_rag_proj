# Shortcuts. Every recipe uses venv/bin/python directly, so none of them need
# the venv activated first -- `make web` works from a fresh terminal.
#
#   make up      start Postgres, wait until it is actually accepting connections
#   make web     start the browser UI            (http://127.0.0.1:8765)
#   make ask q="your question"
#   make eval    retrieval metrics + refusal rate
#   make ingest  re-ingest from S3 (skips unchanged files)
#   make check   verify Bedrock connectivity
#   make psql    open a SQL prompt against the corpus
#   make down    stop the container (data survives)

PY := ./venv/bin/python
DOCKER := $(shell command -v docker 2>/dev/null || echo $$HOME/.docker/bin/docker)

.PHONY: up down web ask eval ingest check psql stats demo

up:
	$(DOCKER) compose up -d
	@printf "waiting for postgres"
	@for i in $$(seq 1 30); do \
	  if [ "$$($(DOCKER) inspect -f '{{.State.Health.Status}}' rag_pg 2>/dev/null)" = "healthy" ]; then \
	    echo " ready"; exit 0; fi; printf "."; sleep 1; done; \
	  echo " TIMED OUT"; exit 1

down:
	$(DOCKER) compose down

# Everything below assumes `make up` has run.
web: ; $(PY) src/web.py
check: ; $(PY) src/check_bedrock.py
ingest: ; $(PY) src/ingest.py
eval: ; $(PY) eval/run_eval.py --verbose --refusal
psql: ; $(DOCKER) compose exec db psql -U rag -d ragdb

ask:
	@test -n "$(q)" || { echo 'usage: make ask q="your question"'; exit 1; }
	@$(PY) src/answer.py "$(q)"

stats:
	@$(DOCKER) compose exec -T db psql -U rag -d ragdb -c \
	"SELECT d.filename, count(*) AS chunks, max(c.page_end) AS pages \
	 FROM chunks c JOIN documents d ON d.id=c.document_id GROUP BY 1 ORDER BY 1;"

# One command before the demo: container up, AWS warm, UI running.
demo: up check web
