ENV ?= meshgen
RUN = micromamba run -n $(ENV)

.PHONY: test install frontend-build frontend-test e2e e2e-setup serve

install:
	$(RUN) python -m pip install -e . --no-deps

test:
	$(RUN) python -m pytest

frontend-build:
	npm --prefix frontend run build

frontend-test:
	npm --prefix frontend run test

e2e-setup:
	cd frontend && npx playwright install chromium

# Playwright smoke (run e2e-setup once first; builds the frontend so the
# server ships the current bundle). FMESH2_E2E_MESH / FMESH2_E2E_PORT override.
e2e: frontend-build
	npm --prefix frontend run e2e

serve:
	$(RUN) fmesh2 --no-browser
