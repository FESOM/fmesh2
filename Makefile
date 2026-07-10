ENV ?= meshgen
RUN = micromamba run -n $(ENV)

.PHONY: test install frontend-build frontend-test e2e-setup serve

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

serve:
	$(RUN) fmesh2 --no-browser
