.PHONY: live webots replay no-fly race measure

live:
	uv run python -m src.main --source live

webots:
	uv run python -m src.main --source webots

replay:
	@if [ -z "$(REC)" ]; then echo "usage: make replay REC=data/recordings/<id> [SPEED=1.0]"; exit 2; fi
	uv run python -m src.main --source replay --recording $(REC) --speed $(or $(SPEED),1.0)

no-fly:
	uv run python -m src.main --source live --no-fly

race:
	@if [ -z "$(GATES)" ]; then echo "usage: make race GATES=data/gates/<file>.csv"; exit 2; fi
	uv run python -m src.main --source live --race-only --true-gates $(GATES)

race-webots:
	uv run python -m src.main --source webots --race-only

measure:
	uv run python -m scripts.measure_gates