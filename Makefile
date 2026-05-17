.PHONY: live webots replay

live:
	uv run python -m src.main --source live

webots:
	uv run python -m src.main --source webots --autostart

replay:
	@if [ -z "$(REC)" ]; then echo "usage: make replay REC=data/recordings/<id> [SPEED=1.0]"; exit 2; fi
	uv run python -m src.main --source replay --recording $(REC) --speed $(or $(SPEED),1.0)
