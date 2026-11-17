.PHONY: install model calls audio transcripts grade grade-clean report serve test lint

install:
	uv sync

model: ## the speech model, about 150 MB, into models/
	mkdir -p models
	curl -L --fail -o models/ggml-base.en.bin \
		https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

SET ?= heldout

calls: ## free: rebuild both sets of 160 synthetic calls and their true labels
	uv run callqa --set dev generate
	uv run callqa --set heldout generate

audio: ## free, macOS only: speak every call into a two-channel recording (SET=dev for the other set)
	uv run callqa --set $(SET) synthesize

transcripts: ## free, needs whisper-cli and the model: transcribe every recording
	uv run callqa --set $(SET) transcribe

grade: ## paid, about 10 cents: mask and grade the speech-recognition transcripts
	uv run callqa --set $(SET) grade --source asr

grade-clean: ## paid: grade the original text, to see what speech recognition costs
	uv run callqa --set $(SET) grade --source clean

report: ## free: rebuild results/report.md from the saved files
	uv run callqa evaluate --write

serve: ## the dashboard on http://localhost:8020
	uv run callqa --set $(SET) serve

test: ## no audio, network or API key needed
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
