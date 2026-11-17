.PHONY: install model calls audio transcripts grade grade-clean report serve test lint

install:
	uv sync

model: ## the speech model, about 150 MB, into models/
	mkdir -p models
	curl -L --fail -o models/ggml-base.en.bin \
		https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

calls: ## free: rebuild the 160 synthetic calls and their true labels
	uv run callqa generate

audio: ## free, macOS only: speak every call into a two-channel recording
	uv run callqa synthesize

transcripts: ## free, needs whisper-cli and the model: transcribe and mask every recording
	uv run callqa transcribe

grade: ## paid: grade the masked speech-recognition transcripts
	uv run callqa grade --source asr

grade-clean: ## paid: grade the original text, to see what speech recognition costs
	uv run callqa grade --source clean

report: ## free: rebuild results/report.md from the saved files
	uv run callqa evaluate --write

serve: ## the dashboard on http://localhost:8020
	uv run callqa serve

test: ## no audio, network or API key needed
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
