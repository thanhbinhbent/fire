# LLM Server (Ollama)

A lightweight LLM server using Ollama, providing an OpenAI-compatible API.

## Requirements

- Docker & Docker Compose (or Podman)
- NVIDIA GPU + CUDA (optional, for GPU acceleration)

## Quick Start

```bash
# Start Ollama server
docker compose up -d

# Pull a model (first time only)
curl http://localhost:8001/api/pull -d '{"name": "gemma3:1b"}'

# Or pull a larger model
curl http://localhost:8001/api/pull -d '{"name": "gemma3:4b"}'
```

## API Endpoints

Ollama provides an OpenAI-compatible API at `http://localhost:8001/v1/`.

### Health Check
```bash
curl http://localhost:8001/api/tags
```

### List Models
```bash
curl http://localhost:8001/v1/models
```

### Chat Completion (Streaming)
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:1b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": true
  }'
```

### Chat Completion (Non-Streaming)
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:1b",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100,
    "stream": false
  }'
```

### Pull a Model
```bash
curl http://localhost:8001/api/pull -d '{"name": "gemma3:1b"}'
```

## GPU Support

### NVIDIA GPU

Edit `docker-compose.yml` and uncomment the GPU section:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then restart:
```bash
docker compose down
docker compose up -d
```

## Popular Models

| Model | Size | Command |
|-------|------|---------|
| gemma3:1b | ~1GB | `curl http://localhost:8001/api/pull -d '{"name": "gemma3:1b"}'` |
| gemma3:4b | ~3GB | `curl http://localhost:8001/api/pull -d '{"name": "gemma3:4b"}'` |
| llama3.2:1b | ~1GB | `curl http://localhost:8001/api/pull -d '{"name": "llama3.2:1b"}'` |
| llama3.2:3b | ~2GB | `curl http://localhost:8001/api/pull -d '{"name": "llama3.2:3b"}'` |
| mistral:7b | ~4GB | `curl http://localhost:8001/api/pull -d '{"name": "mistral:7b"}'` |
| qwen2.5:1.5b | ~1GB | `curl http://localhost:8001/api/pull -d '{"name": "qwen2.5:1.5b"}'` |

Browse all models: https://ollama.com/library

## Configuration

Environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `0.0.0.0` | Host to bind to |
| `OLLAMA_NUM_PARALLEL` | `1` | Max parallel requests |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Max models in memory |

## Logs

```bash
docker compose logs -f
```

## Stop Server

```bash
docker compose down
```

## Cleanup

### Remove containers and models
```bash
# Stop and remove containers + volumes (deletes all models)
docker compose down -v

# Or keep volumes (preserve models)
docker compose down
```

### Remove images
```bash
# Remove Ollama image
docker rmi ollama/ollama:latest

# Remove all unused images
docker image prune -a
```

### Complete cleanup
```bash
# Remove everything (containers, volumes, images)
docker compose down -v
docker rmi ollama/ollama:latest curlimages/curl:latest
```

### Check disk usage
```bash
# See what's using space
docker system df

# For podman users
podman system df
```
