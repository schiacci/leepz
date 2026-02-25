# Ollama Integration Guide

## Overview

The LEAP Strategic Asset Engine now supports multiple Ollama models running simultaneously for local LLM inference. This enables offline analysis without relying on external APIs.

## Available Ollama Providers

### 1. Ollama (Default)
- **Provider ID**: `ollama`
- **Model**: `qwen3:latest` (configurable via `OLLAMA_MODEL` env var)
- **Priority**: 3 (local fallback)
- **Use Case**: General-purpose analysis

### 2. Ollama DeepSeek-R1
- **Provider ID**: `ollama-deepseek-r1`
- **Model**: `deepseek-r1:latest`
- **Priority**: 4 (alternative local model)
- **Use Case**: Advanced reasoning and analysis

### 3. Ollama DeepSeek-R1 32B
- **Provider ID**: `ollama-deepseek-r1-32b`
- **Model**: `deepseek-r1:32b`
- **Priority**: 5 (smaller local model)
- **Use Case**: Faster inference with smaller model

## Configuration

### Environment Variables

```bash
# Ollama server configuration
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen3:latest"
export OLLAMA_TIMEOUT="120"  # Timeout in seconds
```

### Configuration File (config.py)

```python
class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:latest"
    timeout: int = 120  # Increased to 2 minutes for large models
```

## Usage Examples

### Single Ollama Model
```bash
python main.py --symbols AAPL --llm-enhanced on --llm-providers ollama
```

### Multiple Ollama Models (Simultaneous)
```bash
python main.py --symbols AAPL --llm-enhanced on --llm-providers ollama,ollama-deepseek-r1-32b
```

### Mixed Providers (Ollama + Cloud)
```bash
python main.py --symbols AAPL --llm-enhanced on --llm-providers openrouter,ollama-deepseek-r1
```

### All Available Providers
```bash
python main.py --symbols AAPL --llm-enhanced on --llm-providers openrouter,gemini,ollama,ollama-deepseek-r1,ollama-deepseek-r1-32b
```

## Model Selection Strategy

The system uses a priority-based fallback mechanism:

1. **Priority 1**: OpenRouter (cloud, highest quality)
2. **Priority 2**: Gemini (cloud, free tier)
3. **Priority 3**: Ollama qwen3 (local, default)
4. **Priority 4**: Ollama DeepSeek-R1 (local, advanced reasoning)
5. **Priority 5**: Ollama DeepSeek-R1 32B (local, faster)

## Performance Considerations

### Model Size vs Speed
- **qwen3:latest** (8.2B parameters): Fastest, good for quick analysis
- **deepseek-r1:32b** (32.8B parameters): Balanced speed/quality
- **deepseek-r1:latest** (671B parameters): Slowest but most thorough

### Resource Requirements
- **RAM**: Minimum 8GB for 32B models, 32GB+ for 671B models
- **Storage**: 5GB per model approximately
- **CPU/GPU**: GPU acceleration recommended for large models

## Troubleshooting

### Common Issues

1. **Timeout Errors**
   ```bash
   # Increase timeout
   export OLLAMA_TIMEOUT="300"
   ```

2. **Model Not Found**
   ```bash
   # Check available models
   curl http://localhost:11434/api/tags
   
   # Pull missing model
   ollama pull deepseek-r1:32b
   ```

3. **Connection Refused**
   ```bash
   # Start Ollama server
   ollama serve
   ```

### Debug Mode

Enable verbose output to see model selection and fallback behavior:

```bash
python main.py --symbols AAPL --llm-enhanced on --llm-providers ollama --verbose
```

## Adding New Ollama Models

To add a new Ollama model to the system:

1. **Pull the model**:
   ```bash
   ollama pull llama3:70b
   ```

2. **Add to PROVIDER_REGISTRY** in `agents/ai_agents.py`:
   ```python
   "ollama-llama3-70b": {
       "name": "Ollama Llama3 70B",
       "priority": 6,
       "client_class": lambda: OllamaClient("llama3:70b"),
       "model": lambda: "llama3:70b",
       "requires_key": lambda: True,
       "free_tier": True,
       "supports_streaming": False,
   },
   ```

3. **Use the new provider**:
   ```bash
   python main.py --symbols AAPL --llm-enhanced on --llm-providers ollama-llama3-70b
   ```

## Best Practices

1. **Start Small**: Begin with `ollama` (qwen3) for testing
2. **Scale Up**: Use `ollama-deepseek-r1-32b` for production
3. **Fallback Strategy**: Combine local and cloud providers for reliability
4. **Monitor Resources**: Watch RAM/CPU usage with large models
5. **Batch Processing**: Use multiple models for consensus analysis

## Example Output

```
🤖 Initialized LLM client: OLLAMA
🤖 Initialized LLM client: OLLAMA-DEEPSEEK-R1-32B
🤖 Applying Multi-LLM Enhancement with: OLLAMA, OLLAMA-DEEPSEEK-R1-32B
🧠 Sentiment Model: x-ai/grok-beta
🤖 Multi-LLM Enhancement for AAPL
🔄 LLM 1/2: OLLAMA
✅ OLLAMA insights: sentiment 0.75, confidence 0.85
🔄 LLM 2/2: OLLAMA-DEEPSEEK-R1-32B
✅ OLLAMA-DEEPSEEK-R1-32B insights: sentiment 0.80, confidence 0.90
📊 Consulted 2 LLMs (max: 3, threshold: 0.85)
🤖 Multi-LLM Consensus: 2 providers
📊 Consensus Sentiment: 0.78
🎯 Enhanced Score: 0.856 (Original: 0.819)
```

This demonstrates successful multi-model consensus using local Ollama inference.
