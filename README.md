# Benchmarking with `trtllm-serve` and `genai-perf`

This document outlines the steps to benchmark the [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) model using `trtllm-serve` with the PyTorch backend, and `genai-perf` for performance evaluation.

---

## 1. Launch the NGC Triton Server Docker Container

Use the latest container version from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
```bash
docker run --rm -it --net host --shm-size=8g   --ulimit memlock=-1 --ulimit stack=67108864 --gpus all   -v $(pwd):/workspace -w /workspace   nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3 bash
```

---

## 2. Install TensorRT-LLM

Ensure you're using TensorRT-LLM version `>=0.19.0` for PyTorch backend support.

```bash
pip install tensorrt_llm==0.19.0
```

---

## 3. Start the OpenAI-Compatible API Server

Run `trtllm-serve` with the appropriate arguments:

```bash
trtllm-serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --backend pytorch \
    --tp_size 8 \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --trust_remote_code \
    --extra_llm_api_options extra_llm_api_configs.yml
```

> For more details on supported arguments, refer to the [official documentation](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html).

`extra_llm_api_options` supports extra configs supported for pytorch backend. You can find arguments supported for pytorch backend here: [quickstart_advanced.py](https://github.com/NVIDIA/TensorRT-LLM/blob/f94af0fb86d85641b6fd41f0ddc4cd58131c9b18/examples/pytorch/quickstart_advanced.py#L118) and [PyTorchconfigs](http://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/config.py#L22).

Example `extra_llm_api_configs.yml`:

```yaml
pytorch_backend_config:
  enable_overlap_scheduler: true 
  use_cuda_graph: true
  cuda_graph_max_batch_size: 1024
  cuda_graph_padding_enabled: true
```

The server will start on `localhost:8000`. Test it with:

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages":[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Where is New York?"}
    ],
    "max_tokens": 16,
    "temperature": 0
  }'
```

---

## 4. Benchmark Using `genai-perf`

We use `genai-perf` to generate synthetic prompts and evaluate LLM performance.

### Experiment Settings:

- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
- **Input System Length (ISL)**: ~600 tokens (system prompt) + ~10 tokens (user prompt)
- **Output Sequence Length (OSL)**: ~1300 tokens

### Install `genai-perf`:

```bash
pip install genai-perf==0.0.12
```

### Run Benchmark:

```bash
genai-perf profile -m deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --service-kind openai \
  --endpoint-type chat \
  --random-seed 223 \
  --num-prompts 1280 \
  --num-prefix-prompts 1 \
  --concurrency 256 \
  --request-count 1280 \
  --prefix-prompt-length 600 \
  --synthetic-input-tokens-mean 10 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 1300 \
  --output-tokens-stddev 0 \
  --verbose \
  --warmup-request-count 10 \
  --profile-export-file $GEN_AI_PERF_FILE \
  --url localhost:8000
```

### Flag Descriptions

| Flag | Description |
|------|-------------|
| `--num-prompts` | The number of unique payloads to sample from. These will be reused until benchmarking is complete. |
| `--warmup-request-count` | The number of warmup requests to send before benchmarking |
| `--random-seed` | Random seed to generate prompts. Different random seeds would generate different sets of prompts. |
| `--request-count` | Total number of requests benchmarked. |
| `--concurrency` | Number of concurrent requests during benchmarking. |
| `--num-prefix-prompts` |The number of prefix prompts to select from. If this value is not zero, these are prompts that are prepended to input prompts. It sets the number of system prompts to use |
| `--prefix-prompt-length` | The number of tokens to include in each prefix prompt. This value is only used if --num-prefix-prompts is positive. This flag is useful to set the length of system prompts. |
| `--synthetic-input-tokens-mean` | Average token count for user prompts. |
| `--output-tokens-mean` | Average number of tokens generated in output. |
| `--tokenizer` | Tokenizer used for prompt encoding. |
| `--url` | Endpoint URL of the running inference server. |


Refer to the `input.json` file in artifacts directory(created after running the GAP command) to inspect the synthetic dataset used. If you want to conduct benchmarking for unique prompts, make sure to set `--num-prompts` >=  `--request-count`. For advanced configurations and flag details, see the [GenAI-Perf documentation](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf#model-inputs).
Refer to [trt-llm-benchmarking.sh](.trt-llm-benchmarking.sh) for a combined shell script automating all of the above steps above. 