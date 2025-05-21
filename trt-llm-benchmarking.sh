LOGS_FILE="trt-llm-serve-logs.txt"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
GEN_AI_PERF_FILE="trt-llm-benchmarking.json"
# #install tensorrt-llm 
pip install tensorrt_llm==0.19.0
#create config file
#use these configs for 7B model
file="extra_llm_api_configs.yml"
cat <<EOL > $file
enable_chunked_prefill: true
pytorch_backend_config:
  enable_overlap_scheduler: true 
  use_cuda_graph: true
  cuda_graph_max_batch_size: 512
  cuda_graph_padding_enabled: true
EOL


#start trt-llm-serve server
trtllm-serve $MODEL \
    --backend pytorch \
    --tp_size 8 \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --trust_remote_code \
    --extra_llm_api_options $file > $LOGS_FILE 2>&1&

#genai-perf benchmarking
pip install genai-perf==0.0.12

#Wait for server to be ready
while true; do
  RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
      "messages":[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Where is New York?"}
      ],
      "max_tokens": 16,
      "temperature": 0
    }')

  if [[ -n "$RESPONSE" ]]; then
    echo "Server is ready"
    break
  fi

  echo "Waiting for Server to start..."
  sleep 2
done

genai-perf profile -m $MODEL \
    --tokenizer $MODEL \
    --service-kind openai \
    --endpoint-type chat \
    --random-seed 123 \
    --num-prompts 640 \
    --num-prefix-prompts 1 \
    --concurrency 128 \
    --request-count 640 \
    --prefix-prompt-length 600 \
    --synthetic-input-tokens-mean 10 \
    --synthetic-input-tokens-stddev 0  \
    --output-tokens-stddev 0 \
    --output-tokens-mean 1300 \
    --verbose \
    --warmup-request-count 10 \
    --profile-export-file $GEN_AI_PERF_FILE \
    --url localhost:8000

genai-perf profile -m $MODEL \
    --tokenizer $MODEL \
    --service-kind openai \
    --endpoint-type chat \
    --random-seed 223 \
    --num-prompts 1280 \
    --num-prefix-prompts 1 \
    --concurrency 256 \
    --request-count 1280 \
    --prefix-prompt-length 600 \
    --synthetic-input-tokens-mean 10 \
    --synthetic-input-tokens-stddev 0  \
    --output-tokens-stddev 0 \
    --output-tokens-mean 1300 \
    --verbose \
    --warmup-request-count 10 \
    --profile-export-file $GEN_AI_PERF_FILE \
    --url localhost:8000