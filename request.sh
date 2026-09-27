curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "deepseek-ai/DeepSeek-V4.1-Flash",
      "messages": [
        {"role": "user", "content": "What is 17 * 19?"}
      ],
      "max_tokens": 128,
      "chat_template_kwargs": {"thinking": false}
    }'