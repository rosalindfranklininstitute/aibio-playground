#!/bin/bash
set -e

# Start Ollama serve (bf)
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
sleep 1
until ollama list > /dev/null 2>&1; do
    sleep 1
done

# Pull target model(s)
ollama pull gemma4:26b

wait $OLLAMA_PID
