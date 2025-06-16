#install jq
apt -y install jq

#say hello to Azure Open GPT LLM
response=$(curl -s -X POST "$openai_azure_base_url/openai/deployments/$openai_azure_deployment_name/chat/completions?api-version=$openai_azure_version" \
  -H "Content-Type: application/json" \
  -H "api-key: $openai_api_key" \
  --data '{
    "messages": [{"role": "user", "content": "Dis-moi bonjour et identifie toi"}],
    "temperature": 0.3
  }')

# Vérifie si on a une réponse valide avec 'content'
if echo "$response" | grep -q '"content"'; then
  echo "✅ Succès : réponse reçue du llm"
  echo "$response" | jq -r '.choices[0].message.content'
else
  echo "$response"
  echo "❌ Erreur : réponse invalide"
  exit 1
fi
