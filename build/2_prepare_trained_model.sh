#!/bin/sh

#deploiement .multilingual-e5-small-elasticsearch
response=$(curl -s \
  -u "$es_admin:$es_password" \
  -X POST "$es_endpoint/_inference/text_embedding/.multilingual-e5-small-elasticsearch" \
  -H "Content-Type: application/json" \
  --insecure \
  --data-binary '{"input":"genere moi un vecteur"}')

if echo "$response" | grep -q "embedding"; then
  echo "✅ Succès : vecteur correctement généré et modèle déployé"
else
  echo "$response"
  echo "❌ Erreur : réponse inattendue."
  exit 1
fi

#curl -u "$es_admin:$es_password"   -XPOST  "$es_endpoint/_inference/text_embedding/.multilingual-e5-small-elasticsearch" -H 'Content-Type: application/json' --insecure --data-binary '{"input":"genere moi un vecteur"}'

#deploiement reranker 
response=$(curl -s \
  -u "$es_admin:$es_password" \
  -X POST "$es_endpoint/_inference/rerank/.rerank-v1-elasticsearch" \
  -H "Content-Type: application/json" \
  --insecure \
  --data-binary '{"input":"['sand','cold','frozen','sun']","query":"sahara"}')

if echo "$response" | grep -q "rerank"; then
  echo "✅ Succès : reranking correctement effectué et modèle déployé"
else
  echo "$response"
  echo "❌ Erreur : réponse inattendue."
  exit 1
fi

