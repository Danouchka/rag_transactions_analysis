#creation mapping

response=$(curl -s -u "$es_admin:$es_password" -XDELETE "$es_endpoint/_component_template/transaction.service@mappings")

response=$(curl -s \
  -u "$es_admin:$es_password" \
  -X POST "$es_endpoint/_component_template/transaction.service@mappings" \
  -H "Content-Type: application/json" \
  --insecure \
  --data-binary @mappings.json )

if echo "$response" | grep -q "true"; then
  echo "✅ Succès : mapping component template correctement inséré"
else
  echo "$response"
  echo "❌ Erreur : réponse inattendue."
  exit 1
fi

#creation ingest pipeline
response=$(curl -s -u "$es_admin:$es_password" -XDELETE "$es_endpoint/_ingest/pipeline/pipeline_transaction_service")

response=$(curl -s \
  -u "$es_admin:$es_password" \
  -X PUT "$es_endpoint/_ingest/pipeline/pipeline_transaction_service" \
  -H "Content-Type: application/json" \
  --insecure \
  --data-binary @ingest_pipeline.json )

if echo "$response" | grep -q "true"; then
  echo "✅ Succès : ingest pipeline correctement inséré"
else
  echo "$response"
  echo "❌ Erreur : réponse inattendue."
  exit 1
fi


#creation ilm policy 
response=$(curl -s -u "$es_admin:$es_password" -XDELETE "$es_endpoint/_ilm/policy/policy_ilm_transaction_service")

response=$(curl -s \
  -u "$es_admin:$es_password" \
  -X PUT "$es_endpoint/_ilm/policy/policy_ilm_transaction_service" \
  -H "Content-Type: application/json" \
  --insecure \
  --data-binary @ilm_policy.json )

if echo "$response" | grep -q "true"; then
  echo "✅ Succès : ilm policy correctement insérée"
else
  echo "$response"
  echo "❌ Erreur : réponse inattendue."
  exit 1
fi

#création index template 
response=$(curl -s -u "$es_admin:$es_password" -XDELETE "$es_endpoint/_index_template/logs-transaction.service-template")

response=$(curl -s \
  -u "$es_admin:$es_password" \
  -X PUT "$es_endpoint/_index_template/logs-transaction.service-template" \
  -H "Content-Type: application/json" \
  --insecure \
  --data-binary @index_template.json )

if echo "$response" | grep -q "true"; then
  echo "✅ Succès : index template correctement insérée"
else
  echo "$response"
  echo "❌ Erreur : réponse inattendue."
  exit 1
fi

