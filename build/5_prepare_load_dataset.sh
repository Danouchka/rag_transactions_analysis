#!/bin/sh

input="dataset/aiml_dataset_enriched.log"
output="bulk.log"
counter=0
max_rows=512
create='{"create": {}}'
bulk_data=""

echo "Reading logs from $input..."

while read -r log_event
do
  counter=$((counter + 1))
  bulk_data="$bulk_data$create"'\n'"{\"message\":\"$log_event\""}'\n'

  if [ $counter -eq $max_rows ]; then
    echo "Indexing $counter documents..."
    echo "$bulk_data" > temp.json
    curl -s -u "$es_admin:$es_password" \
         -X POST "$es_endpoint/$es_indice/_bulk" \
         -H 'Content-Type: application/json' \
         --insecure \
         --data-binary @temp.json > "$output"
    rm -f temp.json
    counter=0
    bulk_data=""
  fi
done < "$input"

# Envoi des lignes restantes
if [ $counter -lt $max_rows ] && [ $counter -gt 0 ]; then
  echo "Indexing remaining $counter documents..."
  echo "$bulk_data" > temp.json
  echo "$bulk_data"
  curl -s -u "$es_admin:$es_password" \
       -X POST "$es_endpoint/$es_indice/_bulk" \
       -H 'Content-Type: application/json' \
       --insecure \
       --data-binary @temp.json >> "$output"
  rm -f temp.json
fi
