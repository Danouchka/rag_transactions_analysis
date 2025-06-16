#!/bin/sh
if [ ! -f ../config/.env ]; then
  echo "❌ Fichier ../config/.env introuvable"
  exit 1
fi

. ../config/.env
echo "✅ Succès: variables exportées"
