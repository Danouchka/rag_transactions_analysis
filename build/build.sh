#!/bin/sh
set -e  # Arrêt immédiat en cas d'erreur
trap 'echo "💥 Script interrompu"; exit 1' INT TERM

########################################
## Export des variables d'environnements
########################################
echo "=============================================="
echo "Export des variables d'environnement"
echo "=============================================="
. ./1_prepare_env_variables.sh
echo " * ES Cloud ID: $es_cloud_id"
echo " * ES user: $es_admin"
echo " * ES password: $es_password"
echo " * ES Index utilisé : $es_indice"

echo " * Azure OpenAI api key: $openai_api_key"
echo " * Azure OpenAI base url: $openai_azure_base_url"
echo " * Azure OpenAI version: $openai_azure_version"
echo " * Azure OpenAI deployment name:  $openai_azure_deployment_name"
echo " * Azure OpenAI model name: $openai_model_name"

########################################
## preparation des trained models 
########################################
echo "=============================================="
echo "Préparation des trained models"
echo "=============================================="
. ./2_prepare_trained_model.sh
########################################
## vérification LLM Azure OpenAI 
########################################
echo "=============================================="
echo "Vérification LLM Azure OpenAI"
echo "=============================================="
. ./3_prepare_test_azure_openai_connection.sh
########################################
## création datastream
########################################
echo "=============================================="
echo "Création Datastream"
echo "=============================================="
. ./4_prepare_create_datastream.sh
########################################
## Installation env python
########################################
echo "=============================================="
echo "Install python packages"
echo "=============================================="
apt install -y python3-pip python3-venv
cd ../run/
python3 -m venv .venv
. .venv/bin/activate
# Mettre à jour pip, setuptools et wheel
pip install --upgrade pip setuptools wheel
# Purger le cache pour éviter les conflits de versions
pip cache purge
pip install -r ../build/requirements_clean.txt
########################################
## Load dataset
########################################
echo "=============================================="
echo "Load Dataset"
echo "=============================================="
deactivate
cd ../build/
sh 5_prepare_load_dataset.sh


