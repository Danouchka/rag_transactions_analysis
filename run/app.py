import re
from datetime import datetime,timedelta
from collections import Counter,defaultdict
import traceback
import openai
import os
import hmac
import streamlit as st
import logging
import json
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

from PIL import Image
from typing import Any, Dict, Iterable

from elasticsearch import Elasticsearch
from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_elasticsearch import ElasticsearchStore,ElasticsearchRetriever
from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import StructuredTool  # Import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from typing import Optional
from pydantic import BaseModel, Field

#Phase de login sur l'interface 
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            #del st.session_state["password"]  # Don't store the username or password.
            #del st.session_state["username"]
            st.session_state["logged_username"]=st.session_state["username"]
            st.session_state["logged_password"]=st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 UUilisateur/Password incorrect")
    return False

if not check_password():
    st.stop()


#Configuration de la page d'accueil
imc = Image.open("favicon.png")

st.set_page_config(
    page_title="Analyses Financières",
    layout="wide",
    page_icon=imc,
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'This App shows you how to use Elasticsearch with LLM'
    }
)

col1, col2, col3 = st.columns(3)
with col1:
     st.write("");
     st.write("");
     st.write("");
with col2:     
     st.image('https://upload.wikimedia.org/wikipedia/fr/9/97/Elastic_NV_logo.svg',width=200)
with col3:
     st.write("");
     st.write("");
     st.write("");
     st.image('https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1DBe4cCQYfpM0oNXl_kH2w.png',width=200)

st.header("Interface de Recherche Sémantique")
st.write("[lien](https://www.elastic.co/)")

#Read config from environment variables first
openai.model_name  = os.getenv('openai_model_name')
openai.api_key     = os.getenv('openai_api_key')
openai.api_base    = os.getenv('openai_azure_base_url') # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type    = 'azure'
openai.api_version =os.getenv('openai_azure_version') # this may change in the future
deployment_name    =os.getenv('openai_azure_deployment_name')
es_cloud_id        =os.getenv('es_cloud_id')
es_password        =os.getenv('es_password')
es_admin	       =os.getenv('es_admin') 
INDEX_NAME         =os.getenv('es_indice') 

#connect to elasticsearch 
basic_auth=(es_admin, es_password) # Alternatively use `api_key` instead of `basic_auth`

if st.session_state["logged_username"]!="admin":
   basic_auth=(st.session_state["logged_username"],st.session_state["logged_password"])

es = Elasticsearch(
  cloud_id = es_cloud_id,
  basic_auth=basic_auth,
  request_timeout=60
)

#Affichage de la personne loggée
def ui_display_logged_user():
    logged_user=st.session_state['logged_username']
    st.success(f"Logged as {logged_user}",icon="✅")

#Définition d'une mémoire qu'on peut mettre à 0 
if "my_memory" not in st.session_state:
   st.session_state['my_memory'] = ConversationBufferWindowMemory(k=10, memory_key="chat_history", output_key='output', return_messages=True)
   st.session_state['my_memory'].chat_memory.messages=[]

def reset_conversation():
  if "my_memory" in st.session_state and st.session_state['my_memory'].chat_memory.messages:
     st.session_state['my_memory'].chat_memory.messages=[]

#For streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.container:
        #self.container.markdown(self.text)
        #self.container.chat_message("ai",avatar='👨<200d>⚕️').write(self.text)
           with self.container.chat_message("ai",avatar='👨'):
             st.markdown(self.text)


import re
from datetime import datetime

# Mapping des mois français vers anglais
mois_fr_to_en = {
    "janvier": "January",
    "février": "February",
    "mars": "March",
    "avril": "April",
    "mai": "May",
    "juin": "June",
    "juillet": "July",
    "août": "August",
    "septembre": "September",
    "octobre": "October",
    "novembre": "November",
    "décembre": "December"
}

def parse_date_fr(date_str):
    # Remplace le mois français par le mois anglais
    for fr, en in mois_fr_to_en.items():
        if fr in date_str:
            date_str = date_str.replace(fr, en)
            break
    # Pars la date en utilisant les mois anglais
    return datetime.strptime(date_str, "%d %B %Y")

def parse_markdown_transactions(text):
    lines = text.strip().split('\n')
    table_lines = []
    capture = False
    for line in lines:
        if line.strip().startswith('|'):
            capture = True
        if capture:
            table_lines.append(line)
        if line.strip().startswith('| **Cumul') or line.strip().startswith('| **Cumulative'):
            break

    if not table_lines:
        return []

    # Vérifie les en-têtes de la 1ère et dernière colonne
    headers_raw = [h.strip() for h in table_lines[0].split('|') if h.strip()]
    if not headers_raw:
        return []

    first_header = headers_raw[0].lower()
    last_header = headers_raw[-1].lower()

    if not ("date" in first_header) or not ("montant" in last_header):
        # En-têtes incorrectes pour parsing sûr
        return []

    num_cols = len(headers_raw)
    transactions = []

    for line in table_lines[2:]:
        if line.strip().startswith('|'):
            cols = [c.strip() for c in line.split('|') if c.strip()]
            if len(cols) == num_cols:
                # Colonnes : première = date/heure, dernière = montant
                date_heure = cols[0]
                montant_str = cols[-1]

                # Colonnes intermédiaires (dynamiques)
                source = cols[1] if len(cols) > 1 else ""
                destination = cols[2] if len(cols) > 2 else ""
                type_ = cols[3] if len(cols) > 3 else ""
                description = cols[4] if len(cols) > 4 else ""

                # Extraction date et heure
                date_heure_clean = re.sub(r"^\w+\s+", "", date_heure)  # Supprime le jour de la semaine
                match = re.search(r"(\d{1,2} \w+ \d{4}) à (\d{1,2}h\d{2})", date_heure_clean)
                if match:
                    date_str = match.group(1)
                    heure_str = match.group(2).replace('h', ':')
                    try:
                        date_obj = parse_date_fr(date_str)
                        date = date_obj.strftime("%Y-%m-%d")
                        heure = heure_str
                    except Exception:
                        date = ""
                        heure = ""
                else:
                    date = ""
                    heure = ""

                # Nettoyage du montant
                montant_str = montant_str.replace(",", ".")
                montant_str = re.sub(r"[^\d.]", "", montant_str)
                try:
                    montant = float(montant_str) if montant_str else 0.0
                except Exception:
                    montant = 0.0

                # Structure finale
                entry_structured = {
                    "date": date,
                    "heure": heure,
                    "source": source,
                    "destination": destination,
                    "type": type_,
                    "description": description,
                    "montant": montant
                }

                transactions.append(entry_structured)

    return transactions


def plot_somme_montants_par_jour(result_docs, container):
    """
    Génère un histogramme Altair des sommes des montants par jour
    et l'affiche dans un conteneur Streamlit.
    
    """
    somme_montants_par_jour = defaultdict(float)
    
    for doc in result_docs:
        # Extraire la date
        date_str = doc.get("date","")
        if not date_str:
            continue
        
        try:
            date_obj  = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue          
  
        # Extraire le montant
        montant = 0.0
        montant_str=doc.get("montant",0.0)
        try:
           montant = float(montant_str) if montant_str else 0.0
        except Exception:
           montant = 0.0

        somme_montants_par_jour[date_obj] += montant

    # Générer la liste de dates complète
    start_date = datetime(2023, 1, 1).date()
    end_date = datetime.today().date()
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # Récupérer les sommes des montants par jour
    somme_jour = [somme_montants_par_jour.get(date, 0) for date in date_range]

    # Créer un DataFrame pour Altair
    df_plot = pd.DataFrame({
        'Date': date_range,
        'Somme des montants (€)': somme_jour
    })

    # Créer le graphique Altair
    chart = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('Date:T', axis=alt.Axis(title='Date', labelAngle=-45)),
        y=alt.Y('Somme des montants (€):Q', axis=alt.Axis(title='Somme des montants (€)'))
    ).properties(
        #title="Somme des montants des transactions par jour (2023 - aujourd'hui)",
        width=600,
        height=200
    ).configure_axis(
        labelFontSize=8,
        titleFontSize=10
    )

    # Afficher le graphique dans Streamlit
    container.markdown("##### Somme des montants des transactions par jour (2023 - aujourd'hui)")
    container.altair_chart(chart, use_container_width=True)
    
#Instrumentation OTEL
#class my_llm_Handler(BaseCallbackHandler):
#    def on_llm_start(self, serialized, prompts, **kwargs):
#        #with tracer.start_as_current_span("on_llm_start") as span:
#            current_span = trace.get_current_span()
#            prompts_len = 0
#            prompts_len += sum([len(prompt) for prompt in prompts])
#            current_span.set_attribute("num_processed_prompts", len(prompts))
#            current_span.set_attribute("prompts_len", prompts_len)
#
#    def on_llm_end(self, response, **kwargs):
#        #with tracer.start_as_current_span("on_llm_end") as span:
#            current_span = trace.get_current_span()
#            # example output: {'completion_tokens': 14, 'prompt_tokens': 71, 'total_tokens': 85}
#            if response.llm_output is not None:
#               token_usage = response.llm_output["token_usage"]
#               for k, v in token_usage.items():
#                 current_span.set_attribute(k, v)
def safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

my_query_body=""
result_docs=[]

#definition de la recherche de transactions particulières 
def transactions_search(query: str, dates: Optional[str] = None, sources: Optional[str] = None, destinations: Optional[str] = None, type_of_transactions: Optional[str] = None):
    if es is None:
       return "ES client not initialized"
    else:
       try:
         #Build the elasticsearch query
         must_clauses_standard = []
         must_clauses_knn = []
         
         global my_query_body
         my_query_body=None

         global result_docs
 
         #Dates
         if dates:
            #Dates must be in format 'YYYY-MM-DD' or 'YYYY-MM-DD à YYYY-MM-DD'
            date_parts = dates.strip().split(' à ') 
            if len(date_parts) == 1:
               start_date = date_parts[0]
               end_date = date_parts[0]
            elif len(date_parts) == 2:
               start_date = date_parts[0]
               end_date = date_parts[1]
            else:
               return "Invalid date format Please use YYYY-MM-DD or YYYY-MM-DD à YYYY-MM-DD."
             
            date_range = {
               "range": {
                  "@timestamp": {
                    "gte": start_date,
                    "lte": end_date
                  }
               }
            } 
            
            must_clauses_standard.append(date_range)
            must_clauses_knn.append(date_range) 
        
         if sources:
            sources_list = [s.strip() for s in sources.split(",")]
            sources_accounts = {
               "terms":{
                 "source.account.id":sources_list  
               }
            } 
            must_clauses_standard.append(sources_accounts)
            must_clauses_knn.append(sources_accounts)

         if destinations:
            destinations_list = [d.strip() for d in destinations.split(",")]
            destinations_accounts = {
               "terms":{
                 "destination.account.id":destinations_list
               }
            }
            must_clauses_standard.append(destinations_accounts)
            must_clauses_knn.append(destinations_accounts)

         if type_of_transactions:
            transaction_types = {
               "terms":{
                 "event.action":[type_of_transactions]
               }
            }
            must_clauses_standard.append(transaction_types)
            must_clauses_knn.append(transaction_types) 

         #main query standard
         main_query_standard={
            "match": {
               "text": query
            }
         }
         must_clauses_standard.append(main_query_standard)

         #main query knn 
         main_query_knn={
            "knn": {
               "field": "text_embeddings_e5",
               "query_vector_builder":{
                  "text_embedding": {
                     "model_id": ".multilingual-e5-small-elasticsearch",
                     "model_text": query
                  }
               },
               "k":40,
               "num_candidates":200
            }
         }
         must_clauses_knn.append(main_query_knn)

         es_query={
             "_source": ["@timestamp","amount", "text", "source.account.id", "destination.account.id", "source.account.balance.old","source.account.balance.new","destination.account.balance.old","destination.account.balance.new","event.action"],
            "retriever":{
              "text_similarity_reranker": {
                "retriever": {
                  "rrf":{
                    "retrievers":[
                       { 
                        "standard":{
                          "query":{
                            "bool":{
                              "must":must_clauses_standard
                            }
                          }  
                        }
                       },
                       {
                        "standard":{
                          "query":{
                            "bool":{
                              "must":must_clauses_knn
                            }
                          }
                        } 
                       }
                    ],
                    "rank_window_size": 40,
                    "rank_constant": 1
                  }
                },
                "inference_id": ".rerank-v1-elasticsearch",
                "inference_text": query,
                "field": "text"
              }
            }
         }
         
         my_query_body=es_query
         #print(json.dumps(es_query, indent=2))
 
         response = es.search(index=INDEX_NAME, body=es_query)
         hits = response["hits"]["hits"]
         if not hits:
              return "No articles found for your query."
         result_docs = []
         for hit in hits:
              source = hit["_source"]
              event = safe_get(source,"event","action")
              text = source.get("text")
              date = source.get("@timestamp")
              amount=source.get("amount")
              source_account_id = safe_get(source, "source", "account", "id")
              source_balance_old = safe_get(source, "source", "account", "balance", "old")
              source_balance_new = safe_get(source, "source", "account", "balance", "new")
              destination_account_id = safe_get(source, "destination", "account", "id")
              destination_balance_old = safe_get(source, "destination", "account", "balance", "old")
              destination_balance_new = safe_get(source, "destination", "account", "balance", "new")
              doc = f"{date};{event};montant {amount};{text};source account id {source_account_id};destination account id {destination_account_id};source old balance {source_balance_old};source new balance {source_balance_new}; destination old balance {destination_balance_old}; destination new balance {destination_balance_new} \n"
              result_docs.append(doc)
         return "\n".join(result_docs)
       
       except Exception as e:
         return f"Error during RAG search: {e}"         
        
class transactions_search_input(BaseModel):
    query: str = Field(..., description="The search query for the knowledge base.")
    dates: Optional[str]  = Field(None, description="Date or date range for filtering results. Specify in format YYYY-MM-DD or YYYY-MM-DD à YYYY-MM-DD.")
    sources: Optional[str] = Field(None, description="1 to several sources account ids. Specify like \"source_account_id1,source_account_id2,source_account_id3, etc...\".Starts by 'C' or 'M' followed by 9 digits at least ")
    destinations: Optional[str] = Field(None, description="1 to several destination account  ids. Specify like \"destination_account_id1,destination_account_id2,destination_account_id3, etc..\".Starts by 'C' or 'M' followed by 9 digits at least")
    type_of_transactions: Optional[str] = Field(None, description="1 to several transaction types among 'PAYMENT','DEBIT','TRANSFER','CASH_IN','CASH_OUT'") 
    

transactions_search_tool = StructuredTool(
    name="transactions_search",
    func=transactions_search,
    description=(
        "Use this tool to search for list of transactions"
        "**Input must include a search query** "
        "Dates are optional buf if present  must be specified in this format YYYY-MM-DD or YYYY-MM-DD à YYYY-MM-DD."
        "Source and Destination account ids are optonal identifiers starting by 'C' or 'M' followed by at least 9 digits."
        "Transaction types are also optional. Precise the nature of the transaction"
    ),
    args_schema=transactions_search_input
)


#definition de la recherche de transactions particulières 
def transactions_search_all(dates: Optional[str] = None, sources: Optional[str] = None, destinations: Optional[str] = None, type_of_transactions: Optional[str] = None):
    if es is None:
       return "ES client not initialized"
    else:
       try:
         #Build the elasticsearch query
         must_clauses= []

         global my_query_body
         my_query_body=None

         global result_docs
         
         #Dates
         if dates:
            #Dates must be in format 'YYYY-MM-DD' or 'YYYY-MM-DD à YYYY-MM-DD'
            date_parts = dates.strip().split(' à ') 
            if len(date_parts) == 1:
               start_date = date_parts[0]
               end_date = date_parts[0]
            elif len(date_parts) == 2:
               start_date = date_parts[0]
               end_date = date_parts[1]
            else:
               return "Invalid date format Please use YYYY-MM-DD or YYYY-MM-DD à YYYY-MM-DD."
             
            date_range = {
               "range": {
                  "@timestamp": {
                    "gte": start_date,
                    "lte": end_date
                  }
               }
            } 
            
            must_clauses.append(date_range)
        
         if sources:
            sources_list = [s.strip() for s in sources.split(",")]
            sources_accounts = {
               "terms":{
                 "source.account.id":sources_list
               }
            } 
            must_clauses.append(sources_accounts)

         if destinations:
            destinations_list = [d.strip() for d in destinations.split(",")]
            destinations_accounts = {
               "terms":{
                 "destination.account.id":destinations_list
               }
            }
            must_clauses.append(destinations_accounts)

         if type_of_transactions:
            transaction_types = {
               "terms":{
                 "event.action":[type_of_transactions]
               }
            }
            must_clauses.append(transaction_types)
           

         es_query={
            "size":100,
            "_source": ["@timestamp","amount", "text", "source.account.id", "destination.account.id", "source.account.balance.old","source.account.balance.new","destination.account.balance.old","destination.account.balance.new","event.action"],
            "query":{
               "bool":{
                  "must":must_clauses
               }
            }  
         }     
             
         my_query_body=es_query
         #print(json.dumps(es_query, indent=2))

         response = es.search(index=INDEX_NAME, body=es_query)
         hits = response["hits"]["hits"]
         if not hits:
              return "No articles found for your query."
         result_docs = []
         for hit in hits:
              source = hit["_source"]
              event = safe_get(source,"event","action")
              text = source.get("text")
              date = source.get("@timestamp")
              amount=source.get("amount")
              source_account_id = safe_get(source, "source", "account", "id")
              source_balance_old = safe_get(source, "source", "account", "balance", "old")
              source_balance_new = safe_get(source, "source", "account", "balance", "new")
              destination_account_id = safe_get(source, "destination", "account", "id")
              destination_balance_old = safe_get(source, "destination", "account", "balance", "old")
              destination_balance_new = safe_get(source, "destination", "account", "balance", "new")
              doc = f"{date};{event};montant {amount};{text};source account id {source_account_id};destination account id {destination_account_id};source old balance {source_balance_old};source new balance {source_balance_new}; destination old balance {destination_balance_old}; destination new balance {destination_balance_new} \n"
              result_docs.append(doc)
         return "\n".join(result_docs)
       
       except Exception as e:
         return f"Error during RAG search: {e}"   


class transactions_search_all_input(BaseModel):
    dates: Optional[str]  = Field(None, description="Date or date range for filtering results. Specify in format YYYY-MM-DD or YYYY-MM-DD à YYYY-MM-DD.")
    sources: Optional[str] = Field(None, description="1 to several sources account ids. Specify like \"source_account_id1,source_account_id2,source_account_id3, etc...\". Starts by 'C' or 'M' followed by 9 digits at least ")
    destinations: Optional[str] = Field(None, description="1 to several destination account  ids. Specify like \"destination_account_id1,destination_account_id2,destination_account_id3,etc...\".Starts by 'C' or 'M' followed by 9 digits at least")
    type_of_transactions: Optional[str] = Field(None, description="1 to several transaction types among 'PAYMENT','DEBIT','TRANSFER','CASH_IN','CASH_OUT'") 
    

transactions_search_all_tool = StructuredTool(
    name="transactions_search_all",
    func=transactions_search_all,
    description=(
        "Use this tool to search for a list of transactions the nature of which is not precised"
        "Dates are optional buf if present  must be specified in this format YYYY-MM-DD or YYYY-MM-DD à YYYY-MM-DD."
        "Source and Destination account ids are optonal identifiers starting by 'C' or 'M' followed by at least 9 digits."
        "Transaction types are also optional."
    ),
    args_schema=transactions_search_all_input
)

# List of tools
tools = [transactions_search_tool,transactions_search_all_tool]

#design de l'interface 
st.button('Reset Chat', on_click=reset_conversation)
with st.container():
     ui_display_logged_user()
     chat_output_container= st.container()
     stream_container = st.empty()
     chat_input_container = st.container()
     chat_debug_container = st.container()
     stream_handler = StreamHandler(stream_container) 
    
     #definition du llm 
     myazureopenai_llm_stream=AzureChatOpenAI(
             openai_api_key=openai.api_key,
             azure_endpoint=openai.api_base,
             openai_api_version=openai.api_version,
             model_name=openai.model_name,
             deployment_name=deployment_name,
             streaming=True,
             callbacks=[stream_handler])

     #myazureopenai_llm_stream=AzureChatOpenAI(
     #  model=openai.model_name,
     #  openai_api_key=openai.api_key,
     #  azure_endpoint=openai.api_base,
     #  openai_api_version=openai.api_version,
     #  api_version=openai.api_version,
     #  model_name=openai.model_name,
     #  deployment_name=deployment_name,
     #  azure_deployment=deployment_name,
     #  streaming=True,
     #  callbacks=[stream_handler]) 

 
     #definition du RAG agentic 
     prompt = ChatPromptTemplate.from_messages([     
       ("system","""
          Tu es un assistant serviable, expert antifraude travaillant dans une banque et  qui a pour but d'explorer une liste de transactions financières et de répondre aux questions de l'utilisateur en conséquence. 
          Les différents types de transactions sont:
          CASH-IN
          ➤ Quand un utilisateur dépose de l’argent sur son compte mobile via un commerçant (agent).
          ➤ Le solde du compte augmente.
          CASH-OUT
          ➤ L’inverse du CASH-IN : l’utilisateur retire de l’argent de son compte mobile chez un commerçant.
          ➤ Le solde du compte diminue.
          DEBIT
          ➤ Similaire à un CASH-OUT, mais ici, l’argent est envoyé vers un compte bancaire.
          ➤ Cela diminue aussi le solde du compte mobile.
          PAYMENT
          ➤ L’utilisateur paie un commerçant pour acheter un bien ou un service.
          ➤ Le solde de l’utilisateur diminue, et celui du commerçant augmente.
          TRANSFER
          ➤ L’utilisateur envoie de l’argent à un autre utilisateur du même service.
          ➤ Le solde de l’émetteur diminue, celui du destinataire augmente. 
         
          Regarde toujours dans l'historique de la conversation pour voir si tu ne trouverais pas la réponse à la question de l'utilisateur.
          Si tel n'est pas le cas, tu disposes des outils suivants:
          - **transactions_search**: Utilise cet outil pour rechercher les transactions dont l'utilisateur doit te préciser la nature ou le libellé. 
          - **transactions_search_all**: Utilise cet outil pour rechercher les transactions dont la nature ou le libellé ne sont pas précisés du tout. 
        
          **Instructions importantes**
          - **Extrait les dates ou périodes de la question utilisateur lorsqu'il y'en a**
          - **Extrait le ou les comptes de destinations ou d'origine de la question utilisateur lorsqu'ils sont mentionnés. Ils commmencent par la lettre 'C' ou 'M' suivi d'au moins 9 chiffres**
          - **Extrait le ou les types de transactions lorsque c'est spécifié par l'utilisateur**

          Lorsque tu décides d'utiliser l'outil, respecte exactement ce formalisme:
          Action: [L'action à prendre, doit être l'un des outils [transactions_search,transaction_search_all]]
          Action Input: {{\"query\":\"la question utilisateur\", \"dates\":\"la date ou la périod entre 2 dates\", \"sources\":\"le ou les comptes à l'origine de la transaction\",\"destinations\":\"le ou les comptes de destination de la transation\",\"type_of_transactions\":\"le ou les types de transactions\"}}
          
          **Examples**
          - **Question utilisateur** \"Qui a acheté des produits chimiques?\"
          - Action: transactions_search
          - Action Input: {{\"query\":\"produits chimiques\",\"type_of_transactions\":\"PAYMENT\"}} 
           
          - **Question utilisateur** \"Liste moi les transactions concernant les montres de luxe entre le 1er janvier 2023 et le 31 mars 2024\"
          - Action: transactions_search
          - Action Input: {{\"query\":\"montres de luxe\",\"type_of_transactions\":\"PAYMENT\",\"dates\":\"2023-01-01 à 2024-03-31\"}}
          
          - **Question utilisateur** \"est ce que C1890099098 a réservé un sejour à la montagne\"
          - Action: transactions_search
          - Action Input: {{\"query\":\"réservation séjour à la montagne\",\"sources\":\"C1890099098\"}}

          - **Question utilisateur** \"Qui a remboursé un prêt familial à M1890099098\"
          - Action: transactions_search
          - Action Input: {{\"query\":\"rembourser un prêt familial\","\destinations\":\"M1890099098\"}}

          - **Question utilisateur** \"Quelles sont toutes les transactions à destination de M1890099098\"
          - Action: transactions_search_all
          - Action Input: {{"\destinations\":\"M1890099098\"}}

          - **Question utilisateur** \"Quelles sont tous les paiements effectués par C1890099098 au 1er janvier 2023\"
          - Action: transactions_search_all
          - Action Input: {{"\sources\":\"C1890099098\",\"dates\":\"2023-01-01 à 2023-01-01\", \"type_of_transactions\":\"PAYMENT\"}}
         
          - **Question utilisateur** \"Est-ce que C1890099098 a reçu des transfers\"
          - Action: transactions_search_all
          - Action Input: {{"\destinations\":\"C1890099098\", \"type_of_transactions\":\"TRANSFER\"}}
          
          - **Question utilisateur** \"liste toutes les transactions reçues par C0000001345 et C0000001348\"
          - Action: transactions_search_all
          - Action Input: {{"\destinations\": \"C0000001345,C0000001348\"}}
           
 
          Rappelle-toi
          - ton but est d'assister l'utilisateur en utilisant les outils quand c'est nécessaire
          - Sois précis et concis. N'invente pas. Réponds en Français
          - N'ajoute pas d'explication additionnelles
          - Respecte le formatage des données.  
          - Présente toutes les transactions trouvées dans un tableau mentionnant la date et l'heure (ex jeudi 24 mars 2025 à 16h10) , le compte source, le compte de  destination, le type de transaction, la description (text), le montant libellé en euros. Rajoute **TOUJOURS** une ligne au bas du tableau cumulative (ie somme des montants dans la colonne correspondante des montants)
          - A part le point pour séparer partie entière et partie décimale d'un montant, n'utilise **JAMAIS** de séparateur (virgule ou espace) entre les centaines, les milliers, les millions, ...etc  
          - N'oublie jamais de répondre précisément à la question utilisateur en faisant une phrase.   
          - Ne donne jamais dans ta réponse à l'utilisateur l'**Action** que tu as choisi ni  les **Action Input**
          - Nous sommes à l'anti-fraude. Donc tu peux aussi donner une interprétation à l'ensemble des achats effectués par une personne ou un groupe de personnes en t'appuyant sur le libellé des transactions ou regarder de plus près l'ancien et nouveau solde des comptes sources et destinataires lors de ces transactions.      
       """),
       MessagesPlaceholder(variable_name="chat_history"),
       ("human", "{input}"),
       MessagesPlaceholder(variable_name="agent_scratchpad")       
     ])
     
     agent = create_openai_functions_agent(llm=myazureopenai_llm_stream, tools=tools, prompt=prompt) 
     agent_chain = AgentExecutor(agent=agent, tools=tools, memory=st.session_state['my_memory'], verbose=False, handle_parsing_errors=True) 
 
     with chat_output_container:
       	if "my_memory" in st.session_state and len(st.session_state['my_memory'].chat_memory.messages)>0:
            for msg in st.session_state['my_memory'].chat_memory.messages:
                if msg.type=="ai":
                   st.chat_message(msg.type,avatar="👨").write(msg.content)
                else:
                   st.chat_message(msg.type,avatar="🧑").write(msg.content)
        else:
            st.chat_message("ai",avatar="👨").write("Comment vous aider?")

     with chat_input_container:
        if user_input:=st.chat_input("Envoyez votre message ici"):
            with st.spinner("Processing..."):
                chat_output_container.chat_message("human",avatar="🧑").write(user_input)
                my_query_body=None
                try:
                   resp=agent_chain.invoke({"input":user_input})
                   transactions_filtered = parse_markdown_transactions(resp['output'])
                   #print(resp['output'])
                   #print("\n--------------------------------------\n")
                   #print(transactions_filtered)
                   if transactions_filtered is not None:
                      plot_somme_montants_par_jour(transactions_filtered, chat_debug_container) 
                   if my_query_body is not None:
                      chat_debug_container.write(f"**Requête associée**")
                      chat_debug_container.write(my_query_body)
                   if result_docs:   
                      chat_debug_container.write(f"**Documents associés**")
                      chat_debug_container.write(result_docs)
                      dates=[]
                except Exception as e:
                      # Gestion spécifique de l’erreur Azure OpenAI filtrage
                      if hasattr(e, "args") and len(e.args) > 0 and "content_filter" in str(e.args[0]):
                         chat_debug_container.error("⚠️ La requête a été bloquée par le filtre de contenu d'Azure OpenAI.")
                      else:
                         # Pour toute autre erreur inattendue
                         chat_debug_container.error(f"Erreur inattendue : {e}")   
