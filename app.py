from openai import OpenAI
import markdown
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Optional
import streamlit as st
from streamlit_chat import message
import os
import json
from dotenv import load_dotenv
from Bio import Entrez
from unipressed import UniprotkbClient
# Set your email address for NCBI Entrez API usage
Entrez.email = "shiftshuffle27@gmail.com"
import logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

# Setting page title and header
st.set_page_config(page_title="ProteinGPT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>ProteinGPT Analize Biomedical Text</h1>", unsafe_allow_html=True)

from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

def get_uniprot_id_for_gene(gene_name):
    logging.info(f"trying to find uniprot id for: {gene_name}")
    client = UniprotkbClient()

    # Search for the gene name in the Uniprot database
    result = client.search(query=f"(gene:{gene_name}*)")  # 9606 is the organism ID for human
    result = [i for i in result.each_record()][0]
    # Extract UniProt ID if available
    uniprot_id = result['primaryAccession']
    return uniprot_id if uniprot_id else None

def uniprot_link(gene):
    return f"https://www.uniprot.org/uniprotkb/{gene}/entry"

def mobi_db_link(gene):
    return f"https://mobidb.bio.unipd.it/{gene}"


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def find_link(entity: str) -> Optional[list]:
    """
    find protein
    """
    try:
        gene = get_uniprot_id_for_gene(entity)
        if gene:
            return [uniprot_link(gene),mobi_db_link(gene)]
    except:
        logging.error(f'Error occurred while searching link for entity {entity}')

    return None

def find_all_links(label_entities:dict) -> dict:
    """
    Finds all links for the dictionary entities in the whitelist label list.
    """
    whitelist = ['protein','gene']

    return {e: find_link(e) for label, entities in label_entities.items()
                            for e in entities
                            if label in whitelist}

def enrich_entities(text: str, label_entities: dict) -> str:
    """
    Enriches text with knowledge base links.
    """
    entity_link_dict = find_all_links(label_entities)
    logging.info(f"entity_link_dict: {entity_link_dict}")

    for entity, link in entity_link_dict.items():
        if link:
            text = text.replace(entity, f"[{entity}^1]({link[0]}) [{entity}^2]({link[1]})")
    return text

def generate_functions(labels: dict) -> list:
    return [
        {
            "name": "enrich_entities",
            "description": "Enrich Text with Knowledge Base Links",
            "parameters": {
                "type": "object",
                    "properties": {
                        "r'^(?:' + '|'.join({labels}) + ')$'":
                        {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "additionalProperties": False
            },
        }
    ]



labels = [
    "protein",
    "transcription factor",
    "gene"
]

def system_message(labels):
    return f"""
You are an expert in Natural Language Processing for Bio Medical Texts Your task is to identify common Named Entities (NER) in a given text. in this case for Uniprot Gene id protein identifiers
The possible common Named Entities (NER) types are exclusively: ({", ".join(labels)})."""

def assisstant_message():
    return f"""
EXAMPLE:
    Text: 'The BRCA1 gene is associated with an increased risk of breast and ovarian cancer...Mutations in the TP53 gene, which encodes the p53 protein'
    {{
        "protein": ["p53", "BRCA1"],
        "gene": ["TP53"]
    }}

    avoid large entities as they are ids for other databases for example instead of "protein": ["beta-amyloid protein"] you should return "protein": ["beta-amyloid"]
--"""

def user_message(text):
    return f"""
TASK:
    Text: {text}
"""




# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
          {"role": "system", "content": system_message(labels=labels)},
          {"role": "assistant", "content": assisstant_message()}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo-0613"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(labels,text):
    st.session_state['messages'].append(  {"role": "user", "content": user_message(text=text)})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=st.session_state['messages'],
        functions=generate_functions(labels),
        function_call={"name": "enrich_entities"},
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response_message = response.choices[0].message
    available_functions = {"enrich_entities": enrich_entities}
    function_name = response_message.function_call.name
    function_to_call = available_functions[function_name]
    # function_to_call = enrich_entities
    # logging.info(f"function_to_call: {function_to_call}")

    function_args = json.loads(response_message.function_call.arguments)
    logging.info(f"function_args: {function_args}")

    function_response = function_to_call(text, function_args)
    # html_string = markdown.markdown(f"""{function_response}""")
    st.session_state['messages'].append({"role": "assistant", "content":  function_response})

    total_tokens = response.usage.total_tokens
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    return function_response, total_tokens, prompt_tokens, completion_tokens





# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(labels,user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
