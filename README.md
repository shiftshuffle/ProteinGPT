# ProteinGPT

ProteinGPT Analizes Biomedical text in form of chat and proceeds to identify NER (Named Entity Recognition) on proteins in the text to then enhance the text and find the Uniprot protein Identifier, which enables to have details from the protein, mainly from this 2 sources:
1. https://www.uniprot.org
2. https://mobidb.bio.unipd.it

Uniprot being a freely accessible database of protein sequence and functional information, many entries being derived from genome sequencing projects. It contains a large amount of information about the biological function of proteins derived from the research literature.

Here we can see it in action:


Sample Input:
```'Here , it is shown that the product of the argP gene , the ArgP protein , is a modulator molecule that regulates the expression of the arginine transport system'```

Sample Output:
 'Here , it is shown that the product of the [argP^1](https://www.uniprot.org/uniprotkb/P0A8S1/entry) [argP^2](https://mobidb.bio.unipd.it/P0A8S1) gene , the [ArgP^1](https://www.uniprot.org/uniprotkb/P0A8S1/entry) [ArgP^2](https://mobidb.bio.unipd.it/P0A8S1) protein , is a modulator molecule that regulates the expression of the arginine transport system'



Sample Input:
```'A recent study identified a novel interaction between the AKT1 gene and the mTOR protein'```

Sample Output:
'A recent study identified a novel interaction between the [AKT1^1](https://www.uniprot.org/uniprotkb/Q96B36/entry) [AKT1^2](https://mobidb.bio.unipd.it/Q96B36) gene and the [mTOR^1](https://www.uniprot.org/uniprotkb/P42345/entry) [mTOR^2](https://mobidb.bio.unipd.it/P42345) protein'


# Links

example literature about proteins ()[https://www.bmglabtech.com/en/blog/misfolded-proteins-and-neurodegenerative-diseases/]
(https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)[]


## Setup

1. Create a .env file in the project root and add your OpenAI API key:

```OPENAI_API_KEY=YOUR_API_KEY='xxxxxxxx'```

# Run it locally
```
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run the app.py script with Streamlit:

```streamlit run app.py```

The default configuration should spin up a web server on port 8501. You can open the web app by navigating to http://localhost:8501 in your browser.
