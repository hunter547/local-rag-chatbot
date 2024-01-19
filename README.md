# Local RAG Chatbot 🤖

## Global Setup

### Docker

Insure [Docker Desktop](https://docs.docker.com/desktop/) is installed on your machine
Then run:
`docker compose up -d`
This will get a [ChromaDB](https://docs.trychroma.com) instance up and running so that documents for later retrieval can be stored

### Environment Variables

Create a `.env` file at the root of the project containing these two keys:

```
OPENAI_API_KEY
HUGGINGFACEHUB_API_KEY
```

Both of these keys can be found with accounts from both [OpenAI](https://platform.openai.com/api-keys) and [HuggingFace](https://huggingface.co/settings/tokens), respectively

## Node.js Implementation

### Node.js setup

Install dependencies

```
cd node
yarn install
```

### Node.js load process

After dependecies are installed, the load process can be started. This will scrap text from the web and load them as documents, with the urls, in the running ChromaDB container. To start the load process:
`yarn load`

## Python Implementation

### Python setup

Setup virtual environment and install dependencies
**NOTE: Ensure the Python Interpreter version is not above 3.10.x. Otherwise LangChain in the requirements list will not install**

```
cd python
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Python load process

After dependecies are installed, the load process can be started. This will scrap text from the web and load them as documents, with the urls, in the running ChromaDB container. To start the load process:
`python load.py`

### Python retrieval process

After all documents are loaded, the question and retrieval process is ready to begin. We will be running a streamlit app to have a UI for interacting with the LLM:
`streamlit run app.py`
Go to the localhost url provided in the terminal and start asking questions!
