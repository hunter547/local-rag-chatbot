from config import get_parent_document_retriever
from langchain.chains import RetrievalQA
from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st

PATH = (
    "/Users/h.evanoff/Library/Application Support/nomic.ai/GPT4All/mistral-7b-openorca.Q4_0.gguf"
)
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=PATH, callbacks=callbacks, verbose=True)

# Prompt
template = """[INST] <<SYS>> Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. <</SYS>>
{context}
Question: {question}
Helpful Answer:[/INST]"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

parent_document_retriever = get_parent_document_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=parent_document_retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
)

st.title("Pride Auto Care Chat 🤖")
question = st.text_input("What would you like to know?")

if question:
    result = qa_chain.invoke({"query": question})
    st.subheader("Result")
    st.write(result)
