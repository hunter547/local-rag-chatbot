import { LocalFileStore } from "langchain/storage/file_system";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CHROMA_DB_CONFIG, GPT4ALL_LLM, HF_EMBEDDINGS } from "./config";
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";
import { ParentDocumentRetriever } from "langchain/retrievers/parent_document";
import { createDocumentStoreFromByteStore } from "langchain/storage/encoder_backed";
import { RetrievalQAChain } from "langchain/chains";
import { PromptTemplate } from "@langchain/core/prompts";
import { BaseLanguageModelInterface } from "@langchain/core/language_models/base";

async function main() {
  // Docstore and retreiver setup
  const docstore = createDocumentStoreFromByteStore(
    await LocalFileStore.fromPath("./full_docs")
  );
  const parentSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 0,
  });

  const childSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 400,
    chunkOverlap: 0,
  });

  const chromaStore = new Chroma(HF_EMBEDDINGS, CHROMA_DB_CONFIG);
  await chromaStore.ensureCollection();

  const childDocumentRetriever = ScoreThresholdRetriever.fromVectorStore(
    chromaStore,
    { minSimilarityScore: 0.01, maxK: 3 }
  );

  const parentDocumentRetriever = new ParentDocumentRetriever({
    vectorstore: chromaStore,
    childSplitter,
    parentSplitter,
    childDocumentRetriever,
    docstore,
  });

  const template =
    "[INST] <<SYS>> Use the following pieces of context to answer the question at the end. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
    Use three sentences maximum and keep the answer as concise as possible. <</SYS>> \
    {context} \
    Question: {question} \
    Helpful Answer:[/INST]";

  const chain = RetrievalQAChain.fromLLM(
    (await GPT4ALL_LLM()) as unknown as BaseLanguageModelInterface,
    parentDocumentRetriever,
    {
      prompt: new PromptTemplate({
        inputVariables: ["context", "question"],
        template,
      }),
      returnSourceDocuments: true,
    }
  );

  const results = await chain.invoke({
    query: "Are there any tire services provided?",
  });
  console.log(results);
}

main();
