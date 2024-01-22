import { OpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { ChromaLibArgs } from "langchain/dist/vectorstores/chroma";
import { loadModel } from "gpt4all";
import dotenv from "dotenv";
dotenv.config({
  path: "../.env",
});

export const GPT4ALL_LLM = async () =>
  await loadModel("gpt4all-13b-snoozy-q4_0.gguf", {
    verbose: true,
    allowDownload: false,
    modelPath: "/Users/h.evanoff/.cache/gpt4all/",
  });

export const HF_EMBEDDINGS = new HuggingFaceInferenceEmbeddings();

export const CHROMA_DB_CONFIG: ChromaLibArgs = {
  url: "http://localhost:8000",
  collectionName: "primeautocare",
};
