import { OpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { ChromaLibArgs } from "langchain/dist/vectorstores/chroma";
import dotenv from "dotenv";
dotenv.config({
	path: "../.env",
});

export const HF_EMBEDDINGS = new HuggingFaceInferenceEmbeddings();

export const OPEN_AI_MODEL = new OpenAI({
	modelName: "gpt-3.5-turbo",
});

export const CHROMA_DB_CONFIG: ChromaLibArgs = {
	url: "http://localhost:8000",
	collectionName: "primeautocare",
};
