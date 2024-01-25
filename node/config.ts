import { OpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { ChromaLibArgs } from "langchain/dist/vectorstores/chroma";
import { loadModel } from "gpt4all";
import dotenv from "dotenv";
dotenv.config({
	path: "../.env",
});

export const GPT4ALL_LLM = async () => {
	const PATH = process.env.GPT4ALL_PATH;
	const MODEL = process.env.GPT4ALL_MODEL;
	const MODEL_LIST = process.env.GPT4ALL_MODEL_LIST;
	if (!PATH || !MODEL || !MODEL_LIST) {
		throw new Error(
			"PATH, MODEL, and MODEL_LIST need to be set in the enviroment variables to locate the GPT4ALL model in your filesystem, as well as confirming the model passed in is contained in the available model list."
		);
	}
	return await loadModel(MODEL, {
		verbose: true,
		allowDownload: false,
		modelPath: PATH,
		modelConfigFile: MODEL_LIST,
		device: "gpu",
	});
};

export const HF_EMBEDDINGS = new HuggingFaceInferenceEmbeddings();

export const CHROMA_DB_CONFIG: ChromaLibArgs = {
	url: "http://localhost:8000",
	collectionName: "primeautocare",
};
