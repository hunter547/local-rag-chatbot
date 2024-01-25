import { LocalFileStore } from "langchain/storage/file_system";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CHROMA_DB_CONFIG, GPT4ALL_LLM, HF_EMBEDDINGS } from "./config";
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";
import { ParentDocumentRetriever } from "langchain/retrievers/parent_document";
import { createDocumentStoreFromByteStore } from "langchain/storage/encoder_backed";
import { createCompletion } from "gpt4all";
import { run } from "@backroad/backroad";

const LLM = GPT4ALL_LLM();
const LOADING_STRING = "Searching...";

async function getResponse(
	query: string,
	parentDocumentRetriever: ParentDocumentRetriever
) {
	const relevantDocuments = await parentDocumentRetriever.getRelevantDocuments(
		query
	);

	const context = relevantDocuments.map((doc) => doc.pageContent).join("\n");

	const template = `[INST] <<SYS>> Use the following pieces of context to answer the question at the end. \
		  If you don't know the answer, just say that you don't know, don't try to make up an answer. \
		  Use three sentences maximum and keep the answer as concise as possible. Don't mention context in the response. <</SYS>> \
		  ${context} \
		  Question: ${query} \
		  Helpful Answer:[/INST]`;

	console.log(
		"Initiating chat completion for query:",
		`"${query.replaceAll("\n", "")}"\nThis may take some time...`
	);
	const response = await createCompletion(await LLM, [
		{
			role: "user",
			content: template,
		},
	]);

	return {
		llmResponse: response.choices[0].message.content,
		relevantDocuments,
	};
}

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

	run(async (br) => {
		br.title({ label: "Pride Auto Care Chat 🤖" });
		br.write({
			body: `# Pride Auto Care Chat 🤖`,
		});
		const messages = br.getOrDefault("messages", [
			{ by: "ai", content: "What would you like to know?" },
		]);

		messages.forEach((message) => {
			br.chatMessage({ name: message.by }).write({ body: message.content });
		});

		const input = br.chatInput({ id: "input" });
		const alreadySearching = messages
			.map((message) => message.content)
			.includes(LOADING_STRING);
		if (input && !alreadySearching) {
			br.setValue("messages", [
				...messages,
				{ by: "human", content: input },
				{ by: "ai", content: LOADING_STRING },
			]);
		} else if (!input && alreadySearching) {
			const currentInput =
				messages[
					messages.findIndex((message) => message.content === LOADING_STRING) -
						1
				].content;
			if (currentInput) {
				const response = await getResponse(
					currentInput,
					parentDocumentRetriever
				);
				const filteredMessages = messages.filter(
					(message) => message.content !== LOADING_STRING
				);
				br.setValue("messages", [
					...filteredMessages,
					{
						by: "ai",
						content: `${response.llmResponse}

### Source Documents

---

${response.relevantDocuments
	.map((doc, i) => {
		let text = `

**Doc ${i + 1}**

> ${doc.pageContent.replace(/(\n){2,}/g, "\n")}

URL: [${doc.metadata.url}](${doc.metadata.url})

---`;
		return text;
	})
	.join("\n")}
`,
					},
				]);
			}
		}
	});
}

main();
