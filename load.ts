import { readFileSync } from "fs";
import { XMLParser } from "fast-xml-parser";
import axios from "axios";
import { convert } from "html-to-text";
import { LocalFileStore } from "langchain/storage/file_system";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CHROMA_DB_CONFIG, HF_EMBEDDINGS } from "./config";
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";
import { ParentDocumentRetriever } from "langchain/retrievers/parent_document";
import { createDocumentStoreFromByteStore } from "langchain/storage/encoder_backed";
import { Document } from "@langchain/core/documents";

type Sitemap = {
	urlset: {
		url: {
			loc: string;
			priority: number;
			changefreq: string;
			lastmod: string;
		}[];
	};
};

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

	// Sitemap scrapper setup
	const parser = new XMLParser();
	const object: Sitemap = parser.parse(readFileSync("./sitemap.xml", "utf-8"));
	for (const { loc } of object.urlset.url) {
		const { data } = await axios.get(loc);
		const text = convert(data.content);

		// Start document loading
		const doc = new Document({ pageContent: text, metadata: { url: loc } });
		console.log(`Adding text from page ${loc}`);
		await parentDocumentRetriever.addDocuments([doc]);
	}
	console.log("Done.");
}

main();
