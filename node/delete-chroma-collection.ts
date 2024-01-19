import { ChromaClient } from "chromadb";

(async function () {
	const name = "primeautocare";
	const chromaClient = new ChromaClient({ path: "http://localhost:8000" });
	await chromaClient.deleteCollection({ name });
})();
