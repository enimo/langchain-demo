import { SupabaseVectorStore } from "langchain/vectorstores";
import { createClient } from "@supabase/supabase-js";

import { OpenAI, OpenAIChat } from "langchain/llms";
import { OpenAIEmbeddings } from "langchain/embeddings";
// import { LLMCallbackManager } from "langchain/llms";


import { PDFLoader } from "langchain/document_loaders";
// import { CheerioWebBaseLoader } from "langchain/document_loaders";
// import { loadSummarizationChain } from "langchain/chains";
import { ChatVectorDBQAChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import { Document } from "langchain/document"

//Load environment variables (populate process.env from .env file)
import * as dotenv from "dotenv";
dotenv.config();


const log = function(args:any) { console.log(args) };


export const run = async () => {

  	const supabaseClient = createClient(
   		process.env.SUPABASE_URL || "",
    	process.env.SUPABASE_PRIVATE_KEY || ""
  	);



	// const callbackManager = {
	//   handleStart: (..._args) => {
	//     console.log(JSON.stringify(_args, null, 2));
	//   },
	//   handleEnd: (..._args) => {
	//     console.log(JSON.stringify(_args, null, 2));
	//   },
	//   handleError: (..._args) => {
	//     console.log(JSON.stringify(_args, null, 2));
	//   },
	// } as LLMCallbackManager;


	const model = new OpenAIChat({
	    openAIApiKey: process.env.OPENAI_API_KEY,
	    modelName: "gpt-3.5-turbo",
	    temperature: 0.5,
	    verbose: true,
	    cache: true,
	    // callbackManager: callbackManager,
	    // prefixMessages: [
	    //   { role: "user", content: "My name is John" },
	    //   { role: "assistant", content: "Hi there" },
	    // ],
	});




  	// 方式一，原始的简单  demo
  	/*
  	const vectorStore = await SupabaseVectorStore.fromTexts(
	    ["Hello world", "Bye bye", "What's this?"],
	    [{ id: 2 }, { id: 1 }, { id: 3 }],
	    new OpenAIEmbeddings(),
	    {
	      client: supabaseClient,
	    }
  	);

  	const resultOne = await vectorStore.similaritySearch("Hello world", 1);

  	console.log(resultOne);

	*/




  	// 方式二， 使用spilit 并且 使用已新建好的index进行查询
  	// https://www.youtube.com/watch?v=R2FMzcsmQY8
  	/*
  	const loader = new PDFLoader("resume.pdf");
	const rawDocs = await loader.load();
	  	// console.log({ docs });

	  	//  或者 读取网页内容：https://kodango.com/useful-documents-about-shell
	  	// const loader = new CheerioWebBaseLoader(
		// 	  "https://kodango.com/useful-documents-about-shell"
		// 	);  
	  	// const rawDocs = await loader.load();
	  	// console.log({ docs });

	// 完成读取后 必须进行切块
	const textSplitter = new RecursiveCharacterTextSplitter({
	    chunkSize: 1000,
	    chunkOverlap: 200,
	})
	const docs = await textSplitter.splitDocuments( rawDocs )


  	await SupabaseVectorStore.fromDocuments(
  		supabaseClient,
  		docs, 
  		new OpenAIEmbeddings()
  	);
  	*/


  	// 完成索引构建之后，开始查询
	const vectorStore = await SupabaseVectorStore.fromExistingIndex(
		supabaseClient,
		new OpenAIEmbeddings()
	);

	//Chat  history
	const chain = ChatVectorDBQAChain.fromLLM(model, vectorStore);
	const question = "哪一年在郑州大学毕业，毕业后的工作是什么";
	const res = await chain.call({ question: question, chat_history: [] });
	log({question})
	console.log(res);


};

run();