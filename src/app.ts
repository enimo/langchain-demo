
//Load environment variables (populate process.env from .env file)
import * as dotenv from "dotenv";
dotenv.config();


/*
import { OpenAI } from "langchain/llms";
import { BufferWindowMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

const model = new OpenAI(cache: true );
const memory = new BufferWindowMemory({ k: 1 });
const chain = new ConversationChain({ llm: model, memory: memory });

const res1 = await chain.call({ input: "Hi! I'm Jim." });
console.log({ res1 }); // {response: " Hi Jim! It's nice to meet you. My name is AI. What would you like to talk about?"}

const res2 = await chain.call({ input: "What's my name?" });
console.log({ res2 }); // {response: ' You said your name is Jim. Is there anything else you would like to talk about?'}

*/


/*
import { OpenAIChat } from "langchain/llms";
import { LLMCallbackManager } from "langchain/llms";

const callbackManager = {
  handleStart: (..._args) => {
    console.log(JSON.stringify(_args, null, 2));
  },
  handleEnd: (..._args) => {
    console.log(JSON.stringify(_args, null, 2));
  },
  handleError: (..._args) => {
    console.log(JSON.stringify(_args, null, 2));
  },
} as LLMCallbackManager;


const model = new OpenAIChat({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo",
  // prefixMessages: history,
  temperature: 0.7,
  verbose: true,
  // callbackManager: callbackManager,
  prefixMessages: [
    { role: "user", content: "My name is John" },
    { role: "assistant", content: "Hi there" },
  ],
});

const res = await model.call("What is my name");

console.log({ res });

*/




//Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
import { OpenAI, OpenAIChat } from "langchain/llms";


//Import the Hierarchical Navigable Small World Graphs vector store (you'll learn
//how it is used later in the code)
import { HNSWLib } from "langchain/vectorstores";

//Import OpenAI embeddings (you'll learn
//how it is used later in the code)
import { OpenAIEmbeddings } from "langchain/embeddings";

//Import the text splitter (you'll learn
//how it is used later in the code)
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

//Import file stystem node module
import * as fs from "fs";

import { Document } from "langchain/document"


import { TextLoader } from "langchain/document_loaders";
import { PDFLoader } from "langchain/document_loaders";
import { CheerioWebBaseLoader } from "langchain/document_loaders";


import { loadSummarizationChain } from "langchain/chains";
import { VectorDBQAChain } from "langchain/chains";
import { ChatVectorDBQAChain } from "langchain/chains";

import pdfParse from "pdf-parse"


export const run = async () => {

  //Instantiante the OpenAI LLM that will be used to answer the question
  // const model = new OpenAI({ cache: true });
  const model = new OpenAIChat({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    verbose: true,
    cache: true,
    // callbackManager: callbackManager,
    // prefixMessages: [
    //   { role: "user", content: "My name is John" },
    //   { role: "assistant", content: "Hi there" },
    // ],
  });

  //Load in the file containing the content on which we will be performing Q&A
  //The answers to the questions are contained in this file

  //Split the text from the Q&A content file into chunks
  //This is necessary because we can only pass text of a specifc size to LLMs.  
  //Since the size of the of the file containing the answers is larger than the max size
  //of the text that can be passed to an LLM, we split the the text in the file into chunks.
  //That is what the RecursiveCharacterTextSplitter is doing here



  // 1 读取文件，建立Document对象{pageContent:xxx, metadata:xxx}
  // 1.1 使用fs读取pdf文件, failed
  // const pdfBuffer = fs.readFileSync("miniprogram_ads_doc.pdf");
  // const { text } = await pdfParse(pdfBuffer)

  // const rawDocs = new Document({ pageContent: text })

  // const textSplitter = new RecursiveCharacterTextSplitter({
  //   chunkSize: 1000,
  //   chunkOverlap: 200,
  // })
  // const docs = await textSplitter.splitDocuments([rawDocs])


  // 1.2 使用fs读取txt文件 OK
  //Create documents from the split text as required by subsequent calls
   // const text = fs.readFileSync("state_of_the_union.txt", "utf8");
  // const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  // const docs = await textSplitter.createDocuments([text]);


  // 1.3 使用loader读取PDF 和 txt ok
  // const loader = new TextLoader("state_of_the_union.txt");
  // const loader = new PDFLoader("miniprogram_ads_doc.pdf");
  const loader = new PDFLoader("resume.pdf");
  const rawDocs = await loader.load();
  // console.log({ docs });

  // 1.4 读取网页内容：https://kodango.com/useful-documents-about-shell
  // const loader = new CheerioWebBaseLoader(
// 	  "https://kodango.com/useful-documents-about-shell"
// 	);  
  // const rawDocs = await loader.load();
  // console.log({ docs });


  // 1.9 完成读取后 必须进行切块
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  })
  const docs = await textSplitter.splitDocuments( rawDocs )



  // 2 开始建立索引 hnswlib是一个内存矢量数据库，可以换成可持久化的矢量数据库，如部署pgvector, 或PineconeStore云服务
  //Create the vector store from OpenAIEmbeddings
  //OpenAIEmbeddings is used to create a vector representation of a text in the documents.
  //Converting the docs to the vector format and storing it in the vectorStore enables LangChain.js
  //to perform similarity searches on the "await chain.call"
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());




  // 3 开始提问
  //Create the LangChain.js chain consting of the LLM and the vector store
  //Ask the question that will use content from the file to find the answer
  //The way this all comes together is that when the following call is made, the "query" (question) is used to
  //search the vector store to find chunks of text that is similar to the text in the "query" (question). Those
  //chunks of text are then sent to the LLM to find the answer to the "query" (question). This is done because,
  //as explained earlier, the LLMs have a limit in size of the text that can be sent to them
  

  	/*
	// 3.1 Call the summarization chain, ok 
	const chain = loadSummarizationChain(model);
	const res = await chain.call({
	  input_documents: docs,
	  // query: '请用中文回答',
	  // question: '请用中文回答'
	});
	console.log(res);
	*/

	
	/*
  	 //3.2 Chat  one single qusetion
  	const chain = VectorDBQAChain.fromLLM(model, vectorStore);

	const res = await chain.call({
	    input_documents: docs,
	    // query: "What did the president say about the Cancer Moonshot? answer me in zh_CN",
	    query: "What is this about? Answer me in zh_CN",
	});
	console.log({ res });
	*/


	
  	//Chat  history
	const chain = ChatVectorDBQAChain.fromLLM(model, vectorStore);
	const question = "What is this about? answer me in zh_CN";
	const res = await chain.call({ question: question, chat_history: [] });
	console.log(res);

	// Ask it a follow up question 
	const chatHistory = question + res["text"];
	const followUpRes = await chain.call({
	  question: "作者是谁，用中文回答",
	  chat_history: chatHistory,
	});
	console.log(followUpRes);
	



};

run();




