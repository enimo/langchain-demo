
//Load environment variables (populate process.env from .env file)
import * as dotenv from "dotenv";
dotenv.config();


/*
import { OpenAI } from "langchain/llms";
import { BufferWindowMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

const model = new OpenAI({});
const memory = new BufferWindowMemory({ k: 1 });
const chain = new ConversationChain({ llm: model, memory: memory });

const res1 = await chain.call({ input: "Hi! I'm Jim." });
console.log({ res1 }); // {response: " Hi Jim! It's nice to meet you. My name is AI. What would you like to talk about?"}

const res2 = await chain.call({ input: "What's my name?" });
console.log({ res2 }); // {response: ' You said your name is Jim. Is there anything else you would like to talk about?'}

*/



import { OpenAIChat } from "langchain/llms";
const model = new OpenAIChat({
  modelName: "gpt-3.5-turbo",
  prefixMessages: [
    { role: "user", content: "My name is John" },
    { role: "assistant", content: "Hi there" },
  ],
});
const res = await model.call("What is my name");
console.log({ res });



/*

//Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
import { OpenAI } from "langchain/llms";

//Import the Vector DB QA chain
import { VectorDBQAChain } from "langchain/chains";

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



export const run = async () => {
  //Instantiante the OpenAI LLM that will be used to answer the question
  const model = new OpenAI({});

  //Load in the file containing the content on which we will be performing Q&A
  //The answers to the questions are contained in this file
  const text = fs.readFileSync("state_of_the_union.txt", "utf8");

  //Split the text from the Q&A content file into chunks
  //This is necessary because we can only pass text of a specifc size to LLMs.  
  //Since the size of the of the file containing the answers is larger than the max size
  //of the text that can be passed to an LLM, we split the the text in the file into chunks.
  //That is what the RecursiveCharacterTextSplitter is doing here
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });

  //Create documents from the split text as required by subsequent calls
  const docs = await textSplitter.createDocuments([text]);

  //Create the vector store from OpenAIEmbeddings
  //OpenAIEmbeddings is used to create a vector representation of a text in the documents.
  //Converting the docs to the vector format and storing it in the vectorStore enables LangChain.js
  //to perform similarity searches on the "await chain.call"
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  //Create the LangChain.js chain consting of the LLM and the vector store
  const chain = VectorDBQAChain.fromLLM(model, vectorStore);

  //Ask the question that will use content from the file to find the answer
  //The way this all comes together is that when the following call is made, the "query" (question) is used to
  //search the vector store to find chunks of text that is similar to the text in the "query" (question). Those
  //chunks of text are then sent to the LLM to find the answer to the "query" (question). This is done because,
  //as explained earlier, the LLMs have a limit in size of the text that can be sent to them
  const res = await chain.call({
    input_documents: docs,
    query: "What did the president say about the Cancer Moonshot?",
  });
  console.log({ res });
};
run();

*/
