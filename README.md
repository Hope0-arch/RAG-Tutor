# AI Socratic Tutor which uses RAG
Currently doing a prototype but hope to improve it once I improve as a programmer

Socratic AI Tutor refers to an AI tutor which aims to help the student to understand the underlying concept and make him capable of reasoning with the concept instead of just blindly checking for the answer.

The RAG part of this project is just an AI technique which helps the AI to read verified sources fed to it and then generate answers appropriately instead of giving half assed answers without understanding what it itself is saying.

Full form of RAG is:
R- Retrieval
A- Augmented
G- Generation

 What we do is basically this:
   1.) First create embeds of the sources we provide and also of the question
   2.) Use vector dot product and find the max value of the embed matrix which means just finding out the most appropriate source out of the given ones
   3.) Feed the retrieved data to the AI so that it'll read the pre-defined context and the new data
   4.) Use the data to generate proper answer

# What I have done:
1.) Created a very basic GUI using customtkinter which includes an entry widget, scrollable frame and the ability to stream the ai responses.
2.) Used ollama to pull a local LLM. Currently using gemma3:4b 
3.) Created the R part of RAG


# How to do it?
  You yourself can easily do this project by first downloading ollama [ LINK: https://ollama.com/download/windows ]

  STEP 1:
    After downloading ollama enter this command into the terminal:
    
    --> ollama pull gemma3:4b { 3.3 GB SIZE }
    --> ollama pull embeddinggemma { 621 MB SIZE }
    
    currently I'm using this LLM but remember to find an LLM which suits your project and your hardware!!
  
  
  
