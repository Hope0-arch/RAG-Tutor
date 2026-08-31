# AI Socratic Tutor which uses RAG

Currently doing a prototype but will improve it when as time goes

Socratic AI Tutor refers to an AI tutor wh uhhich aims . to help the student to understand the underlying concept and make them capable of reasoning with the concept instead of just blindly checking for the answer.

This was made possible by the ollama software. OLLAMA has been used here to download and run llms locally in our laptop/desktop. I have given the instructions to download and run your llms using ollam.

The RAG part of this project is just an AI technique which helps the AI to read sources fed to it and then generate answers appropriately instead of giving irrelevant answers without understanding what it itself is saying and using its own metrics to weigh the importance of the sources it has

Full form of RAG is:
R- Retrieval
A- Augmented
G- Generation

# What we do is basically this:
1.) First create embeds of the sources we provide and also of the question
2.) Use vector dot product and find the max value of the embed matrix which means just finding out the most appropriate source out of the given ones
3.) Feed the retrieved data to the AI so that it'll read the pre-defined context and the new data
4.) Use the data to generate proper answer

# What I have done:

1.) Created a very basic GUI using customtkinter which includes an entry widget, scrollable frame and the ability to stream the ai responses.
2.) Used ollama to pull a local LLM. Currently using gemma3:4b
3.) Completed the RAG algorithm but just the base level logic to make AI retrieve the sources

# How to do it?

You yourself can easily do this project by first downloading ollama [ LINK: https://ollama.com/download/windows ]

STEP 1:
After downloading ollama enter this command into the terminal:

ollama pull gemma3:4b { 3.3 GB SIZE }  
 ollama pull embeddinggemma { 621 MB SIZE }

currently I'm using this LLM but remember to find an LLM which suits your project and your hardware!!

STEP 2:
Install these python libraries ( Choose based on your project but these are small stuff)

pip install customtkinter  
       pip install ollama  
       pip install pypdf

STEP 3:
Just try to learn customtkinter till what you feel is best for you cause it's currently the best for rapid prototyping in my honest opinion

STEO 4: PLEASE UNDERSTAND WHAT THE CODE IS DOING AND WHAT ALGORITHM YOU ARE TYPING, The RAG algorithm is very simple but it'll be a bit confusing so don't worry if you don't get it first try

