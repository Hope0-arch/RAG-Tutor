# Socrates — AI Tutor using Retrieval-Augmented Generation (RAG)

Socrates is a prototype AI tutor designed to help students understand concepts and reason through problems instead of simply giving them final answers.

The project combines a **Socratic-style tutoring approach** with **Retrieval-Augmented Generation (RAG)** so that the AI can use trusted study material, such as textbooks and PDFs, while answering academic questions.

The application currently runs locally using **Ollama**, meaning the language models can run directly on the user's computer without requiring a cloud API.

---

## Why this project?

Large Language Models can generate very convincing responses, but those responses are not always accurate or grounded in the student's actual study material.

They may:

- Give unsupported or incorrect information
- Answer using general knowledge instead of the student's course material
- Provide the final answer without helping the student understand the reasoning
- Make assumptions that were never stated by the user

Socrates attempts to reduce these problems by combining:

- A tutoring-oriented system prompt
- Conversation memory
- Retrieval from trusted study documents
- Local language models

The goal is not to claim that hallucinations are completely eliminated, but to create a tutor that is **more grounded, transparent and useful for learning**.

---

# What is RAG?

**RAG** stands for:

- **R — Retrieval**
- **A — Augmented**
- **G — Generation**

Retrieval-Augmented Generation allows an AI system to retrieve relevant information from an external knowledge source before generating its response.

In this project, the knowledge source is currently a PDF supplied by the user.

The basic pipeline is:

```text
Study PDF
    ↓
Extract text
    ↓
Split text into overlapping chunks
    ↓
Create embeddings for each chunk
    ↓

User Question
    ↓
Create question embedding
    ↓

Compare question embedding
with document embeddings
    ↓
Retrieve the most relevant chunks
    ↓
Add retrieved information to the LLM context
    ↓
Generate the final response
