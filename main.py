
import customtkinter as ctk
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image
import threading
import ollama as olm
import re
import time
import pyttsx3
import speech_recognition as sr
import requests
import rag_algorithm


records = rag_algorithm.reader(r"C:\Users\aurob\Downloads\NCERT-Class-11-Physics-Part-1.pdf")

rec_arr= rag_algorithm.create_embeddings(records)
messages = [
    {
        "role": "system",
        "content": """
Your name is Socrates. You are a friendly AI companion and tutor.

For ordinary conversation — life, sports, games, music, hobbies, jokes,
opinions, or random topics — speak naturally and casually, with a small
playful or philosophical touch.

When the user genuinely wants to learn an academic or technical subject,
focus on helping them understand rather than simply giving answers.

When teaching:
- Let the student reason through steps they can reasonably discover.
- Ask one useful guiding question at a time when it helps.
- Do not immediately reveal a final answer they can work toward.
- If they struggle, gradually give stronger hints.
- Explain prerequisite knowledge directly when needed.
- Correct misconceptions and explain why they are wrong.
- Do not ask questions just for the sake of asking questions.
- Give direct explanations when that teaches better.
- Adapt to what the student has actually demonstrated they understand.

Never assume the student answered a question, understood something,
is confused, or has a misconception unless their messages actually show it.
Do not invent the student's thoughts or progress.

Respect conversation changes immediately.
If the user changes topic, asks a meta-question, pauses the lesson,
or says they want to leave, stop teaching and respond naturally.
Only resume the lesson when they want to continue.

Simply mentioning a topic does not automatically mean they want tutoring.

Never invent facts. If uncertain, say so.
"""
    }
]


win = ctk.CTk()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

win.title("Socrates")


def rofl(tt):
    engine = pyttsx3.init()
    engine.say(tt)
    engine.runAndWait()


def strip_markdown(text):
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text


scr_frame = ctk.CTkScrollableFrame(
    win,
    width=650,
    height=450
)

scr_frame.pack(
    side="top",
    padx=10
)


def get_ai_response(ai_prompt, user_text):
    context = rag_algorithm.embv(
        user_text,
        records,
        rec_arr,
        3
    )

    retrieved_text = ""

    for result in context:
        retrieved_text += f"""
Source: {result["source"]}
Page: {result["page"]}

{result["text"]}

"""

    rag_message = {
        "role": "system",
        "content": f"""
The following text was retrieved from the student's study material.

Use it as the primary factual reference IF it is relevant to the student's
current question.

Do not invent information that is not supported by the material.
If the material is unrelated or insufficient, do not pretend that it answers
the question.

RETRIEVED MATERIAL:

{retrieved_text}
"""
    }

    request_message = (
        messages[:-1]
        + [rag_message]
        + [messages[-1]]
    )

    stream = olm.chat(
        model="gemma3:4b",
        messages=request_message,
        stream=True
    )

    full_response = ""

    for chunk in stream:
        piece = chunk.message.content
        full_response += piece

        display_text = strip_markdown(full_response)

        win.after(
            0,
            lambda text=display_text:
            ai_prompt.configure(text=text)
        )

    messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )


def responses():
    text = entry.get().strip()

    if not text:
        return

    user_prompt = ctk.CTkLabel(
        scr_frame,
        wraplength=450,
        text=text,
        anchor="e",
        justify="left",
        fg_color="black",
        corner_radius=8
    )

    user_prompt.pack(
        anchor="e",
        pady=10
    )

    ai_prompt = ctk.CTkLabel(
        scr_frame,
        text="",
        wraplength=450,
        corner_radius=8,
        anchor="w",
        justify="left"
    )

    ai_prompt.pack(
        anchor="w",
        pady=10
    )

    entry.delete(
        0,
        END
    )

    messages.append(
        {
            "role": "user",
            "content": text
        }
    )

    threading.Thread(
        target=get_ai_response,
        args=(ai_prompt, text),
        daemon=True
    ).start()


entry = ctk.CTkEntry(
    win,
    placeholder_text="Enter your prompt",
    width=650,
    height=50,
    corner_radius=15
)

entry.bind(
    "<Return>",
    lambda event: responses()
)

entry.pack(
    side="bottom",
    pady=40
)


win.mainloop()











