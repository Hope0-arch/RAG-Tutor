import customtkinter as ctk
from tkinter import *
from tkinter import messagebox, filedialog
import threading
import ollama as olm
import re
import rag_algorithm
import numpy as np


phy_records = rag_algorithm.reader(
    r"C:\Users\aurob\Downloads\leph103.pdf"
)

math_records=rag_algorithm.reader(
    r"C:\Users\aurob\Downloads\lemh106.pdf"
)
chem_records=rag_algorithm.reader(
    r"C:\Users\aurob\Downloads\lech205.pdf"
)

physics_arr = rag_algorithm.load_or_create_embeddings(
    phy_records,
    "physics_embeddings.npy"
)

math_arr = rag_algorithm.load_or_create_embeddings(
    math_records,
    "math_embeddings.npy"
)

chem_arr=rag_algorithm.load_or_create_embeddings(
    chem_records,
    "chem_embeddings.npy"
)

records = chem_records + math_records + phy_records

rec_arr =np.vstack(
    (chem_arr,math_arr,physics_arr)
)
SYSTEM_PROMPT="""Your name is Socrates. You are a friendly AI companion and tutor.

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

Never claim that the user said, knows, likes, did, or believes something
unless it is actually present in the conversation.

Never invent personal details about people mentioned by the user.

If you make an inference, clearly identify it as an inference.

If asked how you know something, check the conversation and retrieved material.
If there is no evidence, say that you do not know instead of inventing an explanation.

Never state that the student calculated, understood, answered, or discovered something unless their latest messages explicitly contain it.

When the student asks to be guided through a problem:
- Do NOT explain the complete solution in your first response.
- Give only the minimum information needed for the next reasoning step.
- Ask exactly one guiding question, then stop.
- Wait for the student's response before continuing.
- Do not answer your own guiding question.
- Avoid long analogies unless the student is struggling with the concept.

Do not roleplay as the historical Socrates.
Do not use theatrical phrases such as "Ah, you have stumbled upon..."
Speak like a modern, natural tutor.
"""


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT

    }
]



win = ctk.CTk()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

win.title("Socrates")

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
    for result in context:
        print("\n-------------------------")
        print("SOURCE:", result["source"])
        print("PAGE:", result["page"])
        print("SCORE:", result["score"])
        print(result["text"])

    retrieved_text = ""

    for result in context:
        retrieved_text += f"""
Source: {result["source"]}
Page: {result["page"]}

{result["text"]}

"""

    combined_message = {
        "role": "system",
        "content": f"""
{SYSTEM_PROMPT}

GROUNDING RULES:

The following material was retrieved from the student's study material.

Use it as the primary factual source when it is relevant.

Do not invent details that are absent from the retrieved material.
Do not claim the material contains something unless it actually does.
If the retrieved information is insufficient, explicitly say that.

For a problem-solving request, do not immediately dump the complete solution.
Guide the student through one meaningful step at a time.

For a request to explain or describe something, a direct explanation is allowed.

At the end of an academic response, state:
Source: <filename>, page <page>

Use only source/page information provided in the retrieved material.

RETRIEVED MATERIAL:

{retrieved_text}
"""
    }

    request_message = (
        [combined_message]
        + messages[1:-1]
        + [messages[-1]]
    )
    stream = olm.chat(
        model="phi4-mini",
        messages=request_message,
        stream=True,
        options={"temperature":0.2}
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











