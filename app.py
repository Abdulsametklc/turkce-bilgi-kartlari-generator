# app.py
from transformers import pipeline, AutoTokenizer
from utils import extract_text_from_pdf
import gradio as gr 

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum", use_fast=False)
# mT5 modeli, Türkçe'de iyi çalışır
generator = pipeline("text2text-generation", model="csebuetnlp/mT5_multilingual_XLSum", tokenizer=tokenizer)

def generate_flashcards(file):
    text = extract_text_from_pdf(file.name)
    chunks = text.split("\n\n")[:5]  # İlk 5 paragrafı al
    flashcards = []

    for chunk in chunks:
        prompt = f"Aşağıdaki Türkçe metne göre bilgi kartı oluştur: {chunk.strip()}\nSoru:"
        result = generator(prompt, max_length=128, do_sample=True)[0]['generated_text']
        flashcards.append(f"Soru: {result.strip()}")

    return "\n\n".join(flashcards)

iface = gr.Interface(fn=generate_flashcards, 
                     inputs="file", 
                     outputs="text",
                     title="🇹🇷 Türkçe Bilgi Kartı Üretici",
                     description="PDF içeriğinden LLM ile soru-cevap üretir.")
iface.launch()