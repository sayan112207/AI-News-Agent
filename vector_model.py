import os
import requests
import json
import pickle
import faiss
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pyttsx3
import speech_recognition as sr

import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load environment and models
print("[INFO] Loading transformer model for embeddings...")
load_dotenv()
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Embedding model loaded.")

# Loading spaCy
nlp = spacy.load("en_core_web_sm")

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("[Voice Mode] Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You (voice): {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand. Please try again.")
        return get_voice_input()
    except sr.RequestError:
        print("Sorry, speech recognition service is unavailable.")
        return ""

def call_gemini_api(prompt):
    """Call the Gemini API and return the generated text."""
    print("[INFO] Generating answer using Gemini API...")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash-latest:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url,
                         headers={"Content-Type": "application/json"},
                         data=json.dumps(data))
    resp.raise_for_status()
    cands = resp.json().get("candidates", [])
    return cands[0]["content"]["parts"][0]["text"] if cands else ""

def resolve_coref(user_input, chat_history):
    """
    Rewrite pronouns in user_input by asking Gemini to resolve
    based on the last few exchanges in chat_history.
    """
    context = "\n".join(chat_history[-3:])
    prompt = f"""
The following is a conversation context:
{context}

User query: "{user_input}"

Rewrite the user query by replacing any pronouns (he, she, it, they, etc.)
with the appropriate named entity based on the context above.
Return only the rewritten query.
"""
    return call_gemini_api(prompt).strip()

def classify_intent(user_input, current_topic, current_location):
    """
    Ask Gemini to classify the user’s intent into NEW_TOPIC,
    LOCATION_CHANGE, or FOLLOW_UP.
    """
    prompt = f"""
You are a news-agent assistant.
Current topic: "{current_topic}" (location: {current_location})
User query: "{user_input}"

Classify the user’s intent into exactly one of:
- NEW_TOPIC
- LOCATION_CHANGE
- FOLLOW_UP

Reply with the label only.
"""
    return call_gemini_api(prompt).strip().upper()

def is_new_semantic_topic(user_input, topic_vec, embedder, threshold=0.4):
    """
    Return True if the query is semantically far from the current topic.
    """
    q_vec = embedder.encode(user_input, normalize_embeddings=True)
    sim  = cosine_similarity([q_vec], [topic_vec])[0][0]
    return sim < threshold

def fetch_serpapi_news(topic, location, num=5, lang="en"):
    print(f"[INFO] Fetching news for '{topic}' in '{location}'...")
    query = f"{topic} in {location}" if location else topic
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "hl": lang,
        "tbm": "nws",
        "num": str(num)
    }
    resp = requests.get("https://serpapi.com/search", params=params)
    resp.raise_for_status()
    return resp.json().get("news_results", [])

def scrape_full_article(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        paras = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paras[:20])
    except:
        return ""

class FaissVectorDB:
    def __init__(self, index_path="vector.index", doc_path="docs.pkl"):
        self.index_path = index_path
        self.doc_path   = doc_path
        self.documents  = []
        if os.path.exists(index_path) and os.path.exists(doc_path):
            self.index = faiss.read_index(index_path)
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)
            print("[INFO] Vector DB loaded.")
        else:
            self.index = faiss.IndexFlatL2(384)  # embedding dim

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_path, "wb") as f:
            pickle.dump(self.documents, f)
        print("[INFO] Vector DB saved.")

    def add(self, full_text, summary, url):
        emb = embedding_model.encode([full_text])[0].astype("float32")
        self.index.add(np.array([emb]))
        self.documents.append({
            "full_text": full_text,
            "summary": summary,
            "url": url
        })

    def query(self, text, top_k=3):
        """Return up to top_k most relevant docs for the text."""
        emb = embedding_model.encode([text], normalize_embeddings=True)[0].astype("float32")
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        D, I = self.index.search(np.array([emb]), k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results

class NewsAgent:
    def __init__(self, topic, location, num_articles=5, lang="en"):
        self.topic           = topic
        self.location        = location
        self.num_articles    = num_articles
        self.lang            = lang
        self.vector_db       = FaissVectorDB()
        self.general_summary = ""
        self.chat_history    = []
        self.topic_vec       = embedding_model.encode(topic, normalize_embeddings=True)
        self.refresh_news()

    def refresh_news(self):
        print("[INFO] Refreshing news & rebuilding vector index...")
        # Clear old docs, start fresh
        self.vector_db = FaissVectorDB()
        articles = fetch_serpapi_news(self.topic, self.location,
                                      self.num_articles, self.lang)

        for itm in articles:
            full = scrape_full_article(itm.get("link","")) or itm.get("snippet","")
            summ = call_gemini_api(
                f"Summarize this news article in 2-3 sentences:\n{full}\n"
                f"Source: {itm.get('source','')}, Date: {itm.get('date','')}"
            )
            self.vector_db.add(full, summ + f"\n(Source: {itm.get('source','')})",
                               itm.get("link",""))

        # Build general summary from only the top-K relevant docs
        top_docs = self.vector_db.query(self.topic, top_k=self.num_articles)
        joined = "\n\n".join(f"{i+1}. {d['summary']}" for i, d in enumerate(top_docs))
        self.general_summary = call_gemini_api(
            f"Given these {len(top_docs)} news summaries (most relevant to “{self.topic}”), "
            f"write a detailed but concise general summary:\n{joined}"
        )

        self.vector_db.save()
        self.topic_vec = embedding_model.encode(self.topic, normalize_embeddings=True)
        print("[INFO] News refresh done.")

    def handle_fallback(self, user_input):
        print("[INFO] Falling back to live search...")
        live = fetch_serpapi_news(user_input, "", self.num_articles, self.lang)
        if not live:
            return f"Sorry, no recent news or web results for '{user_input}'."

        docs = []
        for itm in live:
            full = scrape_full_article(itm.get("link","")) or itm.get("snippet","")
            summ = call_gemini_api(
                f"Summarize:\n{full}\nSource: {itm.get('source','')}"
            )
            self.vector_db.add(full, summ + f"\n(Source: {itm.get('source','')})",
                               itm.get("link",""))
            docs.append(full)

        self.vector_db.save()
        ctx = "\n---\n".join(docs)
        ans = call_gemini_api(
            f"Given this context:\n{ctx}\nAnswer: {user_input}\nUse only above."
        )
        refs = "\n".join(f"[{i+1}] {itm.get('link','')}"
                         for i, itm in enumerate(live))
        return f"{ans}\n\nReferences:\n{refs}"

    def converse(self, user_input):
        self.chat_history.append(user_input)
        print("[INFO] Classifying intent...")
        intent = classify_intent(user_input, self.topic, self.location)

        # override if semantically new
        if is_new_semantic_topic(user_input, self.topic_vec, embedding_model):
            intent = "NEW_TOPIC"

        # interrogative override for concise fact-queries
        first = user_input.strip().split()[0].lower()
        if intent == "NEW_TOPIC" and first in {
            "who", "what", "when", "where", "how", "why", "which"
        }:
            intent = "FOLLOW_UP"

        if intent == "NEW_TOPIC":
            print("[INFO] NEW_TOPIC detected.")
            self.topic = user_input
            self.refresh_news()
            return f"Fetched news for '{self.topic}' in '{self.location}':\n{self.general_summary}"

        if intent == "LOCATION_CHANGE":
            print("[INFO] LOCATION_CHANGE detected.")
            self.location = user_input
            self.refresh_news()
            return f"Fetched news for '{self.topic}' in '{self.location}':\n{self.general_summary}"

        # FOLLOW_UP path
        print("[INFO] FOLLOW_UP path – resolving coref...")
        resolved = resolve_coref(user_input, self.chat_history)

        # retrieve top-3 relevant docs
        relevant_docs = self.vector_db.query(resolved, top_k=3)
        if not relevant_docs:
            return self.handle_fallback(user_input)

        print("[INFO] Answering from context...")
        context = "\n---\n".join(d["full_text"] for d in relevant_docs)
        prompt = (
            f"Given these articles:\n{context}\n\n"
            f"Answer the user's query: {user_input}\n"
            "Use only the information above, and answer in one concise sentence."
        )
        answer = call_gemini_api(prompt)
        refs   = "\n".join(f"[{i+1}] {d['url']}"
                           for i, d in enumerate(relevant_docs))
        return f"{answer}\n\nReferences:\n{refs}"

    def show_news_list(self):
        print("--- Article Summaries ---")
        for i, doc in enumerate(self.vector_db.documents, 1):
            print(f"{i}. {doc['summary']}\n   URL: {doc['url']}")
        print("\n--- General Summary ---")
        print(self.general_summary)

if __name__ == "__main__":
    print("=== News Agent AI ===")
    mode = input("Voice mode? (yes/no): ").strip().lower()
    voice_mode = mode in ("yes", "y")
    if voice_mode:
        speak_text("Voice mode activated. Please state the news topic.")
        topic    = get_voice_input()
        speak_text("Please state the location.")
        location = get_voice_input()
    else:
        topic    = input("Topic: ").strip()
        location = input("Location: ").strip()

    agent = NewsAgent(topic, location)
    agent.show_news_list()

    if voice_mode:
        speak_text("You can now ask questions. Say 'exit' to quit.")
    else:
        print("\nYou can now chat. Type 'exit' to quit.")

    while True:
        user_input = get_voice_input() if voice_mode else input("You: ")
        if user_input.lower() in ("exit", "quit"):
            if voice_mode:
                speak_text("Goodbye!")
            print("Goodbye!")
            break
        ans = agent.converse(user_input)
        print("Agent:", ans)
        if voice_mode:
            # Only speak the answer portion, never the References block
            if "References" in ans:
                answer_text, _, _ = ans.partition("References")
            else:
                answer_text = ans
            speak_text(answer_text)
