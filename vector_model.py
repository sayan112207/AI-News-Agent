import os
import requests
import json
import re
import pickle
import faiss
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pyttsx3
import speech_recognition as sr

print("[INFO] Loading transformer model for embeddings...")
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("[INFO] Model loaded.")

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
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

class FaissVectorDB:
    def __init__(self, index_path="vector.index", doc_path="docs.pkl"):
        self.index_path = index_path
        self.doc_path = doc_path
        self.documents = []
        if os.path.exists(index_path) and os.path.exists(doc_path):
            self.index = faiss.read_index(index_path)
            with open(doc_path, 'rb') as f:
                self.documents = pickle.load(f)
            print("[INFO] Vector DB loaded from disk.")
        else:
            self.index = faiss.IndexFlatL2(384)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print("[INFO] Vector DB saved to disk.")

    def add(self, full_text, summary, url):
        embedding = embedding_model.encode([full_text])[0].astype('float32')
        self.index.add(np.array([embedding]))
        self.documents.append({"full_text": full_text, "summary": summary, "url": url})

    def query(self, text, top_k=3):
        embedding = embedding_model.encode([text])[0].astype('float32')
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        D, I = self.index.search(np.array([embedding]), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if 0 <= idx < len(self.documents) and dist != 0:
                results.append(self.documents[idx])
        return results

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
    response = requests.get("https://serpapi.com/search", params=params)
    response.raise_for_status()
    data = response.json()
    print("[INFO] News fetched.")
    return data.get("news_results", [])

def scrape_full_article(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paragraphs[:20])
    except:
        return ""

def call_gemini_api(prompt):
    print("[INFO] Generating answer using Gemini API...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
    response.raise_for_status()
    candidates = response.json().get("candidates", [])
    if candidates and "content" in candidates[0]:
        return candidates[0]["content"]["parts"][0]["text"]
    return "No answer received from Gemini API."

def validate_location_with_gemini(location):
    prompt = f"Is '{location}' a valid geographical location? Reply only with 'yes' or 'no'."
    reply = call_gemini_api(prompt).strip().lower()
    return reply.startswith("yes")

class NewsAgent:
    def __init__(self, topic, location, num_articles=5, lang="en"):
        self.topic = topic
        self.location = location
        self.num_articles = num_articles
        self.lang = lang
        self.vector_db = FaissVectorDB()
        self.articles = []
        self.general_summary = ""
        self.chat_history = []
        self.refresh_news()

    def refresh_news(self):
        print("[INFO] Refreshing news articles and building vector index...")
        self.articles = fetch_serpapi_news(self.topic, self.location, self.num_articles, self.lang)
        docs = []
        for item in self.articles:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            source = item.get('source', '')
            full_text = scrape_full_article(link) or f"Title: {title}\nSnippet: {snippet}"
            summary = call_gemini_api(
                f"Summarize this news article in 2-3 sentences:\n{full_text}\nSource: {source}, Date: {item.get('date', '')}"
            )
            self.vector_db.add(full_text, summary + f"\n(Source: {source})", link)
            docs.append(f"{summary}\n(Source: {source})\nURL: {link}")
        joined = "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(docs)])
        self.general_summary = call_gemini_api(
            f"Given these news summaries, write a detailed but concise general summary for a reader:\n{joined}"
        )
        self.vector_db.save()
        print("[INFO] News summaries and general overview generated.")

    def converse(self, user_input):
        self.chat_history.append(user_input)
        print("[INFO] Checking if user input matches existing topic or is a follow-up...")

        # Topic/location change detection
        def parse_t_l(inp, cur_t, cur_l):
            m = re.search(r'(.+?)\s+in\s+([A-Za-z ,]+)', inp, re.IGNORECASE)
            if m: return m.group(1).strip(), m.group(2).strip()
            m = re.search(r'(.+?)\s+for\s+([A-Za-z ,]+)', inp, re.IGNORECASE)
            if m: return m.group(1).strip(), m.group(2).strip()
            if 'about ' in inp.lower():
                return inp.split('about ')[1].strip(), cur_l
            return cur_t, cur_l

        new_topic, new_location = parse_t_l(user_input, self.topic, self.location)
        if new_topic.lower() != self.topic.lower() or new_location.lower() != self.location.lower():
            print("[INFO] Topic or location change detected. Validating and refreshing...")
            if not validate_location_with_gemini(new_location):
                new_location = self.location
            test = fetch_serpapi_news(new_topic, new_location, self.num_articles, self.lang)
            if not test:
                return f"No recent news for '{new_topic}' in '{new_location}'."
            else:
                self.topic, self.location = new_topic, new_location
                self.refresh_news()
                return f"Fetched news for '{self.topic}' in '{self.location}':\n{self.general_summary}"

        # Vector DB retrieval (with pronoun/follow-up detection)
        if any(word in user_input.lower() for word in ['he', 'she', 'they', 'it', 'who', 'that', 'this', 'his', 'her']):
            print("[INFO] Detected a pronoun or follow-up reference. Using chat history for context.")
            context_query = " ".join(self.chat_history[-3:])
            relevant_docs = self.vector_db.query(context_query)
        else:
            relevant_docs = self.vector_db.query(user_input)

        if relevant_docs:
            print("[INFO] Found matching documents. Generating contextual answer...")
            # Use full_text for richer context
            context = "\n---\n".join([doc['full_text'] for doc in relevant_docs])
            prompt = (
                f"Given the following news articles:\n{context}\n\n"
                f"Answer the user's query: {user_input}\n"
                "Only use information from the above articles."
            )
            llm_answer = call_gemini_api(prompt)

            # Ask Gemini if the answer is actually grounded in the context
            verify_prompt = (
                f"Context:\n{context}\n\n"
                f"User question: {user_input}\n"
                f"LLM answer: {llm_answer}\n\n"
                "Does the LLM answer use information from the provided context to answer the user's question? Reply only with 'yes' or 'no'."
            )
            verification = call_gemini_api(verify_prompt).strip().lower()
            references = "\n".join([f"[{i+1}] {doc['url']}" for i, doc in enumerate(relevant_docs)])
            if verification.startswith("yes"):
                return f"{llm_answer}\n\nReferences:\n{references}"
            else:
                print("[INFO] Gemini judged answer is not grounded. Triggering live search fallback...")

        # If nothing relevant or answer not grounded, fetch live news/web results
        print("[INFO] Attempting live news/web search...")
        live_results = fetch_serpapi_news(user_input, "", self.num_articles, self.lang)
        if not live_results:
            return f"Sorry, I couldn't find any recent news or web results for '{user_input}'."
        docs = []
        for item in live_results:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            source = item.get('source', '')
            full_text = scrape_full_article(link) or f"Title: {title}\nSnippet: {snippet}"
            summary = call_gemini_api(
                f"Summarize this news article in 2-3 sentences:\n{full_text}\nSource: {source}, Date: {item.get('date', '')}"
            )
            self.vector_db.add(full_text, summary + f"\n(Source: {source})", link)
            docs.append(full_text)
        self.vector_db.save()
        context = "\n---\n".join(docs)
        prompt = (
            f"Given the following news/web search context:\n{context}\n\n"
            f"Answer the user's query: {user_input}\n"
            "Only use information from the above articles."
        )
        answer = call_gemini_api(prompt)
        references = "\n".join([f"[{i+1}] {item.get('link', '')}" for i, item in enumerate(live_results)])
        return f"{answer}\n\nReferences:\n{references}"

    def show_news_list(self):
        print("--- Article Summaries ---")
        for i, doc in enumerate(self.vector_db.documents, 1):
            print(f"{i}. {doc['summary']}")
            print(f"URL: {doc['url']}")
        print("\n--- General Summary ---")
        print(self.general_summary)

if __name__ == "__main__":
    print("=== News Agent AI ===")
    mode = input("Would you like to converse in voice mode? (yes/no): ").strip().lower()
    voice_mode = mode in ("yes", "y")
    if voice_mode:
        speak_text("Voice mode activated. Please state the news topic.")
        topic = get_voice_input()
        speak_text("Please state the location.")
        location = get_voice_input()
    else:
        topic = input("Topic: ").strip()
        location = input("Location: ").strip()

    agent = NewsAgent(topic, location)
    agent.show_news_list()

    if voice_mode:
        speak_text("You can now start asking your questions. Say 'exit' to quit.")
    else:
        print("\nYou can now chat with the News Agent. Type 'exit' to quit.")

    while True:
        if voice_mode:
            user_input = get_voice_input()
            if user_input.lower() in ("exit", "quit"):
                speak_text("Goodbye!")
                break
            answer = agent.converse(user_input)
            print("Agent:", answer)
            speak_text(answer)
        else:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            answer = agent.converse(user_input)
            print("Agent:", answer)

