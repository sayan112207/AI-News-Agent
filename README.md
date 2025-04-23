# AI News Agent
A **News Agent** that provides users with up-to-date news summaries and context-aware answers. It leverages embedding-based retrieval for efficiency, LLM-powered summarization for clarity, and multi-modal I/O (voice/text) for accessibility.

---
## Features Observed

- **Transformer-based Embeddings**: Uses `SentenceTransformer('all-MiniLM-L6-v2')` to encode text into 384-dimensional vectors.
- **FAISS Vector Store**: Integrates `faiss.IndexFlatL2` for efficient similarity search and persistence of vector indices.
- **Persistent Document Store**: Serializes/reloads document metadata (`full_text`, `summary`, `url`) using Python's `pickle`.
- **News Fetching via SerpAPI**: Retrieves news articles (`nws` engine) with configurable topic, location, language, and count.
- **Web Scraping with BeautifulSoup**: Extracts the first 20 `<p>` tags of article content for deeper context.
- **LLM Summarization and QA**: Calls Google Gemini API (`gemini-1.5-flash-latest`) for summarization, general overview, context-grounded answers, and follow-up verification.
- **Dynamic Topic/Location Parsing**: Regex-based detection and validation (`validate_location_with_gemini`) to update search context.
- **Semantic Query Search**: Embeds user queries into the same vector space and performs k-NN retrieval to surface semantically related articles.
- **Contextual Retrieval & Follow-up Handling**: Queries vector DB using pronoun-detection in user input and multi-turn chat history.
- **Fallback to Live Search**: If vector retrieval or grounding fails, performs a fresh SerpAPI search and reindexes.
- **Voice Interaction**: Supports text-to-speech (`pyttsx3`) and speech-to-text (`speech_recognition`) for full voice-mode conversation.
- **Agent Orchestration**: `NewsAgent` class encapsulates pipelines: fetching, indexing, summarization, querying, and speaking.

---
## Tech Stack

| Component             | Technology/Library                       | Purpose                                                |
|-----------------------|------------------------------------------|--------------------------------------------------------|
| Embedding Model       | Sentence Transformers (all-MiniLM-L6-v2) | Converts text to dense vectors for semantic search     |
| Vector Database       | FAISS                                    | Efficient similarity search over embeddings            |
| LLM API               | Gemini API                               | Summarization, question answering, validation          |
| News/Web Search       | SerpAPI (Google News)                    | Fetches real-time news articles                        |
| Web Scraping          | BeautifulSoup                            | Extracts full article text from news URLs              |
| Voice Recognition     | SpeechRecognition, pyttsx3               | Enables voice input/output                             |
| Environment Management| python-dotenv                            | Loads API keys and config from .env                    |
| Data Persistence      | pickle                                   | Saves/reloads vector DB and documents                  |


## Processes & Techniques

### 1. Embedding Generation
- **What**: Transforms text (full articles) into fixed-size vectors via `all-MiniLM-L6-v2`.
- **How**: `SentenceTransformer.encode` followed by conversion to `float32` and addition to FAISS index.
- **Why**: Embeddings allow efficient nearest-neighbor retrieval, capturing semantic similarity beyond keyword matching.

### 2. Vector Store & Persistence
- **What**: Uses FAISS `IndexFlatL2` to store and query vectors.
- **How**: On initialization, loads existing index and document metadata (`docs.pkl`). On update, writes index (`vector.index`) and pickled docs back to disk.
- **Why**: Persistence ensures a growing knowledge base across sessions and avoids re-indexing from scratch.

### 3. News Fetching
- **What**: Retrieves raw news article metadata from SerpAPI (Google News engine).
- **How**: Issues HTTP GET to `https://serpapi.com/search?engine=nws` with query params (`topic`, `location`, `num`, `lang`).
- **Why**: Provides recent, localized news without manual scraping of multiple outlets.

### 4. Article Scraping
- **What**: Collects textual content for summarization and indexing.
- **How**: HTTP GET with a browser-like user agent using SerpAPI’s Google News engine to retrieve headlines, snippets, URLs, and metadata for recent news articles. For each article URL, fetches the HTML and parses it with BeautifulSoup, extracting the main content (first 20 `<p>` tags). If full text extraction fails (due to paywalls, errors, or missing content), falls back to using the snippet and title provided by the API. Removes extraneous HTML, navigation, and advertisements to retain only the relevant news content.
- **Why**: Ensures LLM has sufficient context to generate accurate summaries and support detailed Q&A maximizing the amount of usable context for downstream summarization and question answering..

### 6. Summarization & General Overview
 Summarization & General Overview
- **What**: Generates 2–3 sentence summaries per article and a consolidated overview.
- **How**: Prompts Google Gemini API twice: once per article, then on the joined summaries list.
- **Why**: Breaks down large text into digestible chunks, then abstracts higher-level insights for quick consumption.

### 6. Semantic Query Search
- **What**: Retrieves relevant articles by embedding the **user’s query** and performing k-nearest neighbor search on the FAISS index.
- **How**: 
  1. Encode the raw query text with the same `SentenceTransformer` to obtain a 384-d vector.
  2. Normalize or convert type to `float32` and call `index.search(embedding, top_k)`.
  3. Retrieve top-k document entries based on L2 distance, filter out zero-distance (exact duplicates).
- **Why**: Ensures the agent surfaces semantically related content even when the user’s phrasing differs from original article text.

### Summary Table: Document Storage & Retrieval Flow
| Step | Technology | Purpose |
|-------------------------|--------------------------------------------------------------------------------|-----------------------------------------------|
| Article scraping |	BeautifulSoup |	Extracts full news content from HTML pages |
| Embedding generation |	SentenceTransformer |	Converts text to dense vectors for semantic search |
| Vector storage |	FAISS |	Stores and indexes vectors for fast similarity search | 
| Metadata/document storage |	pickle |	Saves article text, summaries, and URLs alongside vectors |
| Query & retrieval |	FAISS |	Finds top-k most similar articles to a user query |
| Contextual answer generation |	Gemini API |	Uses retrieved articles to generate grounded, context-aware responses |

### 7. Query Handling & Conversational Logic
- **What**: Manages multi-turn interaction, follow-up questions, and potential topic/location changes.
- **How**: In `NewsAgent.converse`:
  1. **Chat History Tracking**: Appends each user input to `chat_history`. Maintaining a running log of user queries for context tracking and multi-hop reasoning. Every new user_input is appended to `self.chat_history`.
  2. **Topic/Location Parsing**: Allows dynamic switching of the news topic or location during a session using regex (`parse_t_l`) to detect statements like `<topic> in <location>` or `<topic> for <location>`, and `about <topic>`.
  3. **Validation & Refresh**: If a new topic or location is detected (i.e., different from current), the system validates the location using Gemini `validate_location_with_gemini`; upon success, it refreshes the news corpus and vector index (`refresh_news`), then returns the updated general summary.
  4. **Pronoun & Follow-Up Detection**: Scans input for pronouns (`he, she, they, it, who, that, this, his, her`) to decide if context query should be built from last 3 turns.
  5. **Semantic Retrieval**: Executes the semantic query search as described above, returning relevant docs.
  6. **Context-Grounded Response**: Feeds retrieved documents’ `full_text` into Gemini API to generate an answer, then verifies grounding by prompting Gemini to check if its response is supported.
  7. **Fallback Mechanism**: If no relevant docs or grounding fails, performs a fresh SerpAPI search on the raw user input, reindexes new articles, and answers from this live context.
- **Why**: Balances coherence (multi-turn awareness), accuracy (grounding checks), and freshness (dynamic news fetch).

### Summary Table: User Input and Feedback Handling

| Stage	                  | App Logic Action	                                                           | User Feedback Provided
|-------------------------|--------------------------------------------------------------------------------|-----------------------------------------------|
| Input Reception	  | Accepts |and logs user query (text/voice)	                                   | Confirmation of input received  		   |
| Parsing & Validation	  | Detects topic/location or follow-up context; validates input	           | Prompts for clarification if input is invalid |
| Semantic Retrieval	  | Searches vector DB for relevant news articles	                           | Notifies if no relevant news is found	   |
| Answer Generation	  | LLM generates answer using retrieved context	                           | Provides answer and cites sources		   |
| Grounding Verification  | Confirms answer is based on news context; triggers fallback if not grounded	   | Informs user if answer is not grounded	   |
| Iterative Feedback	  | Uses chat history and feedback for multi-turn reasoning and session continuity | Maintains conversational context              |
### 8. Voice-Mode I/O
- **What**: Enables hands-free interaction.
- **How**: Uses `pyttsx3` for TTS and `speech_recognition` for STT, with ambient-noise calibration and recursive retry logic.
- **Why**: Enhances accessibility and user experience for diverse usage scenarios.

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sayan112207/AI-News-Agent.git
   cd AI-News-Agent
   ```
2. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**:
   - Create a `.env` file in the project root containing:
     ```ini
     GEMINI_API_KEY=your_gemini_api_key
     SERPAPI_API_KEY=your_serpapi_api_key
     ```

## Sample Output

```
[INFO] Loading transformer model for embeddings...
[INFO] Model loaded.
=== News Agent AI ===
Would you like to converse in voice mode? (yes/no): yes
[Voice Mode] Listening...
You (voice): recent terrorist attacks
[Voice Mode] Listening...
You (voice): India
[INFO] Refreshing news articles and building vector index...
[INFO] Fetching news for 'recent terrorist attacks' in 'India'...
[INFO] News fetched.
[INFO] Generating answer using Gemini API...
[INFO] Vector DB saved to disk.
[INFO] News summaries and general overview generated.
--- Article Summaries ---
1. Gunmen attacked a group of domestic tourists in Pahalgam, Kashmir, killing at least two dozen and wounding many more in an unprecedented assault on civilians.  The attack, condemned internationally, prompted a swift security response from Indian authorities and triggered widespread outrage.  No group has yet claimed responsibility for the violence, which has renewed concerns about the ongoing insurgency in the region.

(Source: BBC)
URL: https://www.bbc.com/news/articles/cy9vyzzyjzlo
2. Suspected militants opened fire on tourists in the Pahalgam area of Indian-administered Kashmir, killing at least 20 and injuring dozens more.  The attack, the deadliest in the region in nearly a year, targeted a popular tourist destination and occurred amidst a period of recovering tourism.  Indian officials have condemned the attack and vowed to find those responsible.

(Source: CBS News)
URL: https://www.cbsnews.com/news/india-kashmir-terror-attack-tourists-killed-wounded-pahalgam/
3. Please provide me with the text of the Reuters news article. I need the article's content to summarize it for you.

(Source: Reuters)
URL: https://www.reuters.com/world/india/one-killed-seven-injured-militant-attack-indias-kashmir-india-today-tv-says-2025-04-22/
4. In Indian-administered Kashmir, gunmen killed at least 20 tourists and injured dozens more in what police called a major terrorist attack at a popular resort.  The attack, which targeted mostly Indian tourists, marks a significant escalation of violence in the region, where tourists have previously been largely spared.  Authorities are searching for the attackers and have condemned the act.

(Source: PBS)
URL: https://www.pbs.org/newshour/world/indian-police-say-gunmen-kill-at-least-20-tourists-wound-dozens-of-others-at-a-kashmir-resort       
5. At least 26 tourists were killed and 17 wounded in a deadly attack in Indian-administered Kashmir's Pahalgam, marking one of the deadliest attacks on civilians in the region in years.  The little-known group Kashmir Resistance, allegedly linked to Pakistan-based organizations, claimed responsibility, citing grievances over demographic changes.  Indian Prime Minister Narendra Modi condemned the attack and vowed to bring the perpetrators to justice.

(Source: Al Jazeera)
URL: https://www.aljazeera.com/news/2025/4/22/gunmen-open-fire-on-tourists-in-indian-administered-kashmir

--- General Summary ---
Gunmen launched a deadly attack on a group of primarily Indian domestic tourists in Pahalgam, Kashmir, resulting in at least 26 deaths and numerous injuries.  This is one of the deadliest attacks on civilians in the region in years, representing a significant escalation of violence in an area experiencing a period of tourism recovery.  While initial reports varied slightly on the casualty figures, all sources confirm a large-scale assault targeting a popular tourist destination.  The attack has been met with widespread international condemnation.  While several sources initially reported that no group had claimed responsibility, Al Jazeera attributed the attack to the Kashmir Resistance, a little-known group allegedly linked to Pakistan-based organizations, citing grievances over demographic changes as their motivation.  Indian authorities have launched a security response and vowed to apprehend those responsible.

[Voice Mode] Listening...
Sorry, I did not understand. Please try again.
You (voice): when did the attack happened
[INFO] Checking if user input matches existing topic or is a follow-up...
[INFO] Detected a pronoun or follow-up reference. Using chat history for context.
[INFO] Found matching documents. Generating contextual answer...
[INFO] Generating answer using Gemini API...
Agent: The attack happened on Tuesday, April 22, 2025.


References:
[1] https://www.pbs.org/newshour/world/indian-police-say-gunmen-kill-at-least-20-tourists-wound-dozens-of-others-at-a-kashmir-resort        
[2] https://www.cbsnews.com/news/india-kashmir-terror-attack-tourists-killed-wounded-pahalgam/
[3] https://www.aljazeera.com/news/2025/4/22/gunmen-open-fire-on-tourists-in-indian-administered-kashmir
You (voice): exit
Agent: Goodbye!
```

The agent will prompt for topic and location via voice, then read out summaries and answers.

## File Structure
```
vector_model.py  # Main agent logic
view.py          # Utility to inspect saved FAISS index and documents
vector.index     # Serialized FAISS index (auto-created)
docs.pkl         # Serialized document metadata (auto-created)
requirements.txt # Python dependencies
README.md        # This file
```

## Why This Architecture?

- **Modularity**: Each component (fetching, embedding, summarization, storage, I/O) is encapsulated, easing testing and extension.
- **Scalability**: FAISS handles millions of vectors; the agent can scale to larger corpora or different domains.
- **Reliability**: Grounding checks enforce factual consistency; fallbacks ensure uptime even if the index is empty or stale.
- **Accessibility**: Text and voice interfaces cater to varied user preferences and contexts.

