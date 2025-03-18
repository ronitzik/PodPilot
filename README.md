#  PodPilot - AI-Powered Podcast Recommendation System  

##  Overview  
PodPilot is an **AI powered podcast recommendation system** designed to help users discover new podcasts based on their preferences.

###  Why PodPilot?  
Finding new podcasts that match your interests is often frustrating. Generic recommendations lead to repetitive content, making discovery time-consuming. 
PodPilot solves this problem by using AI based embeddings and search queries to enhance podcast discovery.

## Tech Stack  

### **1️⃣ Backend Framework**
- **FastAPI** → High performance RESTful API framework  
- **Uvicorn** → ASGI server to run FastAPI  
- **Postman** → API testing  

### **2️⃣ Cloud & Deployment**
- **Azure Container Apps** → Cloud hosting  
- **Docker** → Containerized FastAPI app  
- **GitHub Actions** → CI/CD pipeline  

### **3️⃣ APIs & External Integrations**
- **Taddy API** → Fetches trending podcasts  
- **Google Custom Search API** → Finds related podcasts  
- **OpenAI API** → Improves text descriptions & generates search prompts  
- **Requests** → Handles API calls  

### **4️⃣ Databases & Data Handling**
- **SQLite** → Stores podcasts & user preferences  
- **SQLAlchemy** → ORM- Object Relational Mappe  
- **Fuzzy Matching (thefuzz)** → Improves recommendations  
- **LangDetect** → Detects podcast languages  

### **5️⃣ Embeddings Creation**
- **Hugging Face Transformers Sentence-BERT (SBERT)** → Generates embeddings & Converts text into numerical vectors 


## How It Works  

### **1️⃣ Embeddings Creation Process**
1. **Fetch Podcast Data** → Retrieves top podcast series from the **Taddy API**  
2. **Detect Language** → Uses **LangDetect** to determine language  
3. **Optimize Descriptions** → If Hebrew, translates to English via **OpenAI GPT**  
4. **Generate Embeddings** → Converts podcast metadata into vectors using **SBERT**  
5. **Store Data** → Saves podcasts & embeddings in SQLite  

### **2️⃣ Recommendation Process**
1. **User Identification** → User inputs their **User ID**  
2. **Preference Collection** → User **likes/dislikes podcasts** to train the model  
3. **Search Query Generation** → **OpenAI GPT** generates a **Google Search API query**  
4. **Find Related Podcasts** → Scrapes **Google Search API** for similar podcasts  
5. **Personalized Recommendations** → **Embeddings & Fuzzy Matching** return the top 3 podcasts  

## API Endpoints  

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/suggest_podcasts` | Suggests random podcasts from DB |
| `POST` | `/process_feedback` | Saves user preferences |
| `POST` | `/generate_search_query` | Creates a search query using OpenAI |
| `POST` | `/search_related_podcasts` | Finds podcasts via Google Search |
| `POST` | `/recommend_podcasts` | Returns the top 3 podcast recommendations |
