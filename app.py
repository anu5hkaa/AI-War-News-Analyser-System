
import os
import numpy as np
import pandas as pd
import requests
import feedparser
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from groq import Groq

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def fetch_news_from_api(api_key: str) -> pd.DataFrame:
    url = "https://newsapi.org/v2/everything"

    war_keywords_unused = [ 
        "war", "conflict", "military", "airstrike", "missile",
        "bomb", "attack", "invasion", "army", "troops",
        "defense", "clash", "border", "violence", "strike"
    ]
    queries = [
        "war",
        "military conflict",
        "airstrike attack",
        "border clash",
        "missile strike",
        "army operation",
        "defense military",
        "troops conflict"
    ]

    all_articles = []
    news_list = []
    for query in queries:
        for page in range(1, 3):
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "page": page,
                "apiKey": api_key
            }
            response = requests.get(url, params=params)
            data = response.json()
            articles = data.get("articles", [])
            all_articles.extend(articles)

    for article in all_articles:
        news = {
            "title": article.get("title", ""),
            "content": article.get("description", ""),
            "source": article.get("source", {}).get("name", ""),
            "url": article.get("url", ""),
            "date": article.get("publishedAt", "")
        }
        news_list.append(news)
    print("=" * 50)
    print(f"Fetched Articles: {len(news_list)}")
    print(pd.DataFrame(news_list).head())
    print("=" * 50)
    return pd.DataFrame(news_list)


def filter_non_movie(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["title"].str.lower().str.contains("movie|film|trailer|review|box office", na=False)]



@st.cache_resource(show_spinner=False)
def load_zero_shot_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )


def is_war_news(text, classifier):
    labels = ["war or military news", "not related"]
    result = classifier(text, labels)
    return result["labels"][0] == "war or military news"


def apply_zero_shot_filter(df: pd.DataFrame, classifier, progress_cb=None) -> pd.DataFrame:
    mask = []
    total = len(df)

    print("Entered apply_zero_shot_filter")

    for i, title in enumerate(df["title"].tolist()):

        if i % 20 == 0:
            print(f"Processing article {i}/{total}")

        mask.append(is_war_news(title, classifier))

        if progress_cb is not None:
            progress_cb((i + 1) / total)

    return df[pd.Series(mask, index=df.index)]


war_keywords = [
    "war", "conflict", "attack", "military", "missile",
    "battle", "army", "troops", "airstrike", "bomb",
    "border", "clash", "violence", "defense", "ceasefire"
]


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def build_embeddings_index(df: pd.DataFrame, model: SentenceTransformer):
    df = df.copy()
    df['content'] = df['content'].fillna('').astype(str).str.lower()
    df['title'] = df['title'].fillna('').astype(str).str.lower()
    df['date'] = df['date'].astype(str)

    df['full_text'] = df['title'] + " " + df['content']

    embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=False)
    embeddings = np.array(embeddings)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return df, index


def expand_query(query):
    return query + " war conflict military attack"


def smart_filter(results, query):
    words = [w.lower() for w in query.split() if len(w) > 2]

    filtered = []

    for item in results:
        text = (item["title"] + " " + item["content"]).lower()

        
        if not any(k in text for k in war_keywords):
            continue

        
        if any(word in text for word in words):
            filtered.append(item)

    return filtered


def clean_google_link(link):
    try:
        response = requests.get(link, allow_redirects=True, timeout=5)
        return response.url
    except Exception:
        return link


def fetch_rss_news(query):
    query_formatted = query.replace(" ", "+")

    rss_urls = [
        f"https://news.google.com/rss/search?q={query_formatted}+war",
        f"https://news.google.com/rss/search?q={query_formatted}+conflict",
        f"https://news.google.com/rss/search?q={query_formatted}+military",
        "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://www.militarytimes.com/arc/outboundfeeds/rss/",
        "https://www.armytimes.com/arc/outboundfeeds/rss/"
    ]

    rss_data = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            text = entry.title.lower()
            if any(k in text for k in war_keywords):
                rss_data.append({
                    "title": entry.title.lower(),
                    "content": entry.title,
                    "url": clean_google_link(entry.link),
                    "date": entry.published if "published" in entry else "",
                    "full_text": entry.title.lower()
                })

    seen = set()
    unique = []
    for item in rss_data:
        if item["url"] not in seen:
            unique.append(item)
            seen.add(item["url"])

    return unique[:40]




def create_prompt(query, clean_results):
    prompt = f"""
You are a war news analyst.

User Query: {query}

Below are some news articles:

"""
    for i, news in enumerate(clean_results, 1):
        prompt += f"""
Article {i}:
Title: {news['title']}
Content: {news['content']}
Date: {news['date']}
"""

    prompt += """
Task:
- Summarize what is happening
- Explain in simple simple words
-Key insights:
- Keep answer short
"""
    return prompt


def get_llm_response(prompt, groq_client):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def get_top_matches(query, df, model, cross_model, index, groq_client):
    
    original_query = query.lower().strip()
    print("=" * 50)
    print("User Query:", query)
    expanded_query = expand_query(original_query)

    query_embedding = model.encode([expanded_query])
    query_embedding = np.array(query_embedding)
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k=20)
    print("FAISS Scores:", D)
    print("FAISS Indices:", I)

    api_results = []
    for i in I[0]:
        if i < len(df):
            api_results.append({
                "title": str(df.iloc[i]['title']),
                "content": str(df.iloc[i]['content'])[:1000],
                "url": df.iloc[i]['url'],
                "date": df.iloc[i]['date']
            })
    print(f"API Results Retrieved: {len(api_results)}")

    rss_results = fetch_rss_news(original_query)
    print(f"RSS Results Retrieved: {len(rss_results)}")
    print(f"RSS Results Retrieved: {len(rss_results)}")
    api_pairs = [(expanded_query, item["title"] + " " + item["content"]) for item in api_results]
    api_scores = cross_model.predict(api_pairs) if api_pairs else []
    api_ranked = sorted(zip(api_scores, api_results), reverse=True)
    print(f"API Ranked: {len(api_ranked)}")
    rss_pairs = [(expanded_query, item["title"] + " " + item["content"]) for item in rss_results]
    rss_scores = cross_model.predict(rss_pairs) if rss_pairs else []
    rss_ranked = sorted(zip(rss_scores, rss_results), reverse=True)
    print(f"RSS Ranked: {len(rss_ranked)}")
    combined = [item for _, item in (api_ranked[:5] + rss_ranked[:5])]

    filtered = smart_filter(combined, original_query)
    print(f"After Smart Filter: {len(filtered)}")
    if not filtered:
        return [], None

    final_results = filtered[:10]

    clean_results = []
    for item in final_results:
        try:
            parsed = pd.to_datetime(item["date"], errors='coerce')
            date_str = parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else str(item["date"])
        except Exception:
            date_str = str(item["date"])

        clean_results.append({
            "title": item["title"],
            "content": item["content"],
            "url": item["url"],
            "date": date_str
        })

    summary = None
    if len(clean_results) >= 2:
        prompt = create_prompt(original_query, clean_results)
        summary = get_llm_response(prompt, groq_client)
    print(f"After Smart Filter: {len(filtered)}")
    return clean_results, summary



@st.cache_resource(show_spinner=False)
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def build_pipeline(newsapi_key: str, enable_zero_shot: bool, status_container):
    status_container.write("Fetching articles from NewsAPI...")
    df = fetch_news_from_api(newsapi_key)
    print(f"After NewsAPI: {len(df)}")
    if df.empty:
        return df

    status_container.write(f"Fetched {len(df)} raw articles. Removing movie/entertainment noise...")
    df = filter_non_movie(df)
    print(f"After Movie Filter: {len(df)}")
    status_container.write(f"{len(df)} articles remain after movie/entertainment filter.")

    if enable_zero_shot and len(df) > 0:
        status_container.write("Loading zero-shot classifier (facebook/bart-large-mnli)... this can take a while on first run.")
        print("Loading Zero Shot Model...")
        classifier = load_zero_shot_classifier()
        print("Zero Shot Model Loaded Successfully")
        progress = status_container.progress(0.0)
        
        df = apply_zero_shot_filter(df, classifier, progress_cb=progress.progress)
        print(f"After Zero Shot Filter: {len(df)}")
        status_container.write(f"{len(df)} articles remain after zero-shot war-news filter.")

    df = df.reset_index(drop=True)

    status_container.write("Building sentence embeddings + FAISS index...")
    embed_model = load_embedding_model()
    df, index = build_embeddings_index(df, embed_model)
    print(f"FAISS Index Size: {index.ntotal}")
    status_container.write("Loading cross-encoder re-ranker...")
    cross_model = load_cross_encoder()

    status_container.write("Pipeline ready.")
    
    print("=" * 50)
    return df, index, embed_model, cross_model


st.set_page_config(page_title="War News Detection", page_icon="🪖", layout="wide")
st.title("🪖 War News Detection & Retrieval")
st.caption("NewsAPI + zero-shot filtering + FAISS semantic search + RSS + cross-encoder re-ranking + Groq LLM summary")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "Keys are read from a `.env` file (`NEWSAPI_KEY`, `GROQ_API_KEY`). "
        "You can override them here for this session only."
    )
    newsapi_key_input = st.text_input("NewsAPI Key", value=NEWSAPI_KEY, type="password")
    groq_key_input = st.text_input("Groq API Key", value=GROQ_API_KEY, type="password")
    if groq_key_input:
        GROQ_API_KEY = groq_key_input

    enable_zero_shot = st.checkbox(
        "Enable zero-shot war-news filter (matches notebook default, slow)",
        value=True,
        help="This runs the same facebook/bart-large-mnli zero-shot classifier the notebook uses to keep only war/military titles. Turn off only for faster iteration while testing."
    )

    build_clicked = st.button("🔄 Fetch data & build index", type="primary", use_container_width=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if build_clicked:
    if not newsapi_key_input:
        st.sidebar.error("NewsAPI key is required to fetch articles.")
    else:
        status_box = st.status("Running pipeline...", expanded=True)
        result = build_pipeline(newsapi_key_input, enable_zero_shot, status_box)
        if isinstance(result, pd.DataFrame) and result.empty:
            status_box.update(label="No articles fetched — check your NewsAPI key/quota.", state="error")
            st.session_state.pipeline = None
        else:
            df, index, embed_model, cross_model = result
            st.session_state.pipeline = {
                "df": df,
                "index": index,
                "embed_model": embed_model,
                "cross_model": cross_model,
            }
            status_box.update(label=f"Pipeline ready — {len(df)} articles indexed.", state="complete")

st.divider()

if st.session_state.pipeline is None:
    st.info("Use the sidebar to fetch data and build the search index before querying.")
else:
    query = st.text_input("Search query", placeholder="e.g. pakistan and india at war")
    search_clicked = st.button("🔍 Search", type="primary")

    if search_clicked and query.strip():
        if not GROQ_API_KEY:
            st.error("Groq API key is required for the LLM summary. Add it in the sidebar.")
        else:
            with st.spinner("Searching, re-ranking, and summarizing..."):
                groq_client = load_groq_client() if not groq_key_input else Groq(api_key=GROQ_API_KEY)
                pipeline_state = st.session_state.pipeline
                clean_results, summary = get_top_matches(
                    query,
                    pipeline_state["df"],
                    pipeline_state["embed_model"],
                    pipeline_state["cross_model"],
                    pipeline_state["index"],
                    groq_client,
                )

            if not clean_results:
                st.warning("Not Found — no matching war/military news for that query.")
            else:
                if summary:
                    st.subheader("Summary")
                    st.write(summary)

                st.subheader("Sources")
                for i, news in enumerate(clean_results, 1):
                    with st.container(border=True):
                        st.markdown(f"**{i}. {news['title']}**")
                        st.caption(f"Date: {news['date']}")
                        st.markdown(news["url"])
