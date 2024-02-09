# streamlit_app.py
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup

# Function to scrape news articles from a website
def scrape_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []

    for article in soup.select('section[data-packageid]'):
        title_element = article.select_one('.storyline__headline a')
        link = title_element['href'] if title_element else None
        title = title_element.text.strip() if title_element else None
        content_element = article.select_one('.container-side__text-content')
        content = content_element.text.strip() if content_element else None

        if title and link and content:
            articles.append({'title': title, 'link': link, 'content': content})

    return articles

# Function to cluster news articles based on content similarity
def cluster_articles(articles):
    content = [article['content'] for article in articles]

    if not any(content):
        return articles

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(content)

    if X.shape[1] == 0:
        return articles

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.labels_

    for i, article in enumerate(articles):
        article['cluster'] = labels[i]

    return articles

# Streamlit app
def main():
    st.title("News Clustering App")
    
    # Replace 'your-news-website-url' with the URL of the news website you want to scrape
    news_url = 'https://www.nbcnews.com/'
    articles = scrape_news(news_url)
    clustered_articles = cluster_articles(articles)

    st.subheader("Clustered News Articles")

    for article in clustered_articles:
        st.write(f"**Title:** {article['title']}")
        st.write(f"**Link:** {article['link']}")
        st.write(f"**Content:** {article['content']}")
        st.write(f"**Cluster:** {article['cluster']}")
        st.write("----")

if __name__ == '__main__':
    main()
