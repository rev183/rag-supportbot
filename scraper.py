import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin, urlparse

# --- Configuration ---
BASE_URL = "https://www.angelone.in/support"
# Directory to save the scraped articles
OUTPUT_DIR = "sources/angelone-support"

ARTICLE_URL_PATTERN = re.compile(r'^/support/[^/]+(?:/.+)?$')

# --- Helper Functions ---

def sanitize_filename(title):
    """Sanitizes a string to be safe for use as a filename."""
    sanitized = re.sub(r'[^\w\s.-]', '', title)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized[:100]
    return sanitized

def fetch_page(url):
    """Fetches the content of a given URL."""
    try:
        response = requests.get(url, timeout=10) # Added a timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_article_content(html_content):
    """
    Parses HTML and extracts the main article content (FAQs) and title
    based on the provided HTML structure.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # --- Extract Title ---
    title_tag = soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

    faq_section = soup.find('section', class_='sidebar-faq-section')

    article_text = ""
    if faq_section:
        # Find the div containing the list of questions and answers
        list_content_div = faq_section.find('div', class_='list-content')

        if list_content_div:
            # Find all individual FAQ tabs
            faq_tabs = list_content_div.find_all('div', class_='tab')

            if faq_tabs:
                article_text += f"FAQs on: {title}\n\n" # Add a header for the FAQs

                for tab in faq_tabs:
                    # Extract the question from the label
                    question_label = tab.find('label', class_='tab-label')
                    question = question_label.get_text(strip=True) if question_label else "No Question Found"

                    # Extract the answer from the content div within tab-content
                    answer_content_div = tab.select_one('.tab-content .content')
                    answer = answer_content_div.get_text(separator=' ', strip=True) if answer_content_div else "No Answer Found"

                    # Append the question and answer to the article text
                    article_text += f"Q: {question}\n"
                    article_text += f"A: {answer}\n\n"

            else:
                print(f"Warning: Could not find any FAQ tabs within list-content for title '{title}'.")
        else:
            print(f"Warning: Could not find list-content div within sidebar-faq-section for title '{title}'.")
    else:
        print(f"Warning: Could not find sidebar-faq-section for title '{title}'.")

    return title, article_text

def find_article_links(html_content, base_url):
    """Finds potential article links within the HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    article_links = set() # Use a set to store unique URLs

    for link in soup.find_all('a', href=True):
        href = link['href']
        # Resolve relative URLs to absolute URLs
        absolute_url = urljoin(base_url, href)

        # Basic check to ensure it's within the same domain and matches the pattern
        parsed_url = urlparse(absolute_url)
        print(f"Checking link: {absolute_url}")
        if parsed_url.netloc == urlparse(base_url).netloc and ARTICLE_URL_PATTERN.search(parsed_url.path) and 'hindi' not in parsed_url.path:
             article_links.add(absolute_url)


    return list(article_links)

# --- Main Scraping Logic ---
def scrape_angelone_support_pages():
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Starting scraping from {BASE_URL}")

    # Step 1 & 2: Fetch and parse the base support page
    base_page_html = fetch_page(BASE_URL)

    if base_page_html:
        potential_article_urls = find_article_links(base_page_html, BASE_URL)
        print(f"Found {len(potential_article_urls)} potential article links on the base page.")

        scraped_count = 0
        for url in potential_article_urls:
            print(f"Processing: {url}")
            article_html = fetch_page(url)
            inner_potential_article_urls = find_article_links(article_html, BASE_URL)
            for inner_url in inner_potential_article_urls:
                inner_article_html = fetch_page(inner_url)
                if inner_article_html:
                    title, content = extract_article_content(inner_article_html)

                    if content and title != "No Title Found":  # Only save if content and a valid title were extracted
                        sanitized_title = sanitize_filename(title)
                        filename = os.path.join(OUTPUT_DIR, f"{sanitized_title}.txt")

                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                # Include URL and Title at the beginning for context
                                f.write(f"Source URL: {inner_url}\n")
                                f.write(f"Title: {title}\n\n")
                                f.write(content)
                            print(f"Saved: {filename}")
                            scraped_count += 1
                        except IOError as e:
                            print(f"Error saving file {filename}: {e}")
                    else:
                        print(f"Skipping saving for {inner_url} due to no content or title extracted.")
                else:
                    print(f"Skipping {inner_url} due to fetch error.")

        print(f"\nFinished scraping. Successfully saved {scraped_count} articles to '{OUTPUT_DIR}'.")

    else:
        print("Failed to fetch the base support page.")

if __name__ == "__main__":
    print("dummy for testing")
