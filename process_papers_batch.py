import os
import time
from supabase import create_client, Client
from arxiv_processor import process_paper

SUPABASE_URL = "https://tncpegymtqdgnfkwndca.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRuY3BlZ3ltdHFkZ25ma3duZGNhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjcxODkxODEsImV4cCI6MjA0Mjc2NTE4MX0.0Ltp_VvJmRQWqFxf9UJHYDwrbiRDRt8aWoFgcY_PPJA"
GEMINI_API_KEY = "AIzaSyBkIfrdK6-9H-DaqcMQT2lSRXHXzPE0t1Y"

# Gemini API limit
MAX_API_CALLS_PER_DAY = 800
BATCH_SIZE = 30  # Safe batch size per run (adjust as needed)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Count how many papers to process today (avoid over limit)
processed_today = 0

def extract_arxiv_id(doi_url):
    # Example: http://arxiv.org/abs/2503.06935v1 -> 2503.06935v1
    if not doi_url:
        return None
    return doi_url.rstrip('/').split('/')[-1]

def main():
    global processed_today
    # Get unprocessed papers
    response = supabase.table("paper").select("paper_id, doi").is_("processed_papers_json", None).limit(BATCH_SIZE).execute()
    papers = response.data
    if not papers:
        print("No unprocessed papers found.")
        return

    for paper in papers:
        if processed_today >= MAX_API_CALLS_PER_DAY:
            print("API daily limit reached. Stopping batch.")
            break
        arxiv_id = extract_arxiv_id(paper["doi"])
        if not arxiv_id:
            print(f"No arXiv ID found for paper {paper['paper_id']}")
            continue
        try:
            print(f"Processing {arxiv_id}...")
            result = process_paper(arxiv_id, GEMINI_API_KEY)
            if result:
                # Update the processed_papers_json column
                supabase.table("paper").update({"processed_papers_json": result}).eq("paper_id", paper["paper_id"]).execute()
                print(f"Updated paper {paper['paper_id']}.")
                processed_today += result.get('api_calls_used', 1)
            else:
                print(f"Processing failed for {arxiv_id}.")
        except Exception as e:
            print(f"Error processing {arxiv_id}: {e}")
        # Sleep a bit to avoid hammering the API (optional, since arxiv_processor does rate limiting)
        time.sleep(2)

if __name__ == "__main__":
    main() 