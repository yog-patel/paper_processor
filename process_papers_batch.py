import os
import time
from supabase import create_client, Client
from arxiv_processor import process_paper

SUPABASE_URL = "https://tncpegymtqdgnfkwndca.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRuY3BlZ3ltdHFkZ25ma3duZGNhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjcxODkxODEsImV4cCI6MjA0Mjc2NTE4MX0.0Ltp_VvJmRQWqFxf9UJHYDwrbiRDRt8aWoFgcY_PPJA"
GEMINI_API_KEY = "AIzaSyBkIfrdK6-9H-DaqcMQT2lSRXHXzPE0t1Y"

# Gemini API limit
MAX_API_CALLS_PER_DAY = 800
BATCH_SIZE = 50  # Safe batch size per run (adjust as needed)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Count how many papers to process today (avoid over limit)
processed_today = 0

def extract_arxiv_id(doi_url):
    # Example: http://arxiv.org/abs/2503.06935v1 -> 2503.06935v1
    if not doi_url:
        return None
    return doi_url.rstrip('/').split('/')[-1]

def process_papers_batch(papers, batch_type="unprocessed"):
    """Process a batch of papers."""
    global processed_today
    
    for paper in papers:
        if processed_today >= MAX_API_CALLS_PER_DAY:
            print(f"API daily limit reached. Stopping {batch_type} batch.")
            return False
            
        arxiv_id = extract_arxiv_id(paper["doi"])
        if not arxiv_id:
            print(f"No arXiv ID found for paper {paper['paper_id']}")
            continue
            
        try:
            print(f"Processing {arxiv_id} ({batch_type})...")
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
    
    return True

def main():
    global processed_today
    
    # Step 1: Process unprocessed papers
    print("=== Processing unprocessed papers ===")
    response = supabase.table("paper").select("paper_id, doi").is_("processed_papers_json", None).limit(BATCH_SIZE).execute()
    unprocessed_papers = response.data
    
    if unprocessed_papers:
        print(f"Found {len(unprocessed_papers)} unprocessed papers.")
        if not process_papers_batch(unprocessed_papers, "unprocessed"):
            return  # Stop if API limit reached
    else:
        print("No unprocessed papers found.")
    
    # Step 2: Reprocess papers that were processed but used 0 API calls (failed processing)
    print("\n=== Reprocessing papers with 0 API calls ===")
    response = supabase.table("paper").select("paper_id, doi, processed_papers_json").not_.is_("processed_papers_json", None).execute()
    all_processed_papers = response.data
    
    # Filter papers that have been processed but used 0 API calls
    failed_papers = []
    for paper in all_processed_papers:
        if paper["processed_papers_json"]:
            try:
                # Check if the JSON contains api_calls_used field and it's 0
                if isinstance(paper["processed_papers_json"], dict):
                    api_calls = paper["processed_papers_json"].get("api_calls_used", 0)
                else:
                    # If it's stored as a string, try to parse it
                    import json
                    parsed_json = json.loads(paper["processed_papers_json"])
                    api_calls = parsed_json.get("api_calls_used", 0)
                
                if api_calls == 0:
                    failed_papers.append(paper)
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                print(f"Error parsing JSON for paper {paper['paper_id']}: {e}")
                # If we can't parse the JSON, consider it failed
                failed_papers.append(paper)
    
    if failed_papers:
        print(f"Found {len(failed_papers)} papers with 0 API calls that need reprocessing.")
        # Limit to batch size to avoid overwhelming the API
        failed_papers = failed_papers[:BATCH_SIZE]
        process_papers_batch(failed_papers, "failed")
    else:
        print("No papers with 0 API calls found.")

if __name__ == "__main__":
    main() 
