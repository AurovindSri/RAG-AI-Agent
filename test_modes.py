import requests
import json
import csv
from pathlib import Path

RAG_URL = "http://localhost:8080/chat_rag/"
NON_RAG_URL = "http://localhost:8080/chat_direct/"
INPUT_FILE = "toy_questions.txt"  # One question per line
OUTPUT_FILE = "rag_comparison_results.csv"
USER_ID = "benchmark_user"

def read_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def get_response(api_url, question):
    try:
        response = requests.post(api_url, json={"user_id": USER_ID, "message": question})
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"ERROR: {response.status_code} - {response.text}"
    except Exception as e:
        return f"EXCEPTION: {str(e)}"

def run_comparison():
    questions = read_questions(INPUT_FILE)
    results = []

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Testing: {question}")
        rag_resp = get_response(RAG_URL, question)
        non_rag_resp = get_response(NON_RAG_URL, question)
        results.append({
            "question": question,
            "rag_response": rag_resp,
            "non_rag_response": non_rag_resp
        })

    # Save results to CSV
    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "rag_response", "non_rag_response"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_comparison()
