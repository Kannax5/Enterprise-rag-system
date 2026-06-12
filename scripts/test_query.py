"""
test_query.py — Manual end-to-end query test against the running service.

Usage:
    python scripts/test_query.py --query "What is the refund policy?" [--url URL]
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_URL = "http://localhost:8000/api/v1/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual end-to-end RAG query test.")
    parser.add_argument("--query", required=True, help="Question to ask the RAG system.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Query endpoint URL.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--prompt-version", default="v2", help="Prompt template version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = json.dumps(
        {
            "query": args.query,
            "top_k": args.top_k,
            "prompt_version": args.prompt_version,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"\n-> Sending query to {args.url}\n  Query: {args.query}\n")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP Error {exc.code}: {exc.read().decode('utf-8')}")
        sys.exit(1)
    except Exception as exc:
        print(f"Request failed: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("ANSWER:")
    print(body.get("answer", "(no answer)"))
    print("=" * 60)
    print(f"Grounded: {body.get('grounded')} | Overlap score: {body.get('overlap_score')}")
    print(f"\nSources ({len(body.get('sources', []))}):")
    for src in body.get("sources", []):
        print(f"  [{src['score']:.4f}] {src['source']}  (chunk_id: {src['chunk_id']})")
        print(f"         {src['text'][:120]}…")


if __name__ == "__main__":
    main()
