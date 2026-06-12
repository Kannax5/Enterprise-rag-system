import requests
import json
import sys
import os

API_URL = "http://localhost:8000/api/v1/query"
CHAT_HISTORY = []

COLORS = {
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "gray":   "\033[90m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(text, color):
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"

def print_banner():
    os.system("cls" if os.name == "nt" else "clear")
    print(c("=" * 60, "cyan"))
    print(c("   DocIntel — Enterprise Document Intelligence", "bold"))
    print(c("   Ask questions about your company documents", "gray"))
    print(c("=" * 60, "cyan"))
    print(c("  Type 'exit' or 'quit' to stop", "gray"))
    print(c("  Type 'clear' to clear the screen", "gray"))
    print(c("  Type 'history' to see past queries", "gray"))
    print(c("=" * 60, "cyan"))
    print()

def query_rag(question: str) -> dict:
    try:
        response = requests.post(
            API_URL,
            json={"query": question, "top_k": 5},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the server running? Run: uvicorn app.main:app --reload"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The LLM might be overloaded."}
    except Exception as e:
        return {"error": str(e)}

def print_response(data: dict):
    if "error" in data:
        print(c(f"\n  Error: {data['error']}", "red"))
        return

    answer = data.get("answer", "No answer returned.")
    sources = data.get("sources", [])
    grounded = data.get("grounded", False)
    retrieval_score = data.get("retrieval_score", 0)

    print()
    print(c("  Bot : ", "green") + answer)
    print()

    if sources:
        print(c("  Sources:", "gray"))
        for s in sources[:3]:
            score = s.get("score", 0)
            source = s.get("source", "unknown")
            snippet = s.get("text", "")[:60].replace("\n", " ")
            print(c(f"    [{score:.2f}] {os.path.basename(source)} — {snippet}...", "gray"))

    status_color = "green" if grounded else "yellow"
    grounded_text = "Grounded" if grounded else "Low confidence"
    print()
    print(c(f"  {grounded_text} | Retrieval score: {retrieval_score:.2f}", status_color))
    print()

def print_history():
    if not CHAT_HISTORY:
        print(c("\n  No history yet.\n", "gray"))
        return
    print()
    for i, entry in enumerate(CHAT_HISTORY, 1):
        print(c(f"  [{i}] You : {entry['query']}", "cyan"))
        answer_preview = entry['answer'][:80].replace("\n", " ")
        print(c(f"       Bot : {answer_preview}...", "gray"))
    print()

def main():
    print_banner()

    while True:
        try:
            user_input = input(c("  You : ", "cyan")).strip()
        except (KeyboardInterrupt, EOFError):
            print(c("\n\n  Goodbye!\n", "yellow"))
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(c("\n  Goodbye!\n", "yellow"))
            break

        if user_input.lower() == "clear":
            print_banner()
            continue

        if user_input.lower() == "history":
            print_history()
            continue

        print(c("  Thinking...", "gray"), end="\r")

        data = query_rag(user_input)
        print_response(data)

        if "answer" in data:
            CHAT_HISTORY.append({
                "query": user_input,
                "answer": data["answer"]
            })

if __name__ == "__main__":
    main()