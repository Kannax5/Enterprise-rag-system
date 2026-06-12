import logging
import re
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock In-house LLM Server")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mock_llm")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.2


@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = request.prompt
    logger.info("Received prompt of length %d", len(prompt))

    # Try to extract context from the prompt
    context_match = re.search(r"CONTEXT:(.*?)\n\n---", prompt, re.DOTALL)
    if not context_match:
        context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)

    if context_match:
        context_text = context_match.group(1).strip()
        # Filter out source tags
        lines = []
        source_doc = None
        for line in context_text.splitlines():
            if line.startswith("[Source:"):
                # Extract first source doc for citation
                if not source_doc:
                    source_doc = line.replace("[Source:", "").replace("]", "").strip()
            elif line.strip() != "---" and line.strip():
                lines.append(line.strip())
        
        cleaned_text = " ".join(lines)
        if cleaned_text and "no relevant context found" not in cleaned_text.lower():
            # Extract first few sentences
            sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
            response_text = " ".join(sentences[:3])
            citation = f" [Source: {source_doc}]" if source_doc else ""
            answer = f"Based on the corporate guidelines: {response_text}{citation}"
        else:
            answer = "The available documents do not contain enough information to answer this question."
    else:
        answer = "No context could be parsed from the prompt. The available documents do not contain enough information to answer this question."

    logger.info("Generated answer: %s", answer)
    return {
        "generated_text": answer,
        "usage": {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(answer) // 4,
            "total_tokens": (len(prompt) + len(answer)) // 4
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
