import logging
from typing import List, Tuple

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import QueryRequest, QueryResponse, SourceChunk
from app.embeddings.encoder import Encoder
from app.embeddings.faiss_store import FaissStore
from app.generation.guardrails import Guardrails, GuardrailViolation
from app.generation.llm_client import generate
from app.generation.prompt_builder import PromptBuilder
from app.generation.response_validator import ResponseValidator
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

router = APIRouter()

# Singletons
_encoder = Encoder()

_store = FaissStore(
    embedding_dim=_encoder.embedding_dim
)

_store.load()

_retriever = Retriever(
    encoder=_encoder,
    faiss_store=_store,
)

_reranker = Reranker()
_prompt_builder = PromptBuilder()
_guardrails = Guardrails()
_validator = ResponseValidator(encoder=_encoder)


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a RAG query against the knowledge base",
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        print("\n========== NEW QUERY ==========")
        print(f"Query: {request.query}")

        # 1. Pre-check
        print("STEP 1: Guardrails Pre-Check")
        clean_query = _guardrails.pre_check(request.query)
        print(f"Clean Query: {clean_query}")

        # 2. Retrieve
        print("STEP 2: Retrieval")
        raw_results: List[Tuple[dict, float]] = _retriever.retrieve(
            clean_query,
            top_k=request.top_k,
        )
        print(f"Retrieved {len(raw_results)} chunks")

        # 3. Rerank
        print("STEP 3: Reranking")
        ranked = _reranker.rerank(raw_results, top_k=request.top_k)
        print(f"Reranked {len(ranked)} chunks")

        # 4. Build prompt
        print("STEP 4: Prompt Building")
        prompt = _prompt_builder.build(clean_query, ranked)
        print(f"Prompt Length: {len(prompt)}")

        # 5. Generate
        print("STEP 5: LLM Generation")
        try:
            generated = generate(prompt)

        except Exception as e:
            print(f"LLM ERROR: {e}")
            if ranked:
                generated = ranked[0][0].get(
                    "text",
                    "No retrieved context available."
                )
            else:
                generated = (
                    "The LLM is unavailable and no context was retrieved."
                )

        print("LLM Response Generated")

        # 6. Post-check
        print("STEP 6: Post Validation")
        validated_response = _guardrails.post_check(generated)

        # 7. Grounding validation
        print("STEP 7: Grounding Validation")
        validation_result = _validator.validate(
            validated_response,
            ranked,
        )

        # 8. Assemble sources
        print("STEP 8: Building Sources")
        sources = [
            SourceChunk(
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", ""),
                source=chunk.get("source", ""),
                score=round(score, 4),
            )
            for chunk, score in ranked
        ]

        print("SUCCESS")

        return QueryResponse(
            answer=validated_response,
            sources=sources,
            grounded=validation_result["grounded"],
            overlap_score=validation_result["overlap_score"],
        )

    except GuardrailViolation as exc:
        print(f"GUARDRAIL ERROR: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception as exc:
        import traceback
        print("\n========== FULL ERROR ==========")
        traceback.print_exc()
        print("================================\n")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {str(exc)}",
        )