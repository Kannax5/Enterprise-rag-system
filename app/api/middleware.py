"""
middleware.py — Auth, structured logging, and rate limiting middleware.
"""

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)


# ── Rate limiter (in-memory, per IP) ─────────────────────────────────────────

class _RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self._counts: dict = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.monotonic()
        window_start = now - self.window
        requests = self._counts[client_ip]
        # Prune old entries
        self._counts[client_ip] = [t for t in requests if t > window_start]
        if len(self._counts[client_ip]) >= self.max_requests:
            return False
        self._counts[client_ip].append(now)
        return True


_rate_limiter = _RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


# ── Middleware class ───────────────────────────────────────────────────────────

class RAGMiddleware(BaseHTTPMiddleware):
    """Handles request ID injection, structured logging, and rate limiting."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        # Rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for IP %s (request_id=%s)", client_ip, request_id)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={"X-Request-ID": request_id},
            )

        logger.info(
            "→ %s %s | client=%s | request_id=%s",
            request.method,
            request.url.path,
            client_ip,
            request_id,
        )

        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "← %s %s | status=%d | %.1fms | request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response


# ── Setup helper ──────────────────────────────────────────────────────────────

def setup_middleware(app: FastAPI) -> None:
    """Attach all custom middleware to the FastAPI app."""
    app.add_middleware(RAGMiddleware)
