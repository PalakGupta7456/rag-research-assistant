FROM python:3.12-slim
RUN useradd --create-home appuser && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[ml,llm,api]"
ENV HF_HOME=/app/hf-cache PYTHONPATH=/app/src:/app PORT=8000
# Bake the embedding model and the cross-encoder into the image
RUN python -c "from rag.models import STEncoder, CrossEncoderReranker; STEncoder(); CrossEncoderReranker()"
COPY api ./api
COPY scripts/entrypoint.sh ./entrypoint.sh
# Corpus: CI copies the published corpus into data/; if it is missing the container builds one from arXiv on first start
COPY data ./data
RUN chmod +x entrypoint.sh && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:${PORT}/health || exit 1
CMD ["./entrypoint.sh"]
