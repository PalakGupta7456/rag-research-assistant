"""Settings for the research RAG system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CORPUS = DATA_DIR / "papers.parquet"
SITE_DATA_DIR = ROOT / "docs" / "data"

ARXIV_API = "https://export.arxiv.org/api/query"
CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV", "stat.ML", "cs.IR"]
FETCH_PER_RUN = 600
MAX_PAPERS = 4000

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # same weights as Xenova/all-MiniLM-L6-v2 used in the browser
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-sonnet-5-5"
RRF_K = 60
