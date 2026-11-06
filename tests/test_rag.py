import json

import numpy as np
import pytest

from rag import build, ingest
from rag.answer import citation_precision, extractive
from rag.evaluate import evaluate
from rag.models import HashingEncoder
from rag.retrieval import BM25, Retriever, rrf, tokenize
from rag.synthetic import generate

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2611.01234v2</id><published>2026-11-05T17:59:00Z</published>
<title>Grounded   Generation
 with Citations</title><summary>We study retrieval augmented generation.</summary>
<author><name>Ada Lovelace</name></author><arxiv:primary_category term="cs.CL"/></entry></feed>"""


def test_parse_atom_strips_version_and_whitespace():
    df = ingest.parse_atom(ATOM)
    assert df.loc[0, "id"] == "2611.01234"
    assert df.loc[0, "title"] == "Grounded Generation with Citations"
    assert df.loc[0, "category"] == "cs.CL"


def test_merge_dedupes_and_caps():
    a = generate(50)
    corpus, added = ingest.merge(None, a, cap=40)
    assert len(corpus) == 40 and added == 50
    corpus2, added2 = ingest.merge(corpus, a, cap=40)
    assert added2 == 10 and len(corpus2) == 40   # only the 10 that had been capped out are "new"


def test_bm25_prefers_rare_matching_terms():
    bm = BM25().fit(["the cat sat", "conformal prediction sets", "prediction of cats"])
    s = bm.scores("conformal prediction")
    assert s.argmax() == 1
    assert "the" not in tokenize("The model")


def test_rrf_rewards_agreement():
    a = np.array([0, 1, 2])
    b = np.array([1, 0, 2])
    fused = rrf([a, b])
    assert fused[2] < fused[0] and fused[2] < fused[1]


def test_quantisation_keeps_cosine():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(50, 384)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    q = build.quantize(e).astype(np.float32) / 127
    assert np.abs(q @ q.T - e @ e.T).max() < 0.06


def test_evaluation_beats_random():
    papers = generate(120)
    res = evaluate(papers, HashingEncoder(), n_queries=60)["results"]
    for mode in ("bm25", "dense", "hybrid"):
        assert res[mode]["hit@10"] > 10 / 120 * 3  # far above chance


def test_extractive_answer_is_cited():
    papers = generate(30).to_dict("records")
    ans = extractive("how to reduce hallucination with retrieval and grounding", papers[:5], HashingEncoder(), min_sim=0.0)
    assert ans["sentences"] and citation_precision(ans["answer"], 5) == 1.0
    assert citation_precision("claim [7]", 5) == 0.0


def test_build_publishes_browser_index(tmp_path):
    out = tmp_path / "site"
    stats = build.run(lightweight=True, new=generate(80), corpus_path=tmp_path / "p.parquet", out_dir=out, eval_queries=40)
    papers = json.loads((out / "papers.json").read_text())
    emb = np.frombuffer((out / "embeddings.bin").read_bytes(), dtype=np.int8)
    assert len(papers) == stats["n_papers"] == 80
    assert emb.size == 80 * stats["dim"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    from fastapi.testclient import TestClient

    from api import main

    os.environ["LIGHTWEIGHT_MODELS"] = "1"
    d = tmp_path_factory.mktemp("c")
    build.run(lightweight=True, new=generate(90), corpus_path=d / "p.parquet", out_dir=d / "site", eval_queries=20)
    main.STATE["corpus_path"] = d / "p.parquet"
    with TestClient(main.app) as c:
        yield c


def test_api_search_and_ask(client):
    r = client.get("/search", params={"q": "conformal prediction coverage", "k": 3}).json()
    assert len(r["results"]) == 3 and r["results"][0]["category"] == "stat.ML"
    a = client.post("/ask", json={"question": "How is coverage guaranteed in conformal prediction?"}).json()
    assert a["answer"]["mode"] == "extractive" and a["checks"]["citation_precision"] == 1.0
    assert Retriever  # imported for API parity
