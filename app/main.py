from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    AlbumRef,
    AlbumSearchResponse,
    ExplainRequest,
    ExplainResponse,
    RecommendResponse,
    Recommendation,
    SharedQuality,
    StatusResponse,
    StoreStatus,
)
from app.services import catalog, embedding, explainer, lastfm

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Root .env first; keep older module .env files as fallbacks (no override).
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "lastfm" / ".env", override=False)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    n = embedding.load()
    catalog.load(embedding.list_albums())
    print(f"Loaded {n} embedding seed albums into catalog")
    yield


app = FastAPI(title="Critical Listener", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/about")
def about() -> FileResponse:
    return FileResponse(STATIC_DIR / "about.html")


@app.get("/model")
def model() -> FileResponse:
    return FileResponse(STATIC_DIR / "model.html")


@app.get("/benchmark")
def benchmark() -> FileResponse:
    return FileResponse(STATIC_DIR / "benchmark.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        status="ok",
        embedding=StoreStatus(**embedding.status()),
    )


@app.get("/albums/search", response_model=AlbumSearchResponse)
def albums_search(
    q: str = Query(..., min_length=1, description="Autocomplete query"),
    limit: int = Query(15, ge=1, le=50),
) -> AlbumSearchResponse:
    hits = catalog.search(q, limit=limit)
    return AlbumSearchResponse(
        query=q,
        count=len(hits),
        albums=[AlbumRef(**a) for a in hits],
    )


@app.get("/recommend/embedding", response_model=RecommendResponse)
def recommend_embedding(
    artist: str = Query(...),
    album: str = Query(...),
    k: int = Query(5, ge=1, le=10),
) -> RecommendResponse:
    rows = embedding.recommend(artist, album, k=k)
    if rows is None:
        raise HTTPException(status_code=404, detail="Album not in embedding-recs store")
    return RecommendResponse(
        source="embedding",
        seed=AlbumRef(artist=artist.strip(), album=album.strip()),
        recommendations=[Recommendation(**r) for r in rows],
    )


@app.get("/recommend/lastfm", response_model=RecommendResponse)
def recommend_lastfm(
    artist: str = Query(...),
    album: str = Query(...),
    k: int = Query(5, ge=1, le=10),
) -> RecommendResponse:
    try:
        seed, rows = lastfm.recommend(artist, album, k=k)
    except Exception as exc:  # Last.fm / network / missing key
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RecommendResponse(
        source="lastfm",
        seed=AlbumRef(
            artist=seed.get("artist", artist),
            album=seed.get("album", album),
        ),
        recommendations=[Recommendation(**r) for r in rows],
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_pair(body: ExplainRequest) -> ExplainResponse:
    try:
        result = explainer.explain(
            body.seed_review_text,
            body.rec_review_text,
            seed_review_id=body.seed_review_id,
            rec_review_id=body.rec_review_id,
            n=body.n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExplainResponse(
        qualities=[SharedQuality(**q) for q in result["qualities"]],
        raw_text=result["raw_text"],
        seed_review_id=result["seed_review_id"],
        rec_review_id=result["rec_review_id"],
    )
