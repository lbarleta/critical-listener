from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from schemas import (
    AlbumListResponse,
    AlbumRef,
    ExplainResponse,
    RecommendResponse,
    Recommendation,
    SharedQuality,
    StatusResponse,
    StoreStatus,
)
from services import embedding, explainer, lastfm


@asynccontextmanager
async def lifespan(_app: FastAPI):
    n = embedding.load()
    print(f"Loaded {n} embedding seed albums")
    yield


app = FastAPI(title="Critical Listener", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        status="ok",
        embedding=StoreStatus(**embedding.status()),
    )


@app.get("/albums", response_model=AlbumListResponse)
def albums() -> AlbumListResponse:
    items = embedding.list_albums()
    return AlbumListResponse(
        count=len(items),
        albums=[AlbumRef(**a) for a in items],
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


@app.get("/explain", response_model=ExplainResponse)
def explain_pair(
    query_artist: str = Query(..., description="Seed album artist"),
    query_album: str = Query(..., description="Seed album title"),
    rec_artist: str = Query(..., description="Recommended album artist"),
    rec_album: str = Query(..., description="Recommended album title"),
    n: int = Query(3, ge=1, le=5, description="Max shared qualities to return"),
) -> ExplainResponse:
    qualities = explainer.explain(
        query_artist=query_artist,
        query_album=query_album,
        rec_artist=rec_artist,
        rec_album=rec_album,
        n=n,
    )
    return ExplainResponse(
        seed=AlbumRef(artist=query_artist.strip(), album=query_album.strip()),
        recommendation=AlbumRef(artist=rec_artist.strip(), album=rec_album.strip()),
        qualities=[SharedQuality(**q) for q in qualities],
    )
