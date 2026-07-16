from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from schemas import (
    AlbumListResponse,
    AlbumRef,
    CompareResponse,
    RecommendResponse,
    Recommendation,
)
from services import embedding, lastfm


@asynccontextmanager
async def lifespan(_app: FastAPI):
    n = embedding.load()
    print(f"Loaded {n} embedding seed albums")
    yield


app = FastAPI(title="Critical Listener", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


@app.get("/recommend/compare", response_model=CompareResponse)
def recommend_compare(
    artist: str = Query(...),
    album: str = Query(...),
    k: int = Query(5, ge=1, le=10),
) -> CompareResponse:
    embedding_rows = embedding.recommend(artist, album, k=k)
    if embedding_rows is None:
        raise HTTPException(status_code=404, detail="Album not in embedding-recs store")

    try:
        _seed, lastfm_rows = lastfm.recommend(artist, album, k=k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return CompareResponse(
        seed=AlbumRef(artist=artist.strip(), album=album.strip()),
        embedding=[Recommendation(**r) for r in embedding_rows],
        lastfm=[Recommendation(**r) for r in lastfm_rows],
    )
