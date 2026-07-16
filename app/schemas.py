from pydantic import BaseModel, Field


class AlbumRef(BaseModel):
    artist: str
    album: str


class Recommendation(BaseModel):
    artist: str
    album: str
    rank: int
    score: float | None = None


class RecommendResponse(BaseModel):
    source: str
    seed: AlbumRef
    recommendations: list[Recommendation]


class CompareResponse(BaseModel):
    seed: AlbumRef
    embedding: list[Recommendation]
    lastfm: list[Recommendation]


class AlbumListResponse(BaseModel):
    count: int
    albums: list[AlbumRef] = Field(description="Seeds available in the embedding-recs store")