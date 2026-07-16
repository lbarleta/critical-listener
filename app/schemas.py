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


class AlbumListResponse(BaseModel):
    count: int
    albums: list[AlbumRef] = Field(description="Seeds available in the embedding-recs store")


class SharedQuality(BaseModel):
    quality: str
    seed_quote: str
    rec_quote: str


class ExplainResponse(BaseModel):
    seed: AlbumRef
    recommendation: AlbumRef
    qualities: list[SharedQuality]
