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


class AlbumSearchResponse(BaseModel):
    query: str
    count: int
    albums: list[AlbumRef] = Field(description="Autocomplete matches from the catalog")


class SharedQuality(BaseModel):
    quality: str
    seed_quote: str
    rec_quote: str


class ExplainRequest(BaseModel):
    seed_review_text: str
    rec_review_text: str
    seed_review_id: str | None = None
    rec_review_id: str | None = None
    n: int = Field(3, ge=1, le=5)


class ExplainResponse(BaseModel):
    qualities: list[SharedQuality]
    seed_review_id: str | None = None
    rec_review_id: str | None = None


class StoreStatus(BaseModel):
    backend: str
    path: str
    size_bytes: int
    seed_albums: int
    recommendation_edges: int


class StatusResponse(BaseModel):
    status: str
    embedding: StoreStatus
