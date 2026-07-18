from pydantic import BaseModel, Field


class AlbumRef(BaseModel):
    artist: str
    album: str


class AlbumStats(BaseModel):
    """Album identity plus catalog stats used in Benchmark."""

    artist: str
    album: str
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    review_count: int | None = None
    listeners: int | None = None


class Recommendation(AlbumStats):
    rank: int
    score: float | None = None


class RecommendResponse(BaseModel):
    source: str
    seed: AlbumStats
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
    """Explain a seed → recommendation pair by looking up review texts."""

    seed: AlbumRef
    recommendation: AlbumRef
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
