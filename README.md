# The Critical Listener

The Critical Listener is a review-based music recommender built on full-text album reviews from Pitchfork, Resident Advisor, and CritiqueBrainz. Instead of optimizing for engagement, user listening history, or genre overlap, it recommends albums whose reviews share semantic qualities with a seed album. The goal is critic-informed discovery: suggestions that reflect how journalists describe music, not just what listeners already consume. We pair the recommender with an LLM explainer that grounds each suggestion in quoted review text, and we benchmark the system against an industry-style recommender built on the Last.fm API.

## Folder Structure

```
critical-listener/
├── datasets/                  # Processed tables and recommendation exports
├── preprocessing/             # Data preprocessing
├── model_selection/           # Embedding comparison, final recommender, explainer demo
├── eda/                       # Exploratory analysis of reviews, sources, and recs
├── evaluation/                # Benchmark vs. Last.fm (metrics, plots, notebooks)
└── lastfm-recommender/        # Industry-style baseline recommender (Last.fm API)
```

## Dataset

We collected over 48,000 reviews (1999–2026) covering more than 44,000 albums across rock, electronic, hip hop, pop, experimental, R&B, and related genres.

**Sources**:

- `pitchfork.ipynb`
- `resident_advisor.ipynb`
- `critique_brainz.ipynb`

**Cleaning and masking**: personnel names mentioned in reviews are masked so the model matches on descriptive language rather than shared band members. Albums with multiple reviews are handled at recommendation time by averaging embeddings (see Recommender).

## Recommender

1. **`recommender_1_model_compare.ipynb`** compares TF-IDF, MiniLM, E5-large, and Nomic on cross-source same-album pairs.
2. **`recommender_2_with_name_masking.ipynb`** embeds the masked corpus with Nomic.
3. **`recommender_3_average_embeddings_final.ipynb`** implements the final system.

**How recommendations are generated:**

1. **Build album embeddings.** Embed each review with Nomic, then average and normalize those vectors into one embedding per album.
2. **Score similarity.** For a query album, compute cosine similarity against every other album in the catalog.
3. **Filter and rank.** Drop the query and same-artist albums, then return the top *k* most similar albums.

Nomic was used in zero-shot form (no fine-tuning). Long-context handling mattered because many Pitchfork reviews exceed 512 tokens.

## Explainer

`model_selection/recommender+explainer_in_action.ipynb` runs the recommender on a seed album and calls an LLM (Anthropic) to explain each suggestion. The explainer reads reviews of the seed and recommended albums, identifies shared qualities, and returns side-by-side quotes so the connection is auditable before listening.

Because no ground-truth "similar album" labels exist outside of listening data, the explainer is our primary qualitative validation tool.

## Evaluation

**Baseline** ([`lastfm-recommender/`](lastfm-recommender/)): track-similarity recommendations via the Last.fm API (seed album → seed track → similar track → parent album). See [lastfm_recommender.md](lastfm-recommender/lastfm_recommender.md) for details.

**Benchmark** ([`evaluation/lastfm_bench.ipynb`](evaluation/lastfm_bench.ipynb)): ~45k baseline recs for ~11k seeds, matched against our catalog (~40k albums vs. Last.fm's much larger index). Metrics include:

- **Disagreement** (Jaccard overlap on albums, artists, tags)
- **Baseline recovery** (precision, recall, hit rate)
- **Beyond-accuracy** (popularity bias, diversity, hubness, reciprocity, novelty, serendipity)

Evaluation results are available in [evaluation/lastfm_bench.ipynb](evaluation/lastfm_bench.ipynb).