# The Critical Listener

*An Embedding-Based Music Recommender Built on Music Criticism*

James McNally, Leonardo Barleta

Music recommendation algorithms on platforms like Spotify optimize for engagement, usually suggesting albums that match what a listener already knows rather than surfacing connections across genres or taste boundaries. The Critical Listener uses full-text reviews from Pitchfork, Resident Advisor, and CritiqueBrainz instead: albums are matched by the semantic language critics use to describe them, not by listening history or genre overlap. The goal is critic-informed discovery for listeners willing to go a bit further than their usual taste profile. We pair the recommender with an LLM explainer that grounds each suggestion in quoted review text, and benchmark the system against an industry-style Last.fm recommender.

[Executive Summary](<001 Executive Summary.pdf>) · [Slides](<002 Slides.pdf>) · [Project Video](<003 Project Video.mp4>)

## Folder Structure

```
critical-listener/
├── datasets/                  # Processed tables and recommendation exports
├── preprocessing/             # Data preprocessing
├── model_selection/           # Embedding comparison, final recommender, explainer demo
├── eda/                       # Exploratory analysis of reviews, sources, and recs
├── evaluation/                # Benchmark vs. Last.fm (metrics, plots, notebooks)
└── lastfm/                    # Industry-style baseline recommender (Last.fm API)
```

## Dataset

48,639 reviews (1999–2026) covering 44,717 albums across rock, electronic, hip hop, pop, experimental, R&B, and related genres.

**Sources**:

- `pitchfork.ipynb`
- `resident_advisor.ipynb`
- `critique_brainz.ipynb`

**Cleaning and masking**: personnel names mentioned in reviews are masked so the model matches on descriptive language rather than shared band members. Albums with multiple reviews are handled at recommendation time by averaging embeddings (see Recommender).

## Recommender

1. **`recommender_1_model_compare.ipynb`** compares TF-IDF, MiniLM, E5-large, and Nomic on a cross-source retrieval task (given a review of album X by reviewer A, retrieve the same album's review by reviewer B).
2. **`recommender_2_with_name_masking.ipynb`** embeds the masked corpus with Nomic.
3. **`recommender_3_average_embeddings_final.ipynb`** implements the final system.

**How recommendations are generated:**

1. **Build album embeddings.** Embed each review with Nomic, then average and normalize those vectors into one embedding per album.
2. **Score similarity.** For a query album, compute cosine similarity against every other album in the catalog.
3. **Filter and rank.** Drop the query and same-artist albums, then return the top *k* most similar albums.

Nomic was used in zero-shot form (no fine-tuning). Long-context handling mattered because many Pitchfork reviews exceed 512 tokens; Nomic's 8,192-token limit outperformed the shorter-context models and the TF-IDF baseline.

## Explainer

`model_selection/recommender+explainer_in_action.ipynb` runs the recommender on a seed album and calls a state-of-the-art LLM to explain each suggestion. The explainer reads reviews of the seed and recommended albums, identifies up to three shared qualities grounded in verbatim quotes, and returns them side by side so the connection is auditable before listening.

Because no ground-truth "similar album" labels exist outside of listening data, the explainer is our primary qualitative validation tool. A worked example using Drake's *Take Care* appears in the [executive summary](<001 Executive Summary.pdf>) appendix.

## Evaluation

**Baseline** ([`lastfm/`](lastfm/)): track-similarity recommendations via the Last.fm API (seed album → seed track → similar track → parent album). See [lastfm/README.md](lastfm/README.md) for details.

**Benchmark** ([`evaluation/lastfm_bench.ipynb`](evaluation/lastfm_bench.ipynb)): top-5 recommendations for ~11,000 seed albums (~45,000 slots total), matched against our catalog (~44,000 albums vs. Last.fm's ~2.2 million). We compute 30+ metrics at album, artist, and tag level, grouped into themes such as disagreement, popularity bias, diversity, repetition, novelty, and serendipity.

At a high level: recommendations diverge substantially from Last.fm (low Jaccard overlap on albums and artists) while still sharing some tag-level alignment. The model shows less popularity bias — recommendations average ~225k listeners per album vs. ~293k for Last.fm — and greater diversity in artists and tags within a top-5 list. It does repeat albums more often than the baseline, which is expected given the smaller catalog.

Full results and plots are in [evaluation/lastfm_bench.ipynb](evaluation/lastfm_bench.ipynb).
