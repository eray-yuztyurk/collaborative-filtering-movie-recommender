<h1 align="center">Collaborative-Filtering Movie Recommender</h1>

A compact collaborative-filtering movie recommender I built to demonstrate how user–item interactions (ratings or implicit feedback) can drive personalized movie suggestions.  
The repository is notebook-first and focuses on reproducible experiments, clear examples, and practical paths to extend or productionize the core ideas.

<table align="center">
  <tr>
    <!-- LEFT: TABLE OF CONTENTS -->
    <td align="left" width="50%" style="vertical-align: top;">
      <h3>📑 Table of Contents</h3>
      <ul>
        <li><a href="#what-this-does">What this does</a></li>
        <li><a href="#key-features">Key features</a></li>
        <li><a href="#how-it-works-high-level">How it works (high level)</a></li>
        <li><a href="#quick-start">Quick start</a></li>
        <li><a href="#example-usage">Example usage</a></li>
        <li><a href="#notebooks-and-experiments">Notebooks and experiments</a></li>
        <li><a href="#extending-the-project">Extending the project</a></li>
        <li><a href="#notes-on-data">Notes on data</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#license">License</a></li>
      </ul>
    </td>
    <!-- RIGHT: IMAGE -->
    <td align="center" width="50%">
      <img width="600" height="600" alt="cf-recommender"
           src="https://github.com/user-attachments/assets/372e5ce2-741b-437b-92a1-fd40658c4a1b" />
    </td>
  </tr>
</table>

---

## What this does
- Implements memory-based and model-based collaborative filtering methods using rating datasets.  
- Builds user–item interaction matrices, computes similarity scores, and generates top-N personalized recommendations.  
- Includes a runnable Jupyter notebook that walks through evaluation (RMSE + ranking metrics) and compares CF approaches.

---

## Key features
- Notebook-first workflow covering:
  - User-based and item-based neighborhood CF  
  - Matrix factorization with SVD  
- Evaluation using standard metrics: RMSE, precision@K, recall@K, MAP.  
- Clean, reproducible processing pipelines with minimal dependencies.  
- Notes on transitioning from experimental notebooks to scripts or simple services.

---

## How it works (high level)
1. Load & preprocess user-item interaction data (explicit ratings or implicit events).  
2. Build the sparse user-item matrix and apply weighting/normalization.  
3. Select an approach:
   - **Memory-based**: user–user or item–item similarities + neighborhood aggregation.  
   - **Model-based**: matrix factorization (SVD/ALS) or factorization-machine-style models.  
4. Rank candidate items for each user and evaluate via ranking metrics.  
5. Iterate with hyperparameters, similarity functions, or loss functions.

---

## Quick start

1. Clone the repo
```bash
git clone https://github.com/eray-yuztyurk/collaborative-filtering-movie-recommender.git
cd collaborative-filtering-movie-recommender
```

2. Start Jupyter and open the main notebook
```bash
jupyter lab
# or
jupyter notebook
```
Then open: `collaborative-filtering-movie-recommendation.ipynb`

> Note: this repository is notebook-first — there is no top-level script such as `main.py` in the current tree. Use the notebook as the canonical entry point.

---

## Example usage

The runnable examples live in the notebook. Look at the first cells for data paths and any dependency notes. The notebook demonstrates:
- data loading and cleaning,
- building user-item matrices,
- neighborhood and SVD-based recommenders,
- evaluation and visualizations.

---

## Notebooks and experiments
- collaborative-filtering-movie-recommendation.ipynb — main, runnable example covering the full workflow.

---

## Extending the project
- Add implicit-feedback models (ALS, BPR, LightFM).  
- Integrate item/user metadata for hybrid approaches.  
- Convert notebook cells into scripts for CI or lightweight services.

---

## Notes on data
- Expected format: CSV/Parquet with userId, movieId, rating (or implicit events).  
- Check notebook cells for the exact file locations; update paths locally as needed.

---

## Contributing
Contributions and suggestions welcome. Please open an issue to discuss or submit a PR.

---

## License
See the repository for license details.
