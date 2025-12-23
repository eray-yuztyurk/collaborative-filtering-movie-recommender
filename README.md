<h1 align="center">Collaborative-Filtering Movie Recommender</h1>

A collaborative-filtering movie recommender with a clean Gradio web interface. Built to demonstrate how user–item interactions drive personalized movie suggestions using memory-based and model-based approaches.

<table align="center">
  <tr>
    <!-- LEFT: TABLE OF CONTENTS -->
    <td align="left" width="50%" style="vertical-align: top;">
      <h3>📑 Table of Contents</h3>
      <ul>
        <li><a href="#what-this-does">What this does</a></li>
        <li><a href="#key-features">Key features</a></li>
        <li><a href="#project-structure">Project structure</a></li>
        <li><a href="#quick-start">Quick start</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#how-it-works">How it works</a></li>
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
- **Web Interface**: Interactive Gradio UI for movie recommendations
- **Item-Based CF**: Find similar movies based on user rating patterns
- **User-Based CF**: Get recommendations based on similar users
- **Fast Loading**: Caches processed data for instant subsequent loads
- **Clean Architecture**: Modular codebase with separated concerns

---

## Key features
- 🎬 **Search Movies**: Find movies by keyword
- 🎯 **Item-Based Recommendations**: "If you liked X, you'll like Y"
- 👤 **User-Based Recommendations**: Personalized suggestions based on similar users
- ⚡ **Instant Start**: Pre-computed data included - no waiting on first run!
- 📊 **Jupyter Notebooks**: Exploratory analysis and experiments included
- 📖 **Detailed Documentation**: See [docs/RECOMMENDATION_LOGIC.md](docs/RECOMMENDATION_LOGIC.md)

---

## Project structure
```
├── app.py                 # Main application entry point
├── src/
│   ├── data_utils.py     # Data loading and preprocessing
│   ├── recommender.py    # Recommendation algorithms
│   └── gradio_app.py     # Gradio web interface
├── docs/                 # Documentation
│   └── RECOMMENDATION_LOGIC.md  # How recommendations work
├── notebooks/            # Jupyter notebooks for analysis
├── data/                 # MovieLens dataset
├── dumps/               # Pre-computed data (included for instant start)
└── requirements.txt     # Python dependencies
```

---

## Quick start

### 1. Clone and setup
```bash
git clone https://github.com/eray-yuztyurk/collaborative-filtering-movie-recommender.git
cd collaborative-filtering-movie-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download data
```bash
bash download_movielens.sh
```

### 3. Run the web app
```bash
python app.py
```

Open your browser at **http://127.0.0.1:7861**

**Note**: The app loads instantly because pre-computed data is included!

---

## Usage

### Web Interface
1. **Initialize System**: Click the initialize button (loads in ~3 seconds with included cache)
2. **Search Movies**: Find movies by keyword (e.g., "Star Wars", "Matrix")
3. **Get Recommendations**:
   - **Item-Based**: Enter exact movie name from search results
   - **User-Based**: Enter a user ID to get personalized recommendations

**First-time users**: The system includes pre-processed data, so you can explore recommendations immediately without waiting for data processing.

### Notebooks
Explore the analysis notebooks:
```bash
jupyter lab notebooks/
```

### Documentation
For detailed explanation of how recommendations work, see [docs/RECOMMENDATION_LOGIC.md](docs/RECOMMENDATION_LOGIC.md)

---

## How it works

### Memory-Based Collaborative Filtering
- **Item-Based**: Finds similar movies based on user rating patterns
- **User-Based**: Finds similar users and recommends their highly-rated movies

### Data Processing
1. Load MovieLens dataset (movies + ratings)
2. Filter users/items with minimum interaction thresholds
3. Create user-item matrix
4. Calculate correlations/similarities
5. Generate recommendations

### Performance Optimization
- Processed data is cached in `dumps/` folder
- First initialization: ~30 seconds
- Subsequent loads: <3 seconds

---

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
