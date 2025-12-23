# How This Movie Recommender Works

## Project Overview

This is a movie recommendation system that suggests movies based on what users have rated in the past. Think of it like this: if you and I both loved the same 10 movies, and I watched another movie that I really enjoyed, there's a good chance you'll like it too.

The system has two main ways to recommend movies:

1. **Item-based**: "People who liked Movie A also liked Movie B"
2. **User-based**: "Users similar to you enjoyed these movies"

I built a simple web interface with Gradio where you can search for movies and get recommendations instantly. 

**Quick Start Note**: This repository includes pre-processed data dumps, so you can run the app immediately without waiting for data processing. The system loads in under 3 seconds on first run!

---

## Recommendation Approach

### Why Collaborative Filtering?

I chose collaborative filtering because it's practical and works well for movie recommendations. Here's why:

- **No need for movie metadata**: I don't need to know if a movie is action or comedy. The system learns from what people actually watch and rate.
- **Discovers hidden patterns**: It can find connections between movies that aren't obvious. Maybe sci-fi fans also love certain documentaries.
- **Proven approach**: Netflix, Amazon, and Spotify all use variations of this.

### Item-Based vs User-Based

**Item-Based Recommendations** ("If you liked X, try Y")
- Looks at which movies are rated similarly by users
- More stable over time (movie similarities don't change much)
- Faster for large user bases
- Works great when users want "more like this"

**User-Based Recommendations** ("Users like you enjoyed these")
- Finds users with similar taste to you
- More personalized and can surprise you with different genres
- Better for discovering new content
- Needs more computation but gives fresher results

I implemented both because they serve different purposes. Item-based is great for "related movies," while user-based is better for "personalized discovery."

---

## Data Flow

Here's what happens when you use the system:

### 1. Data Loading
```
MovieLens Dataset → Load movies.csv + ratings.csv → Merge them
```
We start with two CSV files: one with movie names and one with user ratings.

### 2. Data Preprocessing
```
Remove users who rated < 30 movies
Remove movies with < 1000 ratings
→ This reduces noise and makes calculations faster
```

Why these thresholds? 
- A user who only rated 2 movies doesn't give us enough information
- A movie with only 10 ratings might just be someone's personal favorite, not generally good

### 3. Matrix Creation
```
Create a User-Item Matrix:
          Movie1  Movie2  Movie3
User1       5.0     NaN     4.0
User2       NaN     3.0     5.0
User3       4.0     4.5     NaN
```

This matrix is the heart of the system. Empty cells (NaN) mean the user hasn't rated that movie yet. Our job is to predict what those empty cells should be.

### 4. Recommendation Generation

**For Item-Based:**
1. Pick a movie (e.g., "The Matrix")
2. Look at all users who rated it
3. Find other movies these users also rated similarly
4. Calculate correlation scores
5. Return top N movies with highest correlation

**For User-Based:**
1. Pick a user (e.g., User ID 123)
2. Find other users who rated movies similarly
3. Look at what these similar users liked
4. Filter out movies User 123 already saw
5. Return top N recommendations

### 5. Caching
```
First run: Load from dumps/ folder → Ready in 3 seconds ✨
No dumps: Process data → Save to dumps/ → Takes ~30 seconds
```

**Important**: This repository includes pre-computed dumps in the `dumps/` folder. This means:
- ✅ You can start using the app immediately after cloning
- ✅ No waiting for data processing on first run
- ✅ Perfect for demos, testing, and exploring the system

The dumps contain the processed MovieLens data with:
- Filtered users (who rated at least 30 movies)
- Filtered movies (with at least 1000 ratings)
- Pre-computed user-item matrix

If you delete the dumps folder, the system will recreate them automatically (takes about 30 seconds).

---

## Limitations and Trade-offs

### Cold Start Problem
**The Issue**: New movies with few ratings won't get recommended, and new users with few ratings won't get good recommendations.

**Why it happens**: The system needs data to work. If a movie has only 5 ratings, we can't confidently say it's similar to other movies.

**What I did**: Set minimum thresholds (30 ratings per user, 1000 ratings per movie). This helps, but it means the system won't work well for brand new content.

### Scalability
**The Issue**: As the user-item matrix grows, calculating correlations gets slower.

**Why it happens**: We're computing correlations between thousands of items. That's a lot of math.

**What I did**: Cache everything. After the first run, the system is fast. For production, you'd need more sophisticated caching and maybe approximate methods.

### The "Popularity Bias"
**The Issue**: The system tends to recommend popular movies more often.

**Why it happens**: Popular movies have more ratings, so they appear more frequently in similarity calculations.

**What I did**: Nothing yet, but I'm aware of it. The reduction thresholds help a bit by ensuring we only work with movies that have enough data.

### No Content Understanding
**The Issue**: The system doesn't know anything about movie content (actors, directors, plot).

**Why it happens**: That's just how collaborative filtering works—it only looks at ratings.

**Trade-off**: This is actually a feature sometimes. It can recommend movies from unexpected genres. But it also means it can't explain WHY it recommends something.

---

## Possible Improvements

### 1. Hybrid Approach
Combine collaborative filtering with content-based filtering:
```python
# Could add:
- Movie genres
- Director/actor information
- User demographic data
```
This would help with cold start problems and give better explanations.

### 2. Matrix Factorization (Already Implemented, Not Used in UI)
I coded SVD (Singular Value Decomposition) but didn't add it to the UI yet. It's more accurate but harder to explain to users. Could add a toggle for "Advanced Recommendations."

### 3. Time-Based Recommendations
Currently, old and new ratings are treated equally. Could weight recent ratings more heavily:
```python
# Idea:
recent_ratings = ratings_from_last_6_months * 1.2
old_ratings = ratings_older_than_2_years * 0.8
```

### 4. Explanation System
Add a feature that tells users WHY a movie was recommended:
- "Because you rated The Matrix 5 stars"
- "Because users who liked Inception also liked this"

### 5. Real-Time Updates
Right now, data is cached. For production, you'd want:
- Incremental updates when new ratings come in
- A/B testing different recommendation algorithms
- Feedback loop to improve recommendations based on what users actually watch

### 6. Better Performance
For a production system:
- Use sparse matrices (scipy.sparse) to save memory
- Pre-compute top-N recommendations for all users nightly
- Use approximate nearest neighbor algorithms for user similarity
- Consider Redis or similar for caching

---

## Final Thoughts

This project shows that you don't need complex deep learning to build something useful. Collaborative filtering is simple, effective, and explainable. 

The code is structured to be easy to understand and extend. If you want to add features, you know where to look:
- `src/data_utils.py` for data processing
- `src/recommender.py` for recommendation logic
- `src/gradio_app.py` for the UI

The main lesson I learned: **good caching is as important as good algorithms**. Users don't care if your algorithm is perfect if it takes 30 seconds to load. The dump system made this project actually usable.

---

*Have questions or suggestions? Feel free to open an issue or contribute!*
