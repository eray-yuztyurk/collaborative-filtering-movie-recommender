"""
Application Constants
All magic numbers, thresholds, and configuration values are defined here.
"""

# ===========================
# DATA FILTERING THRESHOLDS
# ===========================
# Minimum number of ratings a user must have to be included
USER_RATING_THRESHOLD = 150

# Minimum number of ratings a movie must have to be included  
ITEM_RATED_THRESHOLD = 5000


# ===========================
# SEARCH & DISPLAY LIMITS
# ===========================
# Maximum number of search results to display
MAX_SEARCH_RESULTS = 20


# ===========================
# RECOMMENDATION SETTINGS
# ===========================
# Minimum similarity percentage to show recommendations (0-100)
MIN_SIMILARITY_THRESHOLD = 20

# Number of similar movies to show in user-based flow
MAX_SIMILAR_MOVIES_TO_SHOW = 5

# Minimum number of rated movies required for personalized recommendations
MIN_RATED_MOVIES_FOR_RECOMMENDATIONS = 5


# ===========================
# USER-BASED RECOMMENDATION ALGORITHM PARAMETERS
# ===========================
# Percentage of common items required between users (0.0 - 1.0)
# Lower = more relaxed, finds more similar users but less accurate
# Higher = stricter, finds fewer but more similar users
USER_SIMILARITY_OVERLAP_THRESHOLDS = {
    "low": 0.3,      # For users with 5-7 ratings
    "medium": 0.4,   # For users with 8-12 ratings  
    "high": 0.5      # For users with 13+ ratings
}

# Minimum correlation coefficient for similar users (0.0 - 1.0)
# Lower = more users but potentially less similar
# Higher = fewer users but more similar preferences
USER_CORRELATION_THRESHOLDS = {
    "low": 0.0,      # For users with 5-7 ratings (accept all)
    "medium": 0.2,   # For users with 8-12 ratings
    "high": 0.3      # For users with 13+ ratings
}

# Rating count breakpoints for dynamic thresholds
RATING_COUNT_BREAKPOINTS = {
    "low": 7,        # 5-7 ratings
    "medium": 12     # 8-12 ratings, 13+ is high
}


# ===========================
# SIMILARITY BADGES & COLORS
# ===========================
# Badge thresholds (percentage)
SIMILARITY_BADGES = {
    "excellent": {"threshold": 80, "text": "🔥 Excellent Match", "color": "#10b981"},
    "great": {"threshold": 60, "text": "✨ Great Match", "color": "#3b82f6"},
    "good": {"threshold": 40, "text": "👍 Good Match", "color": "#f59e0b"},
    "fair": {"threshold": 20, "text": "👌 Fair Match", "color": "#6b7280"},
    "weak": {"threshold": 0, "text": "😐 Weak Match", "color": "#9ca3af"}
}


# ===========================
# UI PROGRESS INDICATORS
# ===========================
PROGRESS_STEPS = {
    "loading_dataset": 0.2,
    "filtering_data": 0.4,
    "creating_matrix": 0.7,
    "saving_cache": 0.9,
    "complete": 1.0
}
