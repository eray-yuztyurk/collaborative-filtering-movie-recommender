"""
Profile management helpers for user ratings and profile display
"""
import pandas as pd
from src.core.recommender import find_item_name_using_id
from src.utils.constants import MIN_RATED_MOVIES_FOR_RECOMMENDATIONS

def get_profile_warning(user_ratings):
    count = len(user_ratings)
    if count >= MIN_RATED_MOVIES_FOR_RECOMMENDATIONS:
        return f"<p style='color: #10b981; font-weight: 600; margin-bottom: 10px;'>✅ Great! You have {count} rated movies. Ready for recommendations!</p>"
    else:
        return f"<p style='color: #f59e0b; margin-bottom: 10px;'>⚠️ You need at least {MIN_RATED_MOVIES_FOR_RECOMMENDATIONS} rated movies to get personalized recommendations (currently: {count})</p>"

def get_user_profile(user_ratings, reduced_df):
    if not user_ratings:
        return pd.DataFrame({"Message": ["No ratings yet. Search and add movies!"]})
    ids = []
    names = []
    ratings = []
    for movie_id, rating in user_ratings.items():
        ids.append(movie_id)
        names.append(find_item_name_using_id(reduced_df, item_id=movie_id))
        ratings.append(str(int(rating) * "⭐"))
    return pd.DataFrame({"ID": ids, "Movie": names, "Your Rating": ratings})

def clear_user_profile(state):
    state.user_ratings = {}
    return state
