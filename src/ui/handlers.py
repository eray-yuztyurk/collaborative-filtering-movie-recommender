"""
UI Handlers for Gradio Application
Business logic for recommendation operations
"""
import gradio as gr
import pandas as pd
from src.utils.data_utils import load_dataset, dataframe_reduction
from src.utils.constants import (
    USER_RATING_THRESHOLD,
    ITEM_RATED_THRESHOLD,
    MAX_SEARCH_RESULTS,
    MIN_SIMILARITY_THRESHOLD,
    MAX_SIMILAR_MOVIES_TO_SHOW,
    MIN_RATED_MOVIES_FOR_RECOMMENDATIONS,
    USER_SIMILARITY_OVERLAP_THRESHOLDS,
    USER_CORRELATION_THRESHOLDS,
    RATING_COUNT_BREAKPOINTS,
    SIMILARITY_BADGES,
    PROGRESS_STEPS
)
from src.core.recommender import (
    create_user_item_matrix,
    search_item_names_with_keyword,
    find_item_id_using_name,
    find_item_name_using_id,
    user_based_recommendation
)
from src.core.cache_manager import save_dumps, load_dumps, dumps_exist
# New modules
from src.ui.helpers.profile_manager import get_user_profile, get_profile_warning, clear_user_profile
from src.ui.helpers.stats import get_system_info

# Global state
class AppState:
    """Global state container"""
    df = None
    user_item_matrix = None
    reduced_df = None
    user_ratings = {}  # {movie_id: rating} - User's movie ratings for profile

state = AppState()

def initialize_system(progress=gr.Progress()):
    """Initialize the recommendation system"""
    gr.Info("⏳ Initializing system, please wait...")
    
    if dumps_exist():
        try:
            progress(0, desc="📦 Loading from cache...")
            state.df, state.reduced_df, state.user_item_matrix = load_dumps()
            gr.Info("✅ System loaded successfully from cache!")
            return gr.Button(value="✅ System Ready", interactive=False)
        except Exception as e:
            gr.Error(f"❌ Failed to load from cache: {str(e)}")
            return gr.Button(value=f"❌ Error: {str(e)}", interactive=True)
    
    try:
        progress(0, desc="⏳ Initializing...")
        
        progress(PROGRESS_STEPS["loading_dataset"], desc="📂 Loading dataset...")
        state.df = load_dataset()
        state.df.columns = ["user_id", "item_id", "rating", "timestamp", "item_name", "genres"]
        
        progress(PROGRESS_STEPS["filtering_data"], desc="🔍 Filtering data...")
        state.reduced_df = dataframe_reduction(
            state.df, 
            user_col="user_id", 
            item_col="item_id",
            user_rating_threshold=USER_RATING_THRESHOLD, 
            item_rated_threshold=ITEM_RATED_THRESHOLD
        )
        
        progress(PROGRESS_STEPS["creating_matrix"], desc="🔢 Creating matrix...")
        state.user_item_matrix = create_user_item_matrix(
            state.reduced_df,
            index_col="user_id",
            columns_col="item_id",
            values_col="rating"
        )
        
        progress(PROGRESS_STEPS["saving_cache"], desc="💾 Saving to cache...")
        save_dumps(state.df, state.reduced_df, state.user_item_matrix)
        
        progress(PROGRESS_STEPS["complete"], desc="✅ Complete!")
        return gr.Button(
            value=f"✅ Ready! {state.user_item_matrix.shape[0]} users, {state.user_item_matrix.shape[1]} movies",
            interactive=False
        )
        
    except Exception as e:
        return gr.Button(value=f"❌ Error: {str(e)}", interactive=True)

def search_movies(keyword):
    """Search for movies by keyword or ID"""
    if state.df is None:
        gr.Warning("⚠️ Please initialize the system first!")
        return gr.Radio(choices=[], label="⚠️ Please initialize the system first!")
    
    if not keyword or not keyword.strip():
        gr.Warning("⚠️ Please enter a movie name or ID")
        return gr.Radio(choices=[], label="⚠️ Please enter a movie name or ID")
    
    gr.Info(f"🔍 Searching for '{keyword}'...")
    
    try:
        # Try to parse as ID
        movie_id = int(keyword)
        movie_data = state.reduced_df[state.reduced_df["item_id"] == movie_id]
        if not movie_data.empty:
            movie_name = movie_data["item_name"].values[0]
            gr.Info(f"✅ Found movie: {movie_name}")
            choices = [(movie_name, str(movie_id))]
            return gr.Radio(choices=choices, label="Search Results", value=str(movie_id))
        else:
            gr.Warning(f"⚠️ Movie ID {movie_id} not found")
            return gr.Radio(choices=[], label=f"Movie ID {movie_id} not found.")
    except ValueError:
        # Search by name
        movies = search_item_names_with_keyword(
            state.reduced_df,
            item_col_name="item_name",
            searched_item_name=keyword
        )
        if not movies:
            gr.Warning("⚠️ No movies found matching your search")
            return gr.Radio(choices=[], label="No movies found.")
        
        gr.Info(f"🎬 Found {len(movies[:MAX_SEARCH_RESULTS])} movies matching '{keyword}'")
        choices = []
        for movie_name in movies[:MAX_SEARCH_RESULTS]:
            movie_id = find_item_id_using_name(
                state.reduced_df,
                item_col_name="item_name",
                item_name=movie_name
            )
            choices.append((movie_name, str(movie_id)))
        
        return gr.Radio(
            choices=choices,
            label="Search Results (click to select, then Get Recommendations)",
            value=None
        )

def get_item_based_recommendations(movie_input, top_n):
    """Get item-based recommendations for a movie"""
    if state.user_item_matrix is None:
        gr.Warning("⚠️ System not initialized yet")
        return pd.DataFrame({"Error": ["⚠️ Please initialize the system first!"]})
    
    if not movie_input:
        gr.Warning("⚠️ Please select a movie first")
        return pd.DataFrame({"Error": ["⚠️ Please search and select a movie first!"]})
    
    gr.Info("🔍 Calculating similar movies...")
    try:
        # Parse movie ID
        try:
            item_id = int(movie_input)
            movie_name = find_item_name_using_id(state.reduced_df, item_id=item_id)
        except (ValueError, TypeError):
            movie_name = movie_input
            item_id = find_item_id_using_name(
                state.reduced_df,
                item_col_name="item_name",
                item_name=movie_name
            )
        
        # Calculate correlations
        selected_item = state.user_item_matrix.loc[:, item_id]
        correlated_items = state.user_item_matrix.corrwith(selected_item).sort_values(ascending=False)[1:top_n+1]
        
        # Build results
        ids = []
        names = []
        scores = []
        
        for rec_item_id, corr_rate in correlated_items.items():
            rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
            ids.append(rec_item_id)
            names.append(rec_item_name)
            scores.append(f"{corr_rate*100:.2f}%")
        
        selected_name = find_item_name_using_id(state.reduced_df, item_id=item_id)
        gr.Info(f"✅ Found {len(ids)} similar movies to '{selected_name}'")
        return pd.DataFrame({"ID": ids, "Movie Name": names, "Score": scores})
    
    except Exception as e:
        gr.Error(f"❌ Failed to get recommendations: {str(e)}")
        return pd.DataFrame({"Error": [f"❌ {str(e)}\nSelect a movie from search results."]})

def get_user_based_recommendations(user_id, top_n):
    """Get user-based recommendations for a user"""
    if state.user_item_matrix is None:
        gr.Warning("⚠️ System not initialized yet")
        return pd.DataFrame({"Error": ["⚠️ Please initialize the system first!"]})
    
    gr.Info(f"🔍 Finding recommendations for User {user_id}...")
    try:
        user_id = int(user_id)
        if user_id not in state.user_item_matrix.index:
            gr.Warning(f"⚠️ User ID {user_id} not found in the system")
            return pd.DataFrame({"Error": [f"❌ User ID {user_id} not found."]})

        # Get recommendations
        result_df = user_based_recommendation(state.user_item_matrix, state.reduced_df, user_id)
        weighted_scores = result_df.mean(axis=1).sort_values(ascending=False).head(top_n)

        # For each recommended movie, show the raw (unnormalized) predicted rating as 'X.XX / 5'
        ids = []
        names = []
        scores = []
        raw_ratings = []
        for rec_item_id, score in weighted_scores.items():
            rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
            ids.append(rec_item_id)
            names.append(rec_item_name)
            scores.append(f"{score:.2f}")
            # Clamp to [0, 5] for display, but show decimals
            display_rating = max(0, min(score, 5))
            raw_ratings.append(f"{display_rating:.2f} / 5")

        df = pd.DataFrame({
            "ID": ids,
            "Movie Name": names,
            "Score": scores,
            "You would rate this": raw_ratings
        })
        gr.Info(f"✅ Generated {len(df)} recommendations for User {user_id}")
        return df
    
    except ValueError:
        gr.Error("❌ Please enter a valid User ID (number)")
        return pd.DataFrame({"Error": ["❌ Please enter a valid User ID (number)."]})
    except Exception as e:
        gr.Error(f"❌ Failed to generate recommendations: {str(e)}")
        return pd.DataFrame({"Error": [f"❌ {str(e)}"]})

def get_system_info_handler():
    gr.Info("📊 System statistics refreshed")
    return get_system_info(state, USER_RATING_THRESHOLD, ITEM_RATED_THRESHOLD)

# ============================================================================
# NEW USER-BASED RECOMMENDATION FUNCTIONS
# ============================================================================

def get_similarity_badge(similarity_pct):
    """Get badge text and color based on similarity percentage"""
    for level in ["excellent", "great", "good", "fair", "weak"]:
        badge_info = SIMILARITY_BADGES[level]
        if similarity_pct >= badge_info["threshold"]:
            return badge_info["text"], badge_info["color"]
    return SIMILARITY_BADGES["weak"]["text"], SIMILARITY_BADGES["weak"]["color"]

def add_movie_and_show_similar(movie_id, rating):
    """Add movie to profile and show similar movies in component slots"""
    if not movie_id or not rating:
        gr.Warning("⚠️ Please select a movie and rating first")
        outputs = [get_user_profile(state.user_ratings, state.reduced_df), get_profile_warning(state.user_ratings)]
        for _ in range(MAX_SIMILAR_MOVIES_TO_SHOW):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * MAX_SIMILAR_MOVIES_TO_SHOW)
        return outputs
    
    try:
        movie_id = int(movie_id)
        rating = float(rating)
        
        # Add to profile
        state.user_ratings[movie_id] = rating
        movie_name = find_item_name_using_id(state.reduced_df, item_id=movie_id)
        
        # Get similar movies (max from constant)
        selected_item = state.user_item_matrix.loc[:, movie_id]
        correlated_items = state.user_item_matrix.corrwith(selected_item).sort_values(ascending=False)[1:MAX_SIMILAR_MOVIES_TO_SHOW+1]
        
        gr.Info(f"✅ Added '{movie_name}' with {rating}⭐ rating to your profile")
        profile = get_user_profile(state.user_ratings, state.reduced_df)
        profile_warning = get_profile_warning(state.user_ratings)
        
        outputs = [profile, profile_warning]
        
        # Fill up to MAX_SIMILAR_MOVIES_TO_SHOW movie slots
        similar_list = list(correlated_items.items())
        ids = []
        
        for i in range(MAX_SIMILAR_MOVIES_TO_SHOW):
            if i < len(similar_list):
                rec_item_id, corr_rate = similar_list[i]
                rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
                similarity_pct = corr_rate * 100
                
                # Skip if below threshold
                if similarity_pct < MIN_SIMILARITY_THRESHOLD:
                    continue
                
                # Get badge and color from constants
                badge, color = get_similarity_badge(similarity_pct)
                
                # Create progress bar
                progress_width = int(similarity_pct)
                movie_html = f"""<div style='display: flex; align-items: center; justify-content: space-between; gap: 10px;'>
                <span style='flex: 1; font-weight: 700; font-size: 15px;'>{rec_item_name}</span>
                    <div style='display: flex; flex-direction: column; align-items: flex-end; min-width: 140px;'>
                        <div style='display: flex; align-items: center; gap: 5px; margin-bottom: 2px;'>
                            <span style='font-size: 0.75rem; color: #666;'>Match: {similarity_pct:.1f}%</span>
                            <span style='font-size: 0.7rem; padding: 1px 6px; background: {color}; color: white; border-radius: 3px;'>{badge}</span>
                        </div>
                        <div style='width: 100px; height: 4px; background: #e5e7eb; border-radius: 2px; overflow: hidden;'>
                            <div style='width: {progress_width}%; height: 100%; background: {color};'></div>
                        </div>
                    </div>
                </div>"""
                
                outputs.append(gr.Row(visible=True))  # Show row
                outputs.append(movie_html)  # Movie info with HTML
                ids.append(int(rec_item_id))
            else:
                outputs.append(gr.Row(visible=False))  # Hide row
                outputs.append("")  # Empty info
                ids.append(None)
        
        # Add IDs as state values
        outputs.extend(ids)
        
        return outputs
        
    except Exception as e:
        gr.Error(f"❌ Failed to add movie: {str(e)}")
        outputs = [get_user_profile(state.user_ratings, state.reduced_df), get_profile_warning(state.user_ratings)]
        for _ in range(MAX_SIMILAR_MOVIES_TO_SHOW):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * MAX_SIMILAR_MOVIES_TO_SHOW)
        return outputs

def add_similar_movie(movie_id, rating):
    """Add similar movie to profile and refresh similar movies list"""
    if movie_id is None or rating is None:
        gr.Warning("⚠️ No movie selected")
        outputs = ["", get_user_profile(state.user_ratings, state.reduced_df)]
        for _ in range(MAX_SIMILAR_MOVIES_TO_SHOW):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * MAX_SIMILAR_MOVIES_TO_SHOW)
        return outputs
    
    try:
        # Add rating
        state.user_ratings[int(movie_id)] = float(rating)
        
        # Get fresh similar movies from all rated movies
        all_similar = {}
        for rated_movie_id in state.user_ratings.keys():
            selected_item = state.user_item_matrix.loc[:, rated_movie_id]
            correlated_items = state.user_item_matrix.corrwith(selected_item).sort_values(ascending=False)[1:50]
            
            for rec_id, corr in correlated_items.items():
                if rec_id not in state.user_ratings:  # Skip already rated
                    if rec_id not in all_similar:
                        all_similar[rec_id] = corr
                    else:
                        all_similar[rec_id] = max(all_similar[rec_id], corr)
        
        # Sort and get top candidates
        sorted_similar = sorted(all_similar.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out items below threshold and get top movies
        filtered_similar = [(id, corr) for id, corr in sorted_similar if corr * 100 >= MIN_SIMILARITY_THRESHOLD][:MAX_SIMILAR_MOVIES_TO_SHOW]
        
        movie_name = find_item_name_using_id(state.reduced_df, item_id=movie_id)
        gr.Info(f"✅ Rated '{movie_name}' and refreshed recommendations")
        profile = get_user_profile(state.user_ratings, state.reduced_df)
        profile_warning = get_profile_warning(state.user_ratings)
        
        outputs = [profile, profile_warning]
        ids = []
        
        for i in range(MAX_SIMILAR_MOVIES_TO_SHOW):
            if i < len(filtered_similar):
                rec_item_id, corr_rate = filtered_similar[i]
                rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
                similarity_pct = corr_rate * 100
                
                # Get badge and color from constants
                badge, color = get_similarity_badge(similarity_pct)
                
                # Create progress bar
                progress_width = int(similarity_pct)
                movie_html = f"""<div style='display: flex; align-items: center; justify-content: space-between; gap: 10px;'>
                    <span style='flex: 1; font-weight: 700; font-size: 15px;'>{rec_item_name}</span>
                    <div style='display: flex; flex-direction: column; align-items: flex-end; min-width: 140px;'>
                        <div style='display: flex; align-items: center; gap: 5px; margin-bottom: 2px;'>
                            <span style='font-size: 0.75rem; color: #666;'>Match: {similarity_pct:.1f}%</span>
                            <span style='font-size: 0.7rem; padding: 1px 6px; background: {color}; color: white; border-radius: 3px;'>{badge}</span>
                        </div>
                        <div style='width: 100px; height: 4px; background: #e5e7eb; border-radius: 2px; overflow: hidden;'>
                            <div style='width: {progress_width}%; height: 100%; background: {color};'></div>
                        </div>
                    </div>
                </div>"""
                
                outputs.append(gr.Row(visible=True))
                outputs.append(movie_html)
                ids.append(int(rec_item_id))
            else:
                outputs.append(gr.Row(visible=False))
                outputs.append("")
                ids.append(None)
        
        outputs.extend(ids)
        return outputs
        
    except Exception as e:
        gr.Error(f"❌ Failed to rate movie: {str(e)}")
        outputs = [get_user_profile(state.user_ratings, state.reduced_df), get_profile_warning(state.user_ratings)]
        for _ in range(MAX_SIMILAR_MOVIES_TO_SHOW):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * MAX_SIMILAR_MOVIES_TO_SHOW)
        return outputs

def clear_user_profile_handler():
    clear_user_profile(state)
    gr.Info("🗑️ Profile cleared successfully")
    outputs = [get_user_profile(state.user_ratings, state.reduced_df), get_profile_warning(state.user_ratings)]
    outputs.extend([gr.Row(visible=False)] * MAX_SIMILAR_MOVIES_TO_SHOW)
    return outputs

def get_profile_warning_handler():
    return get_profile_warning(state.user_ratings)

def get_user_profile_handler():
    return get_user_profile(state.user_ratings, state.reduced_df)

def generate_personalized_recommendations(top_n=10):
    """Generate recommendations based on user's ratings"""
    if len(state.user_ratings) < MIN_RATED_MOVIES_FOR_RECOMMENDATIONS:
        gr.Warning(f"⚠️ Please rate at least {MIN_RATED_MOVIES_FOR_RECOMMENDATIONS} movies (currently {len(state.user_ratings)})")
        return pd.DataFrame({"Message": [f"⚠️ Please rate at least {MIN_RATED_MOVIES_FOR_RECOMMENDATIONS} movies for better recommendations (currently {len(state.user_ratings)})"]})
    
    gr.Info(f"⏳ Generating personalized recommendations from your {len(state.user_ratings)} ratings...")
    import pandas as pd
    try:
        # Create fake user row
        fake_user_id = -1  # Negative ID for fake user
        user_row = pd.Series(index=state.user_item_matrix.columns, dtype=float)
        for movie_id, rating in state.user_ratings.items():
            if movie_id in user_row.index:
                user_row[movie_id] = rating
        temp_matrix = state.user_item_matrix.copy()
        temp_matrix.loc[fake_user_id] = user_row
        num_ratings = len(state.user_ratings)
        if num_ratings <= RATING_COUNT_BREAKPOINTS["low"]:
            perc_threshold = USER_SIMILARITY_OVERLAP_THRESHOLDS["low"]
            corr_threshold = USER_CORRELATION_THRESHOLDS["low"]
        elif num_ratings <= RATING_COUNT_BREAKPOINTS["medium"]:
            perc_threshold = USER_SIMILARITY_OVERLAP_THRESHOLDS["medium"]
            corr_threshold = USER_CORRELATION_THRESHOLDS["medium"]
        else:
            perc_threshold = USER_SIMILARITY_OVERLAP_THRESHOLDS["high"]
            corr_threshold = USER_CORRELATION_THRESHOLDS["high"]
        from src.core.recommender import user_based_recommendation
        result_df, user_corr_dict = user_based_recommendation(
            temp_matrix,
            state.reduced_df,
            fake_user_id,
            perc_threshold_rated_same_products=perc_threshold,
            corr_threshold=corr_threshold,
            return_corrs=True
        )
        if result_df.empty:
            gr.Warning("⚠️ No recommendations found. Try rating more diverse movies")
            return pd.DataFrame({"Message": ["No recommendations found. Try rating more diverse movies."]})
        weights = [user_corr_dict.get(uid, 0) if pd.notnull(user_corr_dict.get(uid, 0)) else 0 for uid in result_df.columns]
        weights = pd.Series(weights, index=result_df.columns)
        def weighted_avg(row):
            if weights.sum() > 0:
                return (row * weights).sum() / weights.sum()
            else:
                return 0
        weighted_scores = result_df.apply(weighted_avg, axis=1).sort_values(ascending=False).head(top_n * 2)
        print(f"Top 5 weighted scores: {weighted_scores.head().to_dict()}")
        print("="*50)

        ids = []
        names = []
        match_info = []
        raw_ratings = []
        for rec_item_id, score in weighted_scores.items():
            rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
            # Clamp to [0, 5] for display, no decimals
            display_rating = max(0, min(score, 5))
            match_pct = int(round((display_rating / 5) * 100))
            if match_pct < MIN_SIMILARITY_THRESHOLD:
                continue
            badge, _ = get_similarity_badge(match_pct)
            ids.append(rec_item_id)
            names.append(rec_item_name)
            raw_ratings.append(f"{display_rating:.1f} / 5")
            match_info.append(f"{match_pct} % {badge}")
        if len(ids) == 0:
            gr.Warning(f"⚠️ All recommendations were below {MIN_SIMILARITY_THRESHOLD}% match threshold. Try rating more movies")
            return pd.DataFrame({"Message": [f"All recommendations were below {MIN_SIMILARITY_THRESHOLD}% match threshold. Please rate more diverse movies."]})
        df = pd.DataFrame({
            "Movie ID": ids,
            "Movie Title": names,
            "Your Predicted Rating": raw_ratings,
            "Match (%)": match_info
        })
        gr.Info(f"✨ Generated {len(df)} personalized recommendations based on your {len(state.user_ratings)} ratings")
        return df
        
    except Exception as e:
        gr.Error(f"❌ Failed to generate recommendations: {str(e)}")
        return pd.DataFrame({"Error": [f"❌ {str(e)}. Try rating more movies."]})
