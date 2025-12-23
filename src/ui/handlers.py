"""
UI Handlers for Gradio Application
Business logic for recommendation operations
"""
import gradio as gr
import pandas as pd
from src.utils.data_utils import load_dataset, dataframe_reduction
from src.core.recommender import (
    create_user_item_matrix,
    search_item_names_with_keyword,
    find_item_id_using_name,
    find_item_name_using_id,
    user_based_recommendation
)
from src.core.cache_manager import save_dumps, load_dumps, dumps_exist

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
    
    if dumps_exist():
        try:
            progress(0, desc="📦 Loading from cache...")
            print("📦 Loading from cache...")
            state.df, state.reduced_df, state.user_item_matrix = load_dumps()
            gr.Info("✅ System loaded successfully from cache!")
            return gr.Button(value="✅ System Ready", interactive=False)
        except Exception as e:
            gr.Error(f"❌ Failed to load from cache: {str(e)}")
            return gr.Button(value=f"❌ Error: {str(e)}", interactive=True)
    
    try:
        progress(0, desc="⏳ Initializing...")
        print("⏳ Processing data...")
        
        progress(0.2, desc="📂 Loading dataset...")
        print("  → Loading dataset...")
        state.df = load_dataset()
        state.df.columns = ["user_id", "item_id", "rating", "timestamp", "item_name", "genres"]
        
        progress(0.4, desc="🔍 Filtering data...")
        print("  → Filtering data...")
        state.reduced_df = dataframe_reduction(
            state.df, 
            user_col="user_id", 
            item_col="item_id",
            user_rating_threshold=100, 
            item_rated_threshold=3000
        )
        
        progress(0.7, desc="🔢 Creating matrix...")
        print("  → Creating matrix...")
        state.user_item_matrix = create_user_item_matrix(
            state.reduced_df,
            index_col="user_id",
            columns_col="item_id",
            values_col="rating"
        )
        
        progress(0.9, desc="💾 Saving to cache...")
        print("  → Saving to dumps/...")
        save_dumps(state.df, state.reduced_df, state.user_item_matrix)
        
        progress(1.0, desc="✅ Complete!")
        print("✅ Done!")
        return gr.Button(
            value=f"✅ Ready! {state.user_item_matrix.shape[0]} users, {state.user_item_matrix.shape[1]} movies",
            interactive=False
        )
        
    except Exception as e:
        return gr.Button(value=f"❌ Error: {str(e)}", interactive=True)

def search_movies(keyword):
    """Search for movies by keyword or ID"""
    if state.df is None:
        return gr.Radio(choices=[], label="⚠️ Please initialize the system first!")
    
    try:
        # Try to parse as ID
        movie_id = int(keyword)
        movie_data = state.reduced_df[state.reduced_df["item_id"] == movie_id]
        if not movie_data.empty:
            movie_name = movie_data["item_name"].values[0]
            choices = [(movie_name, str(movie_id))]
            return gr.Radio(choices=choices, label="Search Results", value=str(movie_id))
        else:
            return gr.Radio(choices=[], label=f"Movie ID {movie_id} not found.")
    except ValueError:
        # Search by name
        movies = search_item_names_with_keyword(
            state.reduced_df,
            item_col_name="item_name",
            searched_item_name=keyword
        )
        if not movies:
            return gr.Radio(choices=[], label="No movies found.")
        
        choices = []
        for movie_name in movies[:20]:
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
        return pd.DataFrame({"Error": ["⚠️ Please initialize the system first!"]})
    
    try:
        user_id = int(user_id)
        
        if user_id not in state.user_item_matrix.index:
            return pd.DataFrame({"Error": [f"❌ User ID {user_id} not found."]})
        
        # Get recommendations
        result_df = user_based_recommendation(state.user_item_matrix, state.reduced_df, user_id)
        weighted_scores = result_df.mean(axis=1).sort_values(ascending=False).head(top_n)
        
        # Build results
        ids = []
        names = []
        scores = []
        
        for rec_item_id, score in weighted_scores.items():
            rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
            ids.append(rec_item_id)
            names.append(rec_item_name)
            scores.append(f"{score:.2f}")
        
        return pd.DataFrame({"ID": ids, "Movie Name": names, "Score": scores})
    
    except ValueError:
        return pd.DataFrame({"Error": ["❌ Please enter a valid User ID (number)."]})
    except Exception as e:
        return pd.DataFrame({"Error": [f"❌ {str(e)}"]})

def get_system_info():
    """Get system information and statistics"""
    if state.df is None or state.reduced_df is None or state.user_item_matrix is None:
        gr.Warning("⚠️ System not initialized yet")
        return "⚠️ Please initialize the system first!"
    
    info = []
    info.append("=" * 80)
    info.append("📊 SYSTEM INFORMATION")
    info.append("=" * 80)
    info.append(f"\n🎬 Original Dataset:")
    info.append(f"   • Total ratings: {len(state.df):,}")
    info.append(f"   • Unique users: {state.df['user_id'].nunique():,}")
    info.append(f"   • Unique movies: {state.df['item_id'].nunique():,}")
    info.append(f"   • Date range: {state.df['timestamp'].min()} to {state.df['timestamp'].max()}")
    
    info.append(f"\n🔍 After Filtering (threshold: 100 ratings/user, 3000 ratings/movie):")
    info.append(f"   • Filtered ratings: {len(state.reduced_df):,}")
    info.append(f"   • Active users: {state.reduced_df['user_id'].nunique():,}")
    info.append(f"   • Popular movies: {state.reduced_df['item_id'].nunique():,}")
    
    info.append(f"\n🔢 User-Item Matrix:")
    info.append(f"   • Dimensions: {state.user_item_matrix.shape[0]:,} users × {state.user_item_matrix.shape[1]:,} movies")
    info.append(f"   • Total cells: {state.user_item_matrix.shape[0] * state.user_item_matrix.shape[1]:,}")
    info.append(f"   • Sparsity: {(1 - state.user_item_matrix.notna().sum().sum() / (state.user_item_matrix.shape[0] * state.user_item_matrix.shape[1])) * 100:.2f}%")
    
    info.append(f"\n📈 Statistics:")
    info.append(f"   • Average rating: {state.reduced_df['rating'].mean():.2f}")
    info.append(f"   • Median rating: {state.reduced_df['rating'].median():.1f}")
    info.append(f"   • Rating std dev: {state.reduced_df['rating'].std():.2f}")
    info.append(f"   • Data retention: {(len(state.reduced_df) / len(state.df)) * 100:.2f}%")
    
    info.append("\n" + "=" * 80)
    
    gr.Info("📊 System statistics refreshed")
    return "\n".join(info)

# ============================================================================
# NEW USER-BASED RECOMMENDATION FUNCTIONS
# ============================================================================

def add_movie_and_show_similar(movie_id, rating):
    """Add movie to profile and show similar movies in component slots"""
    if not movie_id or not rating:
        gr.Warning("⚠️ Please select a movie and rating first")
        outputs = [get_user_profile(), get_profile_warning()]
        # Add 3 hidden rows + empty info + 3 None IDs
        for _ in range(3):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * 3)
        return outputs
    
    try:
        movie_id = int(movie_id)
        rating = float(rating)
        
        # Add to profile
        state.user_ratings[movie_id] = rating
        movie_name = find_item_name_using_id(state.reduced_df, item_id=movie_id)
        
        # Get similar movies (max 3)
        selected_item = state.user_item_matrix.loc[:, movie_id]
        correlated_items = state.user_item_matrix.corrwith(selected_item).sort_values(ascending=False)[1:4]
        
        gr.Info(f"✅ Added '{movie_name}' with {rating}⭐ rating to your profile")
        profile = get_user_profile()
        profile_warning = get_profile_warning()
        
        outputs = [profile, profile_warning]
        
        # Fill up to 3 movie slots
        similar_list = list(correlated_items.items())
        ids = []
        
        for i in range(3):
            if i < len(similar_list):
                rec_item_id, corr_rate = similar_list[i]
                rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
                similarity_pct = corr_rate * 100
                
                # Skip if below 20%
                if similarity_pct < 20:
                    continue
                
                # Determine badge and color (5 tiers)
                if similarity_pct >= 80:
                    badge = "🔥 Excellent Match"
                    color = "#10b981"  # Green
                elif similarity_pct >= 60:
                    badge = "✨ Great Match"
                    color = "#3b82f6"  # Blue
                elif similarity_pct >= 40:
                    badge = "👍 Good Match"
                    color = "#f59e0b"  # Orange
                elif similarity_pct >= 20:
                    badge = "👌 Fair Match"
                    color = "#6b7280"  # Gray
                else:
                    badge = "😐 Weak Match"
                    color = "#9ca3af"  # Light gray
                
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
        outputs = [get_user_profile(), get_profile_warning()]
        for _ in range(3):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * 3)
        return outputs

def add_similar_movie(movie_id, rating):
    """Add similar movie to profile and refresh similar movies list"""
    if movie_id is None or rating is None:
        outputs = ["", get_user_profile()]
        for _ in range(3):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * 3)
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
        
        # Filter out items below 20% and get top 3
        filtered_similar = [(id, corr) for id, corr in sorted_similar if corr * 100 >= 20][:3]
        
        movie_name = find_item_name_using_id(state.reduced_df, item_id=movie_id)
        gr.Info(f"✅ Rated '{movie_name}' and refreshed recommendations")
        profile = get_user_profile()
        profile_warning = get_profile_warning()
        
        outputs = [profile, profile_warning]
        ids = []
        
        for i in range(3):
            if i < len(filtered_similar):
                rec_item_id, corr_rate = filtered_similar[i]
                rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
                similarity_pct = corr_rate * 100
                
                # Determine badge and color (5 tiers)
                if similarity_pct >= 80:
                    badge = "🔥 Excellent Match"
                    color = "#10b981"  # Green
                elif similarity_pct >= 60:
                    badge = "✨ Great Match"
                    color = "#3b82f6"  # Blue
                elif similarity_pct >= 40:
                    badge = "👍 Good Match"
                    color = "#f59e0b"  # Orange
                elif similarity_pct >= 20:
                    badge = "👌 Fair Match"
                    color = "#6b7280"  # Gray
                else:
                    badge = "😐 Weak Match"
                    color = "#9ca3af"  # Light gray
                
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
        outputs = [get_user_profile(), get_profile_warning()]
        for _ in range(3):
            outputs.append(gr.Row(visible=False))
            outputs.append("")
        outputs.extend([None] * 3)
        return outputs

def clear_user_profile():
    """Clear all user ratings and hide similar movies"""
    state.user_ratings = {}
    gr.Info("🗑️ Profile cleared successfully")
    outputs = [get_user_profile(), get_profile_warning()]
    # Hide all 3 movie rows
    outputs.extend([gr.Row(visible=False)] * 3)
    return outputs

def get_profile_warning():
    """Get dynamic warning message based on number of rated movies"""
    count = len(state.user_ratings)
    if count >= 5:
        return f"<p style='color: #10b981; font-weight: 600; margin-bottom: 10px;'>✅ Great! You have {count} rated movies. Ready for recommendations!</p>"
    else:
        return f"<p style='color: #f59e0b; margin-bottom: 10px;'>⚠️ You need at least 5 rated movies to get personalized recommendations (currently: {count})</p>"

def get_user_profile():
    """Get current user profile as DataFrame"""
    if not state.user_ratings:
        return pd.DataFrame({"Message": ["No ratings yet. Search and add movies!"]})
    
    ids = []
    names = []
    ratings = []
    
    for movie_id, rating in state.user_ratings.items():
        ids.append(movie_id)
        names.append(find_item_name_using_id(state.reduced_df, item_id=movie_id))
        ratings.append(str(int(rating) * "⭐"))
    
    return pd.DataFrame({"ID": ids, "Movie": names, "Your Rating": ratings})

def generate_personalized_recommendations(top_n=10):
    """Generate recommendations based on user's ratings"""
    if len(state.user_ratings) < 5:
        gr.Warning(f"⚠️ Please rate at least 5 movies (currently {len(state.user_ratings)})")
        return pd.DataFrame({"Message": [f"⚠️ Please rate at least 5 movies for better recommendations (currently {len(state.user_ratings)})"]})
    
    try:
        # Create fake user row
        fake_user_id = -1  # Negative ID for fake user
        
        # Create new row with user ratings - ensure it matches matrix structure
        user_row = pd.Series(index=state.user_item_matrix.columns, dtype=float)
        for movie_id, rating in state.user_ratings.items():
            if movie_id in user_row.index:
                user_row[movie_id] = rating
        
        # Add to matrix temporarily - use proper DataFrame creation
        temp_matrix = state.user_item_matrix.copy()
        temp_matrix.loc[fake_user_id] = user_row
        
        # Debug: Print fake user row to verify it's correct
        print(f"\n=== DEBUG: Fake User Info ===")
        print(f"Total ratings added: {user_row.notna().sum()}")
        print(f"Rated movie IDs: {user_row[user_row.notna()].index.tolist()}")
        print(f"Temp matrix shape: {temp_matrix.shape}")
        print(f"Fake user row in matrix: {temp_matrix.loc[fake_user_id].notna().sum()} non-null values")
        print("="*50)
        
        # Dynamic threshold based on number of ratings
        num_ratings = len(state.user_ratings)
        if num_ratings <= 7:
            perc_threshold = 0.3  # 5-7 films: need 30% overlap
            corr_threshold = 0.0  # No correlation filter - too few data points!
        elif num_ratings <= 12:
            perc_threshold = 0.4  # 8-12 films: need 40% overlap
            corr_threshold = 0.2  # Low correlation threshold
        else:
            perc_threshold = 0.5  # 13+ films: need 50% overlap
            corr_threshold = 0.3  # Normal correlation threshold
        
        print(f"\n=== DEBUG: Thresholds ==")
        print(f"Number of ratings: {num_ratings}")
        print(f"Percentage threshold: {perc_threshold} ({perc_threshold*100}%)")
        print(f"Correlation threshold: {corr_threshold}")
        print(f"IMPORTANT: If corr_threshold is NOT 0.0, please restart the app!")
        print("="*50)
        
        # Use existing user_based_recommendation function with dynamic thresholds
        result_df = user_based_recommendation(
            temp_matrix, 
            state.reduced_df, 
            fake_user_id,
            perc_threshold_rated_same_products=perc_threshold,
            corr_threshold=corr_threshold
        )
        
        print(f"\n=== DEBUG: Recommendation Result ===")
        print(f"Result DataFrame shape: {result_df.shape}")
        print(f"Result DataFrame empty: {result_df.empty}")
        if not result_df.empty:
            print(f"Number of similar users found: {result_df.shape[1]}")
            print(f"Number of recommended items: {result_df.shape[0]}")
        print("="*50)
        
        if result_df.empty:
            return pd.DataFrame({"Message": ["No recommendations found. Try rating more diverse movies."]})
        
        # Calculate weighted scores and get top recommendations (limit early for performance)
        weighted_scores = result_df.mean(axis=1).sort_values(ascending=False).head(top_n * 2)  # Get 2x to allow filtering
        
        # Normalize scores to 0-100 range
        min_score = weighted_scores.min()
        max_score = weighted_scores.max()
        score_range = max_score - min_score
        
        ids = []
        names = []
        match_info = []
        
        for rec_item_id, score in weighted_scores.items():
            rec_item_name = find_item_name_using_id(state.reduced_df, item_id=rec_item_id)
            
            # Normalize to 0-100 scale
            if score_range > 0:
                similarity_pct = ((score - min_score) / score_range) * 100
            else:
                similarity_pct = 100
            
            # Skip recommendations below 20%
            if similarity_pct < 20:
                continue
            
            # Determine badge and color (5 tiers)
            if similarity_pct >= 80:
                badge = "🔥 Excellent Match"
                color = "#10b981"  # Green
            elif similarity_pct >= 60:
                badge = "✨ Great Match"
                color = "#3b82f6"  # Blue
            elif similarity_pct >= 40:
                badge = "👍 Good Match"
                color = "#f59e0b"  # Orange
            elif similarity_pct >= 20:
                badge = "👌 Fair Match"
                color = "#6b7280"  # Gray
            else:
                badge = "😐 Weak Match"
                color = "#9ca3af"  # Light gray
            
            ids.append(rec_item_id)
            names.append(rec_item_name)
            match_info.append(f"{similarity_pct:.1f}% {badge}")
        
        gr.Info(f"✨ Generated {len(ids)} personalized recommendations based on your {len(state.user_ratings)} ratings")
        return pd.DataFrame({"ID": ids, "Recommended Movie": names, "Match": match_info})
        
    except Exception as e:
        gr.Error(f"❌ Failed to generate recommendations: {str(e)}")
        return pd.DataFrame({"Error": [f"❌ {str(e)}. Try rating more movies."]})
