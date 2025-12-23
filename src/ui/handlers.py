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

state = AppState()

def initialize_system(progress=gr.Progress()):
    """Initialize the recommendation system"""
    
    if dumps_exist():
        try:
            progress(0, desc="📦 Loading from cache...")
            print("📦 Loading from cache...")
            state.df, state.reduced_df, state.user_item_matrix = load_dumps()
            return gr.Button(value="✅ System Ready (from cache)", interactive=False)
        except Exception as e:
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
        return pd.DataFrame({"Error": ["⚠️ Please initialize the system first!"]})
    
    if not movie_input:
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
        
        return pd.DataFrame({"ID": ids, "Movie Name": names, "Score": scores})
    
    except Exception as e:
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
    
    return "\n".join(info)
