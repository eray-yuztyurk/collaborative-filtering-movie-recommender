"""
Gradio UI for Movie Recommendation System
"""
import gradio as gr
import pickle
import os
import pandas as pd
from src.data_utils import load_dataset, dataframe_reduction
from src.recommender import (
    create_user_item_matrix, 
    search_item_names_with_keyword,
    find_item_id_using_name,
    find_item_name_using_id,
    user_based_recommendation
)

# Global variables
df = None
user_item_matrix = None
reduced_df = None

# Dump file paths
DUMP_DIR = "dumps"
DF_DUMP = os.path.join(DUMP_DIR, "df.pkl")
REDUCED_DF_DUMP = os.path.join(DUMP_DIR, "reduced_df.pkl")
MATRIX_DUMP = os.path.join(DUMP_DIR, "user_item_matrix.pkl")

def save_dumps():
    """Save processed data to dump files"""
    os.makedirs(DUMP_DIR, exist_ok=True)
    with open(DF_DUMP, 'wb') as f:
        pickle.dump(df, f)
    with open(REDUCED_DF_DUMP, 'wb') as f:
        pickle.dump(reduced_df, f)
    with open(MATRIX_DUMP, 'wb') as f:
        pickle.dump(user_item_matrix, f)
    print("✅ Dumps saved successfully!")

def load_dumps():
    """Load processed data from dump files"""
    global df, user_item_matrix, reduced_df
    with open(DF_DUMP, 'rb') as f:
        df = pickle.load(f)
    with open(REDUCED_DF_DUMP, 'rb') as f:
        reduced_df = pickle.load(f)
    with open(MATRIX_DUMP, 'rb') as f:
        user_item_matrix = pickle.load(f)
    print("✅ Dumps loaded successfully!")

def dumps_exist():
    """Check if all dump files exist"""
    return (os.path.exists(DF_DUMP) and 
            os.path.exists(REDUCED_DF_DUMP) and 
            os.path.exists(MATRIX_DUMP))

def initialize_system():
    """Initialize the recommendation system"""
    global df, user_item_matrix, reduced_df
    
    if dumps_exist():
        try:
            print("📦 Loading from cache...")
            load_dumps()
            return "✅ Ready! (loaded from cache)"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    try:
        print("⏳ Processing data...")
        
        print("  → Loading dataset...")
        df = load_dataset()
        df.columns = ["user_id", "item_id", "rating", "timestamp", "item_name", "genres"]
        
        print("  → Filtering data...")
        reduced_df = dataframe_reduction(df, user_col="user_id", item_col="item_id", 
                                         user_rating_threshold=100, item_rated_threshold=3000)
        
        print("  → Creating matrix...")
        user_item_matrix = create_user_item_matrix(reduced_df, index_col="user_id", 
                                                   columns_col="item_id", values_col="rating")
        
        print("  → Saving to dumps/...")
        save_dumps()
        
        print("✅ Done!")
        return f"✅ Done! {user_item_matrix.shape[0]} users, {user_item_matrix.shape[1]} movies"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def search_movies(keyword):
    """Search for movies by keyword or ID"""
    if df is None:
        return pd.DataFrame({"Error": ["⚠️ Please initialize the system first!"]})
    
    try:
        movie_id = int(keyword)
        movie_data = reduced_df[reduced_df["item_id"] == movie_id]
        if not movie_data.empty:
            movie_name = movie_data["item_name"].values[0]
            return pd.DataFrame({"ID": [movie_id], "Movie Name": [movie_name]})
        else:
            return pd.DataFrame({"Error": [f"Movie ID {movie_id} not found."]})
    except ValueError:
        movies = search_item_names_with_keyword(reduced_df, item_col_name="item_name", 
                                                searched_item_name=keyword)
        if not movies:
            return pd.DataFrame({"Error": ["No movies found."]})
        
        movie_ids = []
        movie_names = []
        for movie_name in movies[:20]:
            movie_id = find_item_id_using_name(reduced_df, item_col_name="item_name", item_name=movie_name)
            movie_ids.append(movie_id)
            movie_names.append(movie_name)
        
        return pd.DataFrame({"ID": movie_ids, "Movie Name": movie_names})

def get_item_based_recommendations(movie_input, top_n):
    """Get item-based recommendations for a movie"""
    if user_item_matrix is None:
        return pd.DataFrame({"Error": ["⚠️ Please initialize the system first!"]})
    
    try:
        try:
            item_id = int(movie_input)
            movie_name = find_item_name_using_id(reduced_df, item_id=item_id)
        except ValueError:
            movie_name = movie_input
            item_id = find_item_id_using_name(reduced_df, item_col_name="item_name", 
                                             item_name=movie_name)
        
        selected_item = user_item_matrix.loc[:, item_id]
        correlated_items = user_item_matrix.corrwith(selected_item).sort_values(ascending=False)[1:top_n+1]
        
        ids = []
        names = []
        scores = []
        
        for rec_item_id, corr_rate in correlated_items.items():
            rec_item_name = find_item_name_using_id(reduced_df, item_id=rec_item_id)
            ids.append(rec_item_id)
            names.append(rec_item_name)
            scores.append(f"{corr_rate*100:.2f}%")
        
        return pd.DataFrame({"ID": ids, "Movie Name": names, "Score": scores})
    
    except Exception as e:
        return pd.DataFrame({"Error": [f"❌ {str(e)}\nUse Movie ID or exact name from search"]})

def get_user_based_recommendations(user_id, top_n):
    """Get user-based recommendations for a user"""
    if user_item_matrix is None:
        return pd.DataFrame({"Error": ["⚠️ Please initialize the system first!"]})
    
    try:
        user_id = int(user_id)
        
        if user_id not in user_item_matrix.index:
            return pd.DataFrame({"Error": [f"❌ User ID {user_id} not found."]})
        
        result_df = user_based_recommendation(user_item_matrix, reduced_df, user_id)
        weighted_scores = result_df.mean(axis=1).sort_values(ascending=False).head(top_n)
        
        ids = []
        names = []
        scores = []
        
        for rec_item_id, score in weighted_scores.items():
            rec_item_name = find_item_name_using_id(reduced_df, item_id=rec_item_id)
            ids.append(rec_item_id)
            names.append(rec_item_name)
            scores.append(f"{score:.2f}")
        
        return pd.DataFrame({"ID": ids, "Movie Name": names, "Score": scores})
    
    except ValueError:
        return pd.DataFrame({"Error": ["❌ Please enter a valid User ID (number)."]})
    except Exception as e:
        return pd.DataFrame({"Error": [f"❌ {str(e)}"]})

def create_gradio_app():
    """Create Gradio interface"""

    with gr.Blocks(title="Movie Recommender System", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 Movie Recommender System")
        
        with gr.Row():
            init_btn = gr.Button("🚀 Initialize System", variant="primary", size="lg")
            init_output = gr.Textbox(label="Status", lines=1, placeholder="Click button to start")
            
        init_btn.click(fn=initialize_system, outputs=init_output)
        
        gr.Markdown("---")
        
        with gr.Tabs():
            with gr.Tab("🔍 Search Movies"):
                gr.Markdown("Search by movie name or ID")

                with gr.Row():
                    search_input = gr.Textbox(label="Keyword or Movie ID", placeholder="e.g., Star Wars, Matrix, or 1234")
                    search_btn = gr.Button("Search", variant="primary")
                with gr.Row():
                    search_output = gr.Dataframe(label="Search Results", interactive=False)
                
                search_btn.click(fn=search_movies, inputs=search_input, outputs=search_output)
            
            with gr.Tab("🎥 Item-Based Recommendations"):
                gr.Markdown("Get recommendations based on a movie")

                with gr.Row():
                    movie_input = gr.Textbox(label="Movie Name or ID", 
                                            placeholder="Enter movie name or ID from search")
                    top_n_item = gr.Slider(minimum=5, maximum=20, value=10, step=1, 
                                        label="Number of Recommendations")
                    item_rec_btn = gr.Button("Get Recommendations", variant="primary")
                
                with gr.Row():
                    item_rec_output = gr.Dataframe(label="Recommendations", interactive=False)
                
                item_rec_btn.click(fn=get_item_based_recommendations, 
                                  inputs=[movie_input, top_n_item], 
                                  outputs=item_rec_output)
            
            with gr.Tab("👤 User-Based Recommendations"):
                gr.Markdown("Get recommendations based on similar users")

                with gr.Row():
                    user_input = gr.Textbox(label="User ID", placeholder="Enter user ID (e.g., 1, 100, 500)")
                    top_n_user = gr.Slider(minimum=5, maximum=20, value=10, step=1, 
                                        label="Number of Recommendations")
                    user_rec_btn = gr.Button("Get Recommendations", variant="primary")

                with gr.Row():
                    user_rec_output = gr.Dataframe(label="Recommendations", interactive=False)
                
                user_rec_btn.click(fn=get_user_based_recommendations, 
                                  inputs=[user_input, top_n_user], 
                                  outputs=user_rec_output)
        
        gr.Markdown(
            """
            ---
            **Tip:** First run processes and saves data to `dumps/` folder. Next runs load from cache instantly.  
            Check terminal for detailed progress.
            """
        )
    
    return app
