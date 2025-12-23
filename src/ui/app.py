"""
Gradio UI for Movie Recommendation System
Clean interface definition - business logic in handlers.py
"""
import gradio as gr
from src.ui.handlers import (
    initialize_system,
    search_movies,
    get_item_based_recommendations,
    get_user_based_recommendations,
    get_system_info
)

def create_gradio_app():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Movie Recommender System", theme=gr.themes.Soft()) as app:
        # Header
        gr.Markdown(
            """
            <div style='text-align: center'>
            <h1>🎬 Movie Recommendation System</h1>
            <p>Discover movies using collaborative filtering based on user ratings and preferences</p>
            </div>
            """
        )
        
        # Initialize button
        init_btn = gr.Button("🚀 Click to Initialize System", variant="primary", size="lg")
        init_btn.click(fn=initialize_system, outputs=init_btn)
        
        # Tabs
        with gr.Tabs():
            # Item-Based Tab
            with gr.Tab("🔍 Item-Based Recommendations"):
                gr.Markdown("**Step 1:** Search for a movie")
                
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Movie Name or ID",
                        placeholder="e.g., Star Wars, Matrix, or 1234"
                    )
                    search_btn = gr.Button("Search", variant="primary")
                
                search_output = gr.Radio(label="Search Results", choices=[], interactive=True)
                
                gr.Markdown("**Step 2:** Select a movie above and get recommendations")
                
                with gr.Row():
                    top_n_item = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Recommendations"
                    )
                    item_rec_btn = gr.Button("Get Recommendations", variant="primary")
                
                item_rec_output = gr.Dataframe(label="Recommendations", interactive=False)
                
                # Event handlers
                search_btn.click(fn=search_movies, inputs=search_input, outputs=search_output)
                item_rec_btn.click(
                    fn=get_item_based_recommendations,
                    inputs=[search_output, top_n_item],
                    outputs=item_rec_output
                )
            
            # User-Based Tab
            with gr.Tab("👤 User-Based Recommendations"):
                gr.Markdown("Get recommendations based on similar users")
                
                with gr.Row():
                    user_input = gr.Textbox(
                        label="User ID",
                        placeholder="Enter user ID (e.g., 1, 100, 500)"
                    )
                    top_n_user = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Recommendations"
                    )
                    user_rec_btn = gr.Button("Get Recommendations", variant="primary")
                
                user_rec_output = gr.Dataframe(label="Recommendations", interactive=False)
                
                # Event handler
                user_rec_btn.click(
                    fn=get_user_based_recommendations,
                    inputs=[user_input, top_n_user],
                    outputs=user_rec_output
                )
            
            # Stats & Info Tab
            with gr.Tab("📊 System Stats & Info"):
                gr.Markdown("View detailed information about the dataset and recommendation system")
                
                info_btn = gr.Button("🔄 Refresh System Info", variant="primary", size="lg")
                info_output = gr.Textbox(
                    label="System Information",
                    lines=25,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )
                
                info_btn.click(fn=get_system_info, outputs=info_output)
        
        # Footer
        gr.Markdown(
            """
            ---
            **Tip:** First run processes and saves data to `dumps/` folder. Next runs load from cache instantly.  
            Check terminal for detailed progress.
            """
        )
    
    return app
