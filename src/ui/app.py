"""
Gradio UI for Movie Recommendation System
Clean interface definition - business logic in handlers.py
"""
import gradio as gr
from src.ui.handlers import (
    initialize_system,
    search_movies,
    get_item_based_recommendations,
    get_system_info_handler,
    # New user-based functions
    add_movie_and_show_similar,
    add_similar_movie,
    clear_user_profile_handler,
    generate_personalized_recommendations
)

def create_gradio_app():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Movie Recommender System", theme=gr.themes.Soft(), css="""
        .movie-title-text { flex: 1 1 50% !important; min-width: 250px !important; }
        .rating-btn { flex: 0 0 auto !important; min-width: 60px !important; max-width: 110px !important; }
    """) as app:
        # Header
        gr.HTML("""
            <div style='text-align: center; margin: 20px 0;'>
                <h1 style='margin: 0; padding: 0;'>🎬 Movie Recommendation System</h1>
                <p style='margin: 10px 0 0 0; color: #666;'>Discover movies using collaborative filtering based on user ratings and preferences</p>
            </div>
        """)
        
        # Initialize button
        init_btn = gr.Button("🚀 Click to Initialize System", variant="primary", size="lg")
        init_btn.click(fn=initialize_system, outputs=init_btn)
        
        # Tabs
        with gr.Tabs():
            # Item-Based Tab
            with gr.Tab("🔍 Item-Based Recommendations"):
                gr.Markdown("""
**What is this?**  
Find movies similar to a movie you already like. The system looks at rating patterns and suggests movies that people who liked your chosen movie also enjoyed.

**When should you use it?**  
- If you want “more like this” suggestions.  
- When you have a favorite movie and want to discover similar ones.

**How to use:**  
1. Search for a movie by name or ID.  
2. Select your movie from the search results.  
3. Choose how many recommendations you want.  
4. Click “Get Recommendations” to see movies similar to your selection.

**Why choose this?**  
Item-based recommendations are fast and stable. They’re great for finding movies with similar vibes or genres.
                """)
                
                # Step 1: Search
                gr.Markdown("<div style='background: linear-gradient(90deg, #3b82f6 0%, transparent 100%); height: 3px; margin: 25px 0 15px 0;'></div>")
                gr.Markdown("<h3 style='margin: 10px 0;'>📍 Step 1: Search for a movie</h3>")
                
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Movie Name or ID",
                        placeholder="e.g., Star Wars, Matrix, or 1234",
                        scale=3
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                
                search_output = gr.Radio(label="Search Results", choices=[], interactive=True)
                
                # Step 2: Get Recommendations
                gr.Markdown("<div style='background: linear-gradient(90deg, #10b981 0%, transparent 100%); height: 3px; margin: 25px 0 15px 0;'></div>")
                gr.Markdown("<h3 style='margin: 10px 0;'>📍 Step 2: Select a movie and get recommendations</h3>")
                
                with gr.Row():
                    top_n_item = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Recommendations"
                    )
                    item_rec_btn = gr.Button("✨ Get Recommendations", variant="primary")
                
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
                gr.Markdown("""
**What is this?**  
Get personalized movie suggestions based on your own taste profile. The system finds users with similar preferences and recommends movies they liked but you haven’t seen yet.

**When should you use it?**  
- If you want highly personalized recommendations.  
- When you’re open to discovering new genres or surprises.

**How to use:**  
1. Search for your favorite movies and rate them (1-5 stars).  
2. Rate at least 5 movies to build your taste profile.  
3. Review your rated movies in your profile.  
4. Click “Get My Recommendations” to see movies tailored to your taste.

**Why choose this?**  
User-based recommendations are more personal and can introduce you to unexpected favorites. The more movies you rate, the better your suggestions.
                """)
                
                # Step 1: Search for favorite movie
                gr.Markdown("<div style='background: linear-gradient(90deg, #3b82f6 0%, transparent 100%); height: 3px; margin: 25px 0 15px 0;'></div>")
                gr.Markdown("<h3 style='margin: 10px 0;'>📍 Step 1: Search and select your favorite movie</h3>")
                with gr.Row():
                    search_input_user = gr.Textbox(label="Movie Name", placeholder="e.g., Inception, Pulp Fiction", scale=3)
                    search_btn_user = gr.Button("🔍 Search", variant="primary", scale=1)
                

                with gr.Row(visible=False) as search_results_row:
                    search_results_user = gr.Radio(label="Search Results", choices=[], interactive=True)
                with gr.Row(visible=False) as rating_row:
                    gr.Markdown("***Your Rating for selected Movie ??***")
                    rate_btn1 = gr.Button("⭐", size="sm", variant="secondary")
                    rate_btn2 = gr.Button("⭐⭐", size="sm", variant="secondary")
                    rate_btn3 = gr.Button("⭐⭐⭐", size="sm", variant="secondary")
                    rate_btn4 = gr.Button("⭐⭐⭐⭐", size="sm", variant="secondary")
                    rate_btn5 = gr.Button("⭐⭐⭐⭐⭐", size="sm", variant="primary")
                
                # Step 2: Similar movies with direct rating buttons
                gr.Markdown("<div style='background: linear-gradient(90deg, #10b981 0%, transparent 100%); height: 3px; margin: 25px 0 15px 0;'></div>")
                gr.Markdown("<h3 style='margin: 10px 0;'>📍 Step 2: Rate similar movies (click a star to instantly add to profile)</h3>")
                
                # Similar movie slots (3 movies)
                similar_movies = []
                for i in range(3):
                    with gr.Row(visible=False, elem_classes="movie-rating-row") as movie_row:
                        movie_info = gr.HTML("", elem_classes="movie-title-text")
                        btn1 = gr.Button("⭐", size="sm", elem_classes="rating-btn")
                        btn2 = gr.Button("⭐⭐", size="sm", elem_classes="rating-btn")
                        btn3 = gr.Button("⭐⭐⭐", size="sm", elem_classes="rating-btn")
                        btn4 = gr.Button("⭐⭐⭐⭐", size="sm", elem_classes="rating-btn")
                        btn5 = gr.Button("⭐⭐⭐⭐⭐", size="sm", elem_classes="rating-btn")
                    similar_movies.append((movie_row, movie_info, btn1, btn2, btn3, btn4, btn5))
                
                # Step 3: Your profile
                gr.Markdown("<div style='background: linear-gradient(90deg, #f59e0b 0%, transparent 100%); height: 3px; margin: 25px 0 15px 0;'></div>")
                gr.Markdown("<h3 style='margin: 10px 0;'>📍 Step 3: Your rated movies</h3>")
                with gr.Row():
                    profile_output = gr.Dataframe(interactive=False, scale=7)
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary", scale=1, min_width=80)

                # Move warning to Step 2
                profile_warning = gr.HTML("<p style='color: #f59e0b; margin-bottom: 10px;'>⚠️ You need at least 5 rated movies to get personalized recommendations</p>")
                
                # Step 4: Get recommendations
                gr.Markdown("<div style='background: linear-gradient(90deg, #8b5cf6 0%, transparent 100%); height: 4px; margin: 30px 0 15px 0;'></div>")
                gr.Markdown("<h2 style='margin: 10px 0; color: #8b5cf6;'>🎯 FINAL STEP: Generate Your Personalized Recommendations</h2>")
                with gr.Row():
                    top_n_personalized = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of Recommendations")
                    rec_btn = gr.Button("✨ Get My Recommendations", variant="primary", size="lg")
                
                personalized_output = gr.Dataframe(label="Your Personalized Recommendations", interactive=False)
                
                # Hidden state to store similar movie IDs
                similar_ids = [gr.State(None) for _ in range(3)]
                
                # Collect all outputs for similar movies display
                similar_outputs = [profile_output, profile_warning]
                for row, info, _, _, _, _, _ in similar_movies:
                    similar_outputs.append(row)
                    similar_outputs.append(info)
                for state_id in similar_ids:
                    similar_outputs.append(state_id)
                
                # Event handlers
                # Show search results after search
                def show_search_results(*args):
                    return gr.update(visible=True)
                search_btn_user.click(fn=search_movies, inputs=search_input_user, outputs=search_results_user)
                search_btn_user.click(fn=show_search_results, inputs=[], outputs=[search_results_row])

                # Show rating section only when a movie is selected in radio
                def show_rating_section(selected):
                    return gr.update(visible=bool(selected))
                search_results_user.change(fn=show_rating_section, inputs=search_results_user, outputs=rating_row)
                
                # Track selected rating
                selected_rating = gr.State(None)
                def set_selected_rating(rating):
                    return rating
                rate_btn1.click(fn=set_selected_rating, inputs=[], outputs=selected_rating)
                rate_btn2.click(fn=set_selected_rating, inputs=[], outputs=selected_rating)
                rate_btn3.click(fn=set_selected_rating, inputs=[], outputs=selected_rating)
                rate_btn4.click(fn=set_selected_rating, inputs=[], outputs=selected_rating)
                rate_btn5.click(fn=set_selected_rating, inputs=[], outputs=selected_rating)

                # Update button styles based on selected rating
                def update_rating_buttons(rating):
                    return (
                        gr.update(variant="primary" if rating==1 else "secondary"),
                        gr.update(variant="primary" if rating==2 else "secondary"),
                        gr.update(variant="primary" if rating==3 else "secondary"),
                        gr.update(variant="primary" if rating==4 else "secondary"),
                        gr.update(variant="primary" if rating==5 else "secondary"),
                    )
                rate_btn1.click(fn=update_rating_buttons, inputs=[gr.Number(value=1)], outputs=[rate_btn1, rate_btn2, rate_btn3, rate_btn4, rate_btn5])
                rate_btn2.click(fn=update_rating_buttons, inputs=[gr.Number(value=2)], outputs=[rate_btn1, rate_btn2, rate_btn3, rate_btn4, rate_btn5])
                rate_btn3.click(fn=update_rating_buttons, inputs=[gr.Number(value=3)], outputs=[rate_btn1, rate_btn2, rate_btn3, rate_btn4, rate_btn5])
                rate_btn4.click(fn=update_rating_buttons, inputs=[gr.Number(value=4)], outputs=[rate_btn1, rate_btn2, rate_btn3, rate_btn4, rate_btn5])
                rate_btn5.click(fn=update_rating_buttons, inputs=[gr.Number(value=5)], outputs=[rate_btn1, rate_btn2, rate_btn3, rate_btn4, rate_btn5])

                # Connect rating logic to recommendation
                rate_btn1.click(fn=add_movie_and_show_similar, inputs=[search_results_user, gr.Number(value=1, visible=False)], outputs=similar_outputs)
                rate_btn2.click(fn=add_movie_and_show_similar, inputs=[search_results_user, gr.Number(value=2, visible=False)], outputs=similar_outputs)
                rate_btn3.click(fn=add_movie_and_show_similar, inputs=[search_results_user, gr.Number(value=3, visible=False)], outputs=similar_outputs)
                rate_btn4.click(fn=add_movie_and_show_similar, inputs=[search_results_user, gr.Number(value=4, visible=False)], outputs=similar_outputs)
                rate_btn5.click(fn=add_movie_and_show_similar, inputs=[search_results_user, gr.Number(value=5, visible=False)], outputs=similar_outputs)
                for i, (row, info, btn1, btn2, btn3, btn4, btn5) in enumerate(similar_movies):
                    for rating, btn in enumerate([btn1, btn2, btn3, btn4, btn5], 1):
                        btn.click(
                            fn=lambda id_val=similar_ids[i], r=rating: add_similar_movie(id_val, r),
                            inputs=[similar_ids[i]],
                            outputs=similar_outputs
                        )
                
                clear_btn.click(fn=clear_user_profile_handler, outputs=[profile_output, profile_warning] + [row for row, *_ in similar_movies])
                rec_btn.click(fn=generate_personalized_recommendations, inputs=top_n_personalized, outputs=personalized_output)
            
            # Stats & Info Tab
            with gr.Tab("📊 System Stats & Info"):
                gr.Markdown("""
**What is this?**  
View technical details about the dataset and how the recommendation system works.

**When should you use it?**  
- If you’re curious about the data or system performance.  
- For transparency and understanding how recommendations are generated.

**How to use:**  
- Click “Refresh System Info” to see up-to-date stats and system details.
                """)
                
                gr.Markdown("<div style='background: linear-gradient(90deg, #8b5cf6 0%, transparent 100%); height: 3px; margin: 25px 0 15px 0;'></div>")
                gr.Markdown("<h3 style='margin: 10px 0;'>📊 Dataset Statistics & System Information</h3>")
                
                info_btn = gr.Button("🔄 Refresh System Info", variant="primary", size="lg")
                info_output = gr.Textbox(
                    label="System Information",
                    lines=25,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )
                
                info_btn.click(fn=get_system_info_handler, outputs=info_output)
        
        # Footer
        gr.Markdown(
            """
            ---
            **Tip:** First run processes and saves data to `dumps/` folder. Next runs load from cache instantly.  
            Check terminal for detailed progress.
            """
        )
    
    return app
