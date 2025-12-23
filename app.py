"""
Movie Recommendation System - Main Application
"""
from src.ui import create_gradio_app

if __name__ == "__main__":
    app = create_gradio_app()
    app.queue()  # Enable queue for progress tracking
    app.launch(share=False, server_name="127.0.0.1", server_port=7861, inbrowser=True)
