"""
Movie Recommendation System - Main Application
Run this file to start the Gradio web interface
"""
from src.gradio_app import create_gradio_app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7861, inbrowser=True)
