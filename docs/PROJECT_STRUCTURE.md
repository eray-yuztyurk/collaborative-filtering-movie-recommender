# Project Structure

## Directory Layout

```
collaborative-filtering-movie-recommender/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── core/                    # 🔵 Core algorithms & cache
│   │   ├── __init__.py
│   │   ├── recommender.py       # Recommendation algorithms
│   │   └── cache_manager.py     # Data persistence
│   ├── utils/                   # 🟢 Utility functions
│   │   ├── __init__.py
│   │   └── data_utils.py        # Data loading & preprocessing
│   └── ui/                      # 🟡 User interface
│       ├── __init__.py
│       ├── app.py               # Gradio UI definition
│       └── handlers.py          # UI business logic
├── data/
│   ├── movies.csv               # Movie metadata
│   └── ratings.csv              # User ratings
├── dumps/                       # Cached processed data
│   ├── df.pkl
│   ├── reduced_df.pkl
│   └── user_item_matrix.pkl
├── docs/                        # Documentation
├── notebooks/                   # Jupyter notebooks for experiments
├── app.py                       # Main entry point
└── requirements.txt             # Python dependencies
```

## Module Organization

### 🔵 `src/core/` - Core Logic
**Purpose:** Recommendation algorithms and data caching

#### `core/recommender.py`
- `create_user_item_matrix()` - Build user-item matrix
- `search_item_names_with_keyword()` - Search movies
- `find_item_id_using_name()` - ID lookup
- `find_item_name_using_id()` - Name lookup
- `user_based_recommendation()` - User-based filtering

#### `core/cache_manager.py`
- `save_dumps(df, reduced_df, user_item_matrix)` - Save to dumps/
- `load_dumps()` - Load from cache
- `dumps_exist()` - Check cache availability

---

### 🟢 `src/utils/` - Utilities
**Purpose:** Helper functions for data operations

#### `utils/data_utils.py`
- `load_dataset()` - Load MovieLens data
- `dataframe_reduction()` - Filter users/movies by thresholds

---

### 🟡 `src/ui/` - User Interface
**Purpose:** Gradio web interface

#### `ui/app.py` (105 lines)
- `create_gradio_app()` - Build Gradio interface
- Clean component definitions
- Event binding only

#### `ui/handlers.py` (186 lines)
- `AppState` - Global state management
- `initialize_system()` - System initialization
- `search_movies()` - Movie search handler
- `get_item_based_recommendations()` - Item-based logic
- `get_user_based_recommendations()` - User-based logic

---

## Benefits of New Structure

✅ **Clear Separation:** core (algorithms) | utils (helpers) | ui (interface)  
✅ **Modularity:** Each subpackage has single responsibility  
✅ **Scalability:** Easy to add new features in appropriate location  
✅ **Testability:** Can test core logic independently from UI  
✅ **Professional:** Industry-standard project layout

## Import Flow

```
app.py
  └─> src.ui.create_gradio_app()
        └─> src.ui.handlers.*
              ├─> src.core.recommender.*
              ├─> src.core.cache_manager.*
              └─> src.utils.data_utils.*
```

## Quick Access

- **Add new recommendation algorithm** → `src/core/recommender.py`
- **Modify UI layout** → `src/ui/app.py`
- **Change business logic** → `src/ui/handlers.py`
- **Add data preprocessing** → `src/utils/data_utils.py`
- **Modify caching** → `src/core/cache_manager.py`
