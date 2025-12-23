"""
Cache Management for Movie Recommendation System
Handles loading and saving of processed data
"""
import pickle
import os

DUMP_DIR = "dumps"
DF_DUMP = os.path.join(DUMP_DIR, "df.pkl")
REDUCED_DF_DUMP = os.path.join(DUMP_DIR, "reduced_df.pkl")
MATRIX_DUMP = os.path.join(DUMP_DIR, "user_item_matrix.pkl")

def save_dumps(df, reduced_df, user_item_matrix):
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
    with open(DF_DUMP, 'rb') as f:
        df = pickle.load(f)
    with open(REDUCED_DF_DUMP, 'rb') as f:
        reduced_df = pickle.load(f)
    with open(MATRIX_DUMP, 'rb') as f:
        user_item_matrix = pickle.load(f)
    print("✅ Dumps loaded successfully!")
    return df, reduced_df, user_item_matrix

def dumps_exist():
    """Check if all dump files exist"""
    return (os.path.exists(DF_DUMP) and 
            os.path.exists(REDUCED_DF_DUMP) and 
            os.path.exists(MATRIX_DUMP))
