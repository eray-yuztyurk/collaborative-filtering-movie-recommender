"""
Recommendation system functions
"""

import pandas as pd
from surprise import Reader, Dataset, SVD
import unicodedata
try:
    from unidecode import unidecode
    def normalize_str(s):
        return unidecode(str(s)).casefold()
except ImportError:
    def normalize_str(s):
        return unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('utf-8').casefold()

def find_item_name_using_id(dataframe, item_col_name="item_id", item_id=None):
    """Find item name by ID"""
    return dataframe[dataframe[item_col_name] == item_id]["item_name"].values[0]

def find_item_id_using_name(dataframe, item_col_name="item_name", item_name=None):
    """Find item ID by name"""
    # Use normalized comparison for robust matching
    norm_target = normalize_str(item_name)
    for idx, row in dataframe.iterrows():
        if normalize_str(row[item_col_name]) == norm_target:
            return row["item_id"]
    # fallback: try exact match (legacy)
    return dataframe[dataframe[item_col_name] == item_name]["item_id"].values[0]

def search_item_names_with_keyword(dataframe, item_col_name="item_name", searched_item_name=None):
    """Search for items containing a keyword"""
    item_names = []
    norm_keyword = normalize_str(searched_item_name)
    for name in dataframe[item_col_name]:
        if norm_keyword in normalize_str(name) and name not in item_names:
            item_names.append(name)
    return item_names

def create_user_item_matrix(dataframe, index_col="user_id", columns_col="item_id", values_col="rating"):
    """Create user-item matrix for collaborative filtering"""
    user_item_matrix = dataframe.pivot_table(index=index_col, columns=columns_col, values=values_col)
    return user_item_matrix

def item_based_recommendation(user_item_matrix, dataframe, selected_item_id, top_n=10):
    """Generate item-based recommendations"""
    selected_item = user_item_matrix.loc[:,selected_item_id]
    correlated_items = user_item_matrix.corrwith(selected_item).sort_values(ascending=False)[1:top_n+1]
    correlated_item_ids = correlated_items.index.to_list()
    correlated_rates = correlated_items.to_list()
    
    print("*"*100)
    print("'", find_item_name_using_id(dataframe, item_id=selected_item_id), "' - Recommendation List")
    print("*"*100)
    for no, item_id, rate in zip(range(1,top_n+1),correlated_item_ids, correlated_rates):
        item_name = find_item_name_using_id(dataframe, item_id=item_id)
        print(no,"-", item_name, "-"*(70 - len(item_name)-len(str(no))), " - ", round(rate*100,2),"%")
    print("*"*100)

def user_based_recommendation(user_item_matrix, dataframe, selected_user_id, 
                               perc_threshold_rated_same_products=0.7, corr_threshold=0.5, 
                               score_threshold=3, scores_count_to_show=5, return_corrs=False):
    """Generate user-based recommendations. If return_corrs=True, also return dict of userId: corr."""
    print(f"\n=== DEBUG: user_based_recommendation START ===")
    print(f"Selected user ID: {selected_user_id}")
    print(f"Thresholds - perc: {perc_threshold_rated_same_products}, corr: {corr_threshold}")
    # 1. Filter by movies watched by the selected user (columns)
    user_vector = user_item_matrix.loc[selected_user_id]
    rated_items = user_vector[user_vector.notnull()].index.tolist()
    print(f"Step 1: User rated {len(rated_items)} items")
    print(f"Rated item IDs: {rated_items[:10]}...")
    # 2. Find users who watched enough common movies
    candidate_users = user_item_matrix[rated_items]
    overlap_counts = candidate_users.notnull().sum(axis=1)
    count_threshold = len(rated_items) * perc_threshold_rated_same_products
    candidate_users = candidate_users[overlap_counts > count_threshold]
    print(f"Step 2: Need at least {count_threshold:.1f} overlapping items")
    print(f"Step 3: Found {len(candidate_users)} users with enough overlap")
    if len(candidate_users) == 0:
        print("ERROR: No users found with sufficient overlap!")
        print("="*50)
        return (pd.DataFrame(), {}) if return_corrs else pd.DataFrame()
    # 3. Calculate correlation only between the selected user and these users
    correlations = candidate_users.apply(lambda row: row.corr(user_vector), axis=1)
    correlations = correlations.drop(index=selected_user_id, errors='ignore')
    if corr_threshold == 0.0:
        similar_users = correlations[(correlations >= corr_threshold) | (correlations.isna())].sort_values(ascending=False)
    else:
        similar_users = correlations[correlations >= corr_threshold].sort_values(ascending=False)
    print(f"Step 4: After correlation filter (>={corr_threshold}): {len(similar_users)} similar users")
    if len(similar_users) == 0:
        print("ERROR: No users found with sufficient correlation!")
        print("="*50)
        return (pd.DataFrame(), {}) if return_corrs else pd.DataFrame()
    user_corr_dict = similar_users.to_dict()
    list_users_to_filter = list(similar_users.index)
    # 4. Recommend movies watched by these users that the selected user hasn't seen
    final_rec_df = user_item_matrix.loc[list_users_to_filter, :]
    final_rec_excluded_selected_user_items_df = final_rec_df.T
    final_rec_excluded_selected_user_items_df = final_rec_excluded_selected_user_items_df[
        ~final_rec_excluded_selected_user_items_df.index.isin(rated_items)]
    final_rec_excluded_selected_user_items_df = final_rec_excluded_selected_user_items_df.loc[
        ~final_rec_excluded_selected_user_items_df.apply(lambda row: row.isnull().all(), axis=1), :]
    print(f"Step 5: Final recommendations: {final_rec_excluded_selected_user_items_df.shape[0]} items")
    print("="*50)
    if return_corrs:
        return final_rec_excluded_selected_user_items_df, user_corr_dict
    else:
        return final_rec_excluded_selected_user_items_df

def model_based_matrix_factorization(dataframe, index_col="user_id", columns_col="item_id", values_col="rating"):
    """Train SVD model for matrix factorization"""
    scale = Reader(rating_scale=(dataframe[values_col].min(),dataframe[values_col].max()))
    mf_data = Dataset.load_from_df(dataframe[[index_col,columns_col,values_col]], scale)
    
    mf_data_final = mf_data.build_full_trainset()
    svd_model_final = SVD()
    svd_model_final.fit(mf_data_final)
    
    return svd_model_final
