"""
Recommendation system functions
"""
import pandas as pd
from surprise import Reader, Dataset, SVD

def find_item_name_using_id(dataframe, item_col_name="item_id", item_id=None):
    """Find item name by ID"""
    return dataframe[dataframe[item_col_name] == item_id]["item_name"].values[0]

def find_item_id_using_name(dataframe, item_col_name="item_name", item_name=None):
    """Find item ID by name"""
    return dataframe[dataframe[item_col_name] == item_name]["item_id"].values[0]

def search_item_names_with_keyword(dataframe, item_col_name="item_name", searched_item_name=None):
    """Search for items containing a keyword"""
    item_names = list()
    [item_names.append(name) for name in dataframe[item_col_name] 
     if searched_item_name.lower() in name.lower() and name not in item_names]
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
    umm_for_selected_user = user_item_matrix[user_item_matrix.index == selected_user_id]
    bool_for_selected_user = umm_for_selected_user.apply(lambda col: col.notnull(), axis=1)
    item_ids_ratedby_selected_user = bool_for_selected_user.iloc[0].loc[lambda item_id: item_id == True]
    list_item_ids_ratedby_selected_user = bool_for_selected_user.iloc[0].loc[lambda item_id: item_id == True].index.to_list()
    print(f"Step 1: User rated {len(list_item_ids_ratedby_selected_user)} items")
    print(f"Rated item IDs: {list_item_ids_ratedby_selected_user[:10]}...")  # Show first 10
    allUserIds_and_items_ratedby_selected_user = user_item_matrix.loc[:, list_item_ids_ratedby_selected_user]
    count_threshold_rated_same_items = len(list_item_ids_ratedby_selected_user) * perc_threshold_rated_same_products
    print(f"Step 2: Need at least {count_threshold_rated_same_items:.1f} overlapping items")
    userIds_rated_atleast_X_perc_same_items_with_selected_user = allUserIds_and_items_ratedby_selected_user[
        allUserIds_and_items_ratedby_selected_user.notnull().sum(axis=1) > count_threshold_rated_same_items]
    print(f"Step 3: Found {len(userIds_rated_atleast_X_perc_same_items_with_selected_user)} users with enough overlap")
    if len(userIds_rated_atleast_X_perc_same_items_with_selected_user) == 0:
        print("ERROR: No users found with sufficient overlap!")
        print("="*50)
        return (pd.DataFrame(), {}) if return_corrs else pd.DataFrame()
    corr_of_userIds_rated_same_items_with_selected_user = userIds_rated_atleast_X_perc_same_items_with_selected_user.T.corr().unstack()
    userIds_corr_df = pd.DataFrame(corr_of_userIds_rated_same_items_with_selected_user, columns=["corr"])
    userIds_corr_df.index.names = ["userId_1","userId_2"]
    userIds_corr_df.reset_index(inplace=True)
    if corr_threshold == 0.0:
        final_users_corr_df = userIds_corr_df[(userIds_corr_df["userId_1"] == selected_user_id) & 
                                              (userIds_corr_df["userId_2"] != selected_user_id) &
                                              ((userIds_corr_df["corr"] >= corr_threshold) | (userIds_corr_df["corr"].isna()))].sort_values(by="corr", ascending=False)
    else:
        final_users_corr_df = userIds_corr_df[(userIds_corr_df["userId_1"] == selected_user_id) & 
                                              (userIds_corr_df["userId_2"] != selected_user_id) &
                                              (userIds_corr_df["corr"] >= corr_threshold)].sort_values(by="corr", ascending=False)
    print(f"Step 4: After correlation filter (>={corr_threshold}): {len(final_users_corr_df)} similar users")
    if len(final_users_corr_df) == 0:
        print("ERROR: No users found with sufficient correlation!")
        print("="*50)
        return (pd.DataFrame(), {}) if return_corrs else pd.DataFrame()
    list_users_to_filter = final_users_corr_df["userId_2"].to_list()
    user_corr_dict = dict(zip(final_users_corr_df["userId_2"], final_users_corr_df["corr"]))
    final_rec_df = user_item_matrix.loc[list_users_to_filter,:]
    final_rec_excluded_selected_user_items_df = final_rec_df.T
    final_rec_excluded_selected_user_items_df = final_rec_excluded_selected_user_items_df[
        ~final_rec_excluded_selected_user_items_df.index.isin(list_item_ids_ratedby_selected_user)]
    final_rec_excluded_selected_user_items_df = final_rec_excluded_selected_user_items_df.loc[
        ~final_rec_excluded_selected_user_items_df.apply(lambda row: row.isnull().all(), axis=1),:]
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
