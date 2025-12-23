"""
Data loading and preprocessing utilities
"""
import pandas as pd

def load_dataset():
    """Load movies and ratings datasets and merge them"""
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    dataframe = ratings.merge(movies, how="left")
    return dataframe

def data_info(dataframe):
    """Display dataframe information"""
    print("Dataframe Shape:", dataframe.shape)
    print("-"*50)
    print("Dataframe Info:")
    print(dataframe.info())
    print("-"*50)
    print("Missing Values in Each Column:")
    print(dataframe.isnull().sum())
    print("-"*50)
    print("Dataframe Sample:")
    print(dataframe.head())

def data_stats(dataframe, user_col="user_id", item_col="item_id", user_rating_threshold=30, item_rated_threshold=1000):
    """Display statistics about users and items"""
    user_count = pd.DataFrame(dataframe[user_col].value_counts(), columns=["count"])
    product_count = pd.DataFrame(dataframe[item_col].value_counts(), columns=["count"])
    
    u_less_than_threshold = len(user_count[user_count["count"] < user_rating_threshold])
    u_more_than_threshold = len(user_count[user_count["count"] >= user_rating_threshold])
    
    m_less_than_threshold = len(product_count[product_count["count"] < item_rated_threshold])
    m_more_than_threshold = len(product_count[product_count["count"] >= item_rated_threshold])
    
    print("*"*100)
    print("Total User Count: ", len(user_count))
    print("User count rated less than", user_rating_threshold, "times: ", u_less_than_threshold,"(", round((u_less_than_threshold / len(user_count))*100,2), " %)")
    print("User count rated more than", user_rating_threshold, "times: ", u_more_than_threshold,"(", round((u_more_than_threshold / len(user_count))*100,2), " %)")
    print("-"*50)
    print("Total Product Count: ", len(product_count))
    print("Product count rated less than", item_rated_threshold, "times: ", m_less_than_threshold,"(", round((m_less_than_threshold / len(product_count))*100,2), " %)")
    print("Product count rated more than", item_rated_threshold, "times: ", m_more_than_threshold,"(", round((m_more_than_threshold / len(product_count))*100,2), " %)")
    print("*"*100)

def dataframe_reduction(dataframe, user_col="user_id", item_col="item_id", user_rating_threshold=30, item_rated_threshold=1000):
    """Reduce dataframe by removing users and items with low interaction counts"""
    user_count = pd.DataFrame(dataframe[user_col].value_counts(), columns=["count"])
    item_count = pd.DataFrame(dataframe[item_col].value_counts(), columns=["count"])
    
    items_to_be_removed = item_count[item_count["count"] < item_rated_threshold].index.to_list()
    users_to_be_removed = user_count[user_count["count"] < user_rating_threshold].index.to_list()
    
    reduced_dataframe = dataframe[~dataframe[item_col].isin(items_to_be_removed)]
    reduced_dataframe = reduced_dataframe[~reduced_dataframe[user_col].isin(users_to_be_removed)]
    
    before = len(user_count) * len(item_count)
    after = len(reduced_dataframe[user_col].value_counts()) * len(reduced_dataframe[item_col].value_counts())
    
    print("Matrix Size BEFORE reduction:", len(user_count), "*", len(item_count),"=", before)
    print("Matrix Size AFTER reduction:", len(reduced_dataframe[user_col].value_counts()), "*", len(reduced_dataframe[item_col].value_counts()),"=", after)
    print("Matrix Size Reduction Rate:", round((before - after)/before, 2)*100, "%")
    print("Information kept:", round(len(reduced_dataframe)/len(dataframe), 2)*100, "%")
    
    return reduced_dataframe
