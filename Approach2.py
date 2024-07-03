import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
from pprint import pprint
import json
import csv
import openai
import re
import random

# Read test dataset
test_df = pd.read_csv("/content/test_0.8_0.2.csv")
test_df.drop(columns=['timestamp'], inplace=True)

# Read train dataset
train_df = pd.read_csv("/content/train_0.8_0.2.csv")
train_df.drop(columns=['timestamp'], inplace=True)

# Read items dataset
item_df = pd.read_csv("/content/items.csv")

# Drop unnecessary columns from item dataset
item_df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1',
                       'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1', 
                       'IMDb URL', 'Unnamed: 0.1.1.1.1.1.1', 'unknown', 'Unnamed: 38', 
                       'Unnamed: 39', 'Unnamed: 40', 'video release date', 'No. of ratings',	
                       'YT-Trailer ID', 'Runtime'], inplace=True)

# Select required columns from item dataset
item_df = item_df[['movie_id', 'movie_title', 'release date']]

def MakeDataFrame(new_df):
    # Grouping by userId
    grouped_df = new_df.groupby('userId')
    user_movie = grouped_df.count()
    user_movie = user_movie.sort_values(by='movieId', ascending=False)
    user_movie.reset_index(inplace=True)
    user_movie.drop(columns=['movieId', 'rating'], inplace=True)
    # Merging with original dataframe
    merged_df = pd.merge(new_df, user_movie, on='userId')
    group_df = merged_df.groupby('userId')
    new_data = []
    for user_id, group in group_df:
        movies = group['movieId'].tolist()
        ratings = group['rating'].tolist()
        new_data.append({'userId': user_id, 'movies': movies, 'ratings': ratings})

    new_df = pd.DataFrame(new_data)
    return new_df

# Generate DataFrame for test and train datasets
test_df = MakeDataFrame(test_df)
train_df = MakeDataFrame(train_df)

openai_key = "API_KEY"
openai.api_key = openai_key

def get_movie_name_meta_info_pairs(movie_df, item_df):
    # Merge movie and item datasets
    merged_df = pd.merge(movie_df, item_df, on='movie_id')
    # Convert merged dataframe to list
    movie_name_release_date_pairs = merged_df.values.tolist()
    return movie_name_release_date_pairs

def generate_batches(dataframe, batch_size):
    num_batches = math.ceil(len(dataframe) / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataframe))
        yield dataframe.iloc[start_idx:end_idx]



import pandas as pd

# Initialize a counter
cnt = 0

# Iterate over batches of data
for batch in generate_batches(train_df, 10):
    for index, row in batch.iterrows():
        # Extract user id, movies, and ratings
        user_id = row['userId']
        movies = row['movies']
        ratings = row['ratings']

        # Initialize lists to store movie IDs for each rating category
        five_star_movie_id = []
        four_star_movie_id = []
        three_star_movie_id = []
        two_star_movie_id = []
        one_star_movie_id = []

        # Categorize movies based on ratings
        for id, rating in zip(movies, ratings):
            if rating == 5:
                five_star_movie_id.append(id)
            elif rating == 4:
                four_star_movie_id.append(id)
            elif rating == 3:
                three_star_movie_id.append(id)
            elif rating == 2:
                two_star_movie_id.append(id)
            elif rating == 1:
                one_star_movie_id.append(id)

        # Convert lists to DataFrames
        five_star_movie_df = pd.DataFrame(five_star_movie_id, columns=['movie_id'])
        four_star_movie_df = pd.DataFrame(four_star_movie_id, columns=['movie_id'])
        three_star_movie_df = pd.DataFrame(three_star_movie_id, columns=['movie_id'])
        two_star_movie_df = pd.DataFrame(two_star_movie_id, columns=['movie_id'])
        one_star_movie_df = pd.DataFrame(one_star_movie_id, columns=['movie_id'])

        # Get movie names and meta info pairs
        five_star_movies = get_movie_name_meta_info_pairs(five_star_movie_df, item_df) if not five_star_movie_df.empty else []
        four_star_movies = get_movie_name_meta_info_pairs(four_star_movie_df, item_df) if not four_star_movie_df.empty else []
        three_star_movies = get_movie_name_meta_info_pairs(three_star_movie_df, item_df) if not three_star_movie_df.empty else []
        two_star_movies = get_movie_name_meta_info_pairs(two_star_movie_df, item_df) if not two_star_movie_df.empty else []
        one_star_movies = get_movie_name_meta_info_pairs(one_star_movie_df, item_df) if not one_star_movie_df.empty else []

        # Construct a message detailing user preferences
        message1 = f"""Please do not allow any effect of previously generated output in the current response. \
Input and output contain five categories of ratings: 1 star, 2 star, 3 star, 4 star, and 5 star. \
1 star indicates that the user least likes to watch that movie, and a rating of 5 stars means the user \
most liked to watch that movie. Ratings of 2, 3, and 4 follow the same notion accordingly. \
Now, here are the movies with their release dates, rated by this single user, categorized by their star ratings. \
Movies rated 5 stars: '{five_star_movies}'. \
Movies rated 4 stars: '{four_star_movies}'. \
Movies rated 3 stars: '{three_star_movies}'. \
Movies rated 2 stars: '{two_star_movies}'. \
Movies rated 1 star: '{one_star_movies}'. \

You have to find a rating-wise pattern in user preferences using the given movies' names, release dates, and other factors such as cast, director, IMDb rating, genres, summary, movie nature, theme, storyline, cinematography, movie pace, It factor, tag, message, etc. \
And that will help me in predicting the rating value of a new movie for the same single user in the future.

STRICTLY FOLLOW 1: Don't specify any movie name in the response. \
STRICT FOLLOW 2: If I do not provide which type of movies the user gives a 1-star rating to, \
then you don't need to provide any pattern in the user's preferences for 1-star ratings. \
If I do not provide which type of movies the user gives a 2-star rating to, then you don't need to provide any \
pattern in the user's preferences for 2-star ratings. If I do not provide which type of movies the user \
gives a 3-star rating to, then you don't need to provide any pattern in the user's preferences for 3-star ratings. \
If I do not provide which type of movies the user gives a 4-star rating to, then you don't need to provide any pattern \
in the user's preferences for 4-star ratings. If I do not provide which type of movies the user gives a 5-star rating to, \
then you don't need to provide any pattern in the user's preferences for 5-star ratings."""

        # Remove sections with no rated movies
        if not five_star_movies:
           message1 = message1.replace(f"Movies rated 5 stars: '{five_star_movies}'.", "")

        if not four_star_movies:
           message1 = message1.replace(f"Movies rated 4 stars: '{four_star_movies}'.", "")

        if not three_star_movies:
           message1 = message1.replace(f"Movies rated 3 stars: '{three_star_movies}'.", "")

        if not two_star_movies:
           message1 = message1.replace(f"Movies rated 2 stars: '{two_star_movies}'.", "")

        if not one_star_movies:
           message1 = message1.replace(f"Movies rated 1 star: '{one_star_movies}'.", "")

        # Prepare messages for AI response
        messages2 = [{"role": "system", "content":message1}]
        
        # Generate AI response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages2,
                temperature=0.5,
                seed=42
            )
        except openai.InvalidRequestError as e:
            print("Error processing batch:", e)
            continue  # Move to the next batch if there's an error
        
        # Extract content from response
        content = response["choices"][0]["message"]["content"]
        
        # Assign predicted rating pattern to dataframe
        train_df.at[cnt, 'predicted_rating_pattern'] = content
        cnt += 1

# Select relevant columns from train_df and merge with test_df
train_df_predicted_rating_pattern = train_df[['userId','predicted_rating_pattern']]
test_df = pd.merge(test_df, train_df_predicted_rating_pattern, on='userId')


import math
import pandas as pd
import json
import re
import openai

# Function to generate batches of users
def generate_batches(dataframe, batch_size):
    num_batches = math.ceil(len(dataframe) / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataframe))
        yield dataframe.iloc[start_idx:end_idx]

# Iterate over batches
for batch in generate_batches(test_df, 10):
    for index, row in batch.iterrows():
        user_id = row['userId']
        rating_pattern = row['predicted_rating_pattern']

        movies = row['movies']
        # Extract movie IDs
        movie_ids = [int(x) for x in movies if isinstance(x, int) or re.match(r'^\d+$', str(x))]

        # Create DataFrame with movie IDs
        top_movies_df = pd.DataFrame({'movie_id': movie_ids})

        # Merge with movies_df
        movie_name_release_date_pairs = pd.merge(top_movies_df, item_df, on='movie_id').values.tolist()

        # JSON for recommended ratings
        recommended_ratings_json = json.dumps({"ratings": [2, 4, 5, 3, 4, 5, 2, 4, 5]})

        # Message for AI completion
        message1 = f"""Please do not allow any influence of previously generated output in the current response.\
        Output contains five  category of rating that is 1 star, 2 star, 3 star, 4 star, and 5 star.\
        1 star indicates that the user least like to watch that movie, and a rating of 5 star means the user
        most liked to watch that movie. Ratings of 2, 3, and 4 follow the same notion accordingly.\
        Now, here is the rating-wise user preference pattern of the user generated from the user’s watched history: '{rating_pattern}'. Take your time to understand it and keep it in your memory properly.\
        Now, these are the new movies listed below with their release dates: '{movie_name_release_date_pairs}'.\
        For these movies, predict the appropriate rating based on the above given rating-wise user preference pattern, and consider other factors as well.\
        Please provide the predicted ratings in the JSON format shown below: {recommended_ratings_json}.\
        Make sure the predicted ratings are in the same order as the movies listed earlier.

        STRICTLY FOLLOW 1: Don't give text in response, only provide a list of ratings.
        """

        messages2 = [
            {"role": "system", "content": """You are an expert in understanding rating-wise user preference pattern. Please provide ratings for all the given movies based on the user preferences pattern analysis."""},
            {'role': 'user', 'content': message1}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages2,
                temperature=0.5,
                seed=42
            )
        except openai.InvalidRequestError as e:
            print("Error processing batch:", e)
            continue

        predicted_rating = response["choices"][0]["message"]

        try:
            content_json = json.loads(predicted_rating["content"])
            ratings_list = content_json["ratings"]
            print(ratings_list)
            test_df.at[index, 'predicted_rating'] = ratings_list
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)

# Read additional data
temp_test_df = pd.read_csv("/content/test_0.8_0.2.csv")
temp_test_df.drop(columns=['timestamp'], inplace=True)

temp_item_df = pd.read_csv("/content/items.csv")
temp_item_df = temp_item_df[['movie_id', 'Rating']]
temp_item_df.rename(columns={'Rating': 'imdb_rating', 'movie_id': 'movieId'}, inplace=True)

# Merge data
temp_test_df = pd.merge(temp_test_df, temp_item_df, on='movieId')

# Function to create DataFrame for IMDb ratings
def MakeDataFrame_for_imdb(new_df):
    grouped_df = new_df.groupby('userId')
    user_movie = grouped_df.count()
    user_movie = user_movie.sort_values(by='movieId', ascending=False)
    user_movie.reset_index(inplace=True)
    user_movie.drop(columns=['movieId', 'rating', 'imdb_rating'], inplace=True)
    merged_df = pd.merge(new_df, user_movie, on='userId')
    group_df = merged_df.groupby('userId')
    new_data = []
    for user_id, group in group_df:
        movies = group['movieId'].tolist()
        ratings = group['rating'].tolist()
        imdb_ratings = group['imdb_rating'].tolist()
        new_data.append({'userId': user_id, 'movies': movies, 'ratings': ratings, 'imdb_rating': imdb_ratings})
    new_df = pd.DataFrame(new_data)
    return new_df

temp_test_df = MakeDataFrame_for_imdb(temp_test_df)

test_df['imdb_rating'] = temp_test_df['imdb_rating']
test_df.dropna(inplace=True)

# Function to convert IMDb list to float
def convert_imdb_list_to_float(imdb_list):
    float_list = []
    index = 1
    while index < len(imdb_list) - 1:
        num = int(imdb_list[index])
        index += 2
        dec = int(imdb_list[index])
        ratings_list = num + dec / 10.0
        float_list.append(float(ratings_list))
        index += 3
    return float_list

Final_result = test_df.copy()


cleaned_df = Final_result.copy()

# Function to convert ratings column values from string to list
def convert_ratings_to_list(ratings):
    return [int(x) for x in ratings.strip('[]').split(',')]

# Applying conversion to 'ratings' and 'predicted_rating_list' columns
cleaned_df['ratings'] = cleaned_df['ratings'].apply(convert_ratings_to_list)
cleaned_df['predicted_rating_list'] = cleaned_df['predicted_rating_list'].apply(convert_ratings_to_list)

# Extracting necessary columns
actual_ratings = cleaned_df['ratings']
predicted_ratings = cleaned_df['predicted_rating_list']
imdb_list = cleaned_df['imdb_rating']

# Function to order indices based on ratings and IMDb ratings
def order_indices(ratings, imdb_list):
    '''Generates the sequence of indices as rank based on the rating and IMDb rating values. First rank at first position'''
    rating_array = np.array(ratings)
    unique_ratings = sorted(list(set(ratings)))[::-1]
    order_indices = []

    for ur in unique_ratings:
        rtng_indices = np.where(rating_array == ur)[0]
        imdb = [imdb_list[r_indx] for r_indx in rtng_indices]
        imdb = list(set(imdb))
        imdb = sorted(imdb, reverse=True)

        for i in imdb:
            for r_indx in rtng_indices:
                if imdb_list[r_indx] == i:
                    order_indices.append(r_indx)

    return order_indices

# Function to calculate MRR at k
def mrr_at_k(ac_order_indc, pr_order_indc, actual_ratings, predicted_ratings, K):
    '''Generates MRR@k'''
    k, rr = 0, 0
    K = K if K <= len(predicted_ratings) else len(predicted_ratings)
    for i, indx in enumerate(pr_order_indc):
        if k < K:
            if predicted_ratings[indx] >= 3 and actual_ratings[ac_order_indc[i]] >= 3:
                rr += 1 / (i + 1)
            elif predicted_ratings[indx] < 3 and actual_ratings[ac_order_indc[i]] < 3:
                rr += 1 / (i + 1)
        k += 1
    MRR_at_k = rr / K
    return MRR_at_k

# Function to calculate FCP at k
def fcp_at_K(actual_ratings, pr_order_indc, K):
    '''Generates FCP@k'''
    K = K if K <= len(actual_ratings) else len(actual_ratings)
    concordant = 0
    top_k_indices = pr_order_indc[:K]

    for i, indx in enumerate(top_k_indices):
        if i < (K - 1):
            nxt_top_k_indices = pr_order_indc[i + 1:K]
            for nxt_indics in nxt_top_k_indices:
                concordant += 1 if actual_ratings[indx] >= actual_ratings[nxt_indics] else 0

    total_pair = math.comb(K, 2)
    fcp_at_k = concordant / total_pair
    return fcp_at_k

# Function to calculate DCG at k
def dcg_at_K(actual_ratings, indeces_order, K):
    '''Generates the DCG on a cut of value of K'''
    k, dcg = 0, 0
    K = K if K <= len(actual_ratings) else len(actual_ratings)

    for indx, rating_indx in enumerate(indeces_order):
        relavance = actual_ratings[rating_indx]
        numerator = pow(2, relavance) - 1
        denomirator = math.log2(indx + 2)
        dcg += numerator / denomirator

        k += 1
        if k == K:
            return dcg

    return dcg

# Function to calculate NDCG, MRR, and FCP at k
def ndcg_mrr_fpc_k(actual_ratings, predicted_ratings, imdb_list, K):
    '''Takes the actual and predicted ratings of movies with IMDb list and cutoff k. Returns the NDCG@K, MRR@K, and FCP@K'''
    ac_order_indc = order_indices(actual_ratings, imdb_list)
    pr_order_indc = order_indices(predicted_ratings, imdb_list)

    # Calculating NDCG
    dcg_ac = dcg_at_K(actual_ratings, ac_order_indc, K)
    dcg_pr = dcg_at_K(actual_ratings, pr_order_indc, K)
    ndcg_k = dcg_pr / dcg_ac if dcg_ac != 0 else 0

    # Calculating MRR
    MRR_at_k = mrr_at_k(ac_order_indc, pr_order_indc, actual_ratings, predicted_ratings, K)

    # Calculating FCP
    fcp_at_k = fcp_at_K(actual_ratings, pr_order_indc, K)

    return (ndcg_k, MRR_at_k, fcp_at_k)

# Function to calculate MAE, MSE, RMSE, and R2 at k
def MAE_MSE_RMSE_R2_at_K(actual_ratings, predicted_ratings, K):
    '''It returns MAE@K, MSE@K, RMSE@K, and R2'''
    K = K if K <= len(actual_ratings) else len(actual_ratings)

    k, AE, SE, SS_res, SS_tot = 0, 0, 0, 0, 0
    ac_avg = np.average(np.array(actual_ratings))

    for indx, _ in enumerate(actual_ratings):
        if k < K:
            # Calculate AE
            AE += abs(actual_ratings[indx] - predicted_ratings[indx])

            # Calculate SE
            error = (actual_ratings[indx] - predicted_ratings[indx])
            SE += pow(error, 2)

        # Calculate R2
        error = (actual_ratings[indx] - predicted_ratings[indx])
        SS_res += pow(error, 2)
        error = actual_ratings[indx] - ac_avg
        SS_tot += pow(error, 2)

        k += 1

    MAE_at_K = AE / K
    MSE_at_K = SE / K
    RMSE_at_K = pow(MSE_at_K, 0.5)

    # Handling division by zero error for R2 calculation
    if SS_tot == 0:
        R2 = 0
    else:
        R2 = 1 - (SS_res / SS_tot)

    return MAE_at_K, MSE_at_K, RMSE_at_K, R2

# Range of k values
k_values = range(3, 21)

results = []

# Loop through k values
for k in k_values:
    ndcg = 0
    mrr = 0
    fcp = 0
    MAE = 0
    MSE = 0
    RMSE = 0
    R2 = 0
    cnt = 0

    # Loop through rows of the dataframe
    for index, row in cleaned_df.iterrows():
        user_id = row['userId']
        movies = row['movies']
        ratings = row['ratings']
        predicted_rating_list = row['predicted_rating_list']
        imdb_rating = row['imdb_rating']

        # Find the minimum length among the lists
        min_length = min(len(ratings), len(predicted_rating_list), len(imdb_rating))

        # Truncate the lists to the minimum length
        ratings = ratings[:min_length]
        predicted_rating_list = predicted_rating_list[:min_length]
        imdb_rating = imdb_rating[:min_length]

        # Save truncated lists back to the dataframe
        cleaned_df.at[index, 'ratings'] = ratings
        cleaned_df.at[index, 'predicted_rating_list'] = predicted_rating_list
        cleaned_df.at[index, 'imdb_rating'] = imdb_rating

        # Calculate metrics for the current row
        result1 = ndcg_mrr_fpc_k(ratings, predicted_rating_list, imdb_rating, k)
        result2 = MAE_MSE_RMSE_R2_at_K(ratings, predicted_rating_list, k)

        # Accumulate results
        ndcg += result1[0]
        mrr += result1[1]
        fcp += result1[2]
        MAE += result2[0]
        MSE += result2[1]
        RMSE += result2[2]
        R2 += result2[3]
        cnt += 1

    # Calculate average metrics for current k
    ndcg /= cnt
    mrr /= cnt
    fcp /= cnt
    MAE /= cnt
    MSE /= cnt
    RMSE /= cnt
    R2 /= cnt

    # Append results to list
    results.append({
        'k': k,
        'ndcg': ndcg,
        'mrr': mrr,
        'fcp': fcp,
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'R2': R2
    })

# Create DataFrame from results and save to CSV
df = pd.DataFrame(results)
df.to_csv('result_matric.csv', index=False)


