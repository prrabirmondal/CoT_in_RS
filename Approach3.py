import pandas as pd
import numpy as np
import math
from pprint import pprint
import json
import csv
import openai

# Read test dataset
path_test = "/content/test_0.8_0.2.csv"
test_df = pd.read_csv(path_test)

# Read train dataset
path_train = "/content/train_0.8_0.2.csv"
train_df = pd.read_csv(path_train)

# Read movies info dataset
csv_file = 'movies_info.csv'
movies_df = pd.read_csv(csv_file)

# Read movies with IMDb dataset
path_movie = "/content/movie_with_imdb.csv"
movies_imdb_df = pd.read_csv(path_movie)

# Drop timestamp column from train and test datasets
train_df.drop(columns=['timestamp'], inplace=True)
test_df.drop(columns=['timestamp'], inplace=True)

# Select relevant columns from movies info dataset
movies_df = movies_df[['movie id', 'movie title', 'release date']]

# Rename and select columns from movies with IMDb dataset
movies_imdb_df['imdb_rating'] = movies_imdb_df['Rating']
movies_imdb_df = movies_imdb_df[['movie_id', 'movie_title', 'imdb_rating']]

# Define a function to create DataFrame from train and test datasets
def MakeDataFrame(new_df):
    grouped_df = new_df.groupby('userId')
    user_movie = grouped_df.count()
    user_movie = user_movie.sort_values(by='movieId', ascending=False)
    user_movie.reset_index(inplace=True)
    user_movie.drop(columns=['movieId', 'rating'], inplace=True)
    merged_df = pd.merge(new_df, user_movie, on='userId')
    group_df = merged_df.groupby('userId')
    new_data = []
    for user_id, group in group_df:
        movies = group['movieId'].tolist()
        ratings = group['rating'].tolist()
        new_data.append({'userId': user_id, 'movies': movies, 'ratings': ratings})
    new_df = pd.DataFrame(new_data)
    return new_df

# Generate DataFrames for test and train datasets
test_df = MakeDataFrame(test_df)
train_df = MakeDataFrame(train_df)

# Set OpenAI API key
openai_key = "API_KEY"
openai.api_key = openai_key


function = {
    "name": "pattern_in_user_preference",
    "description": "A function that identifies patterns in user preference based on movie rated by him/her",
    "parameters": {
        "type": "object",
        "properties": {
            "5_star":{
                "type": "string",
                "description":"Pattern in user preference for 5 star rated movies"
            },
            "4_star":{
                "type": "string",
                "description":"Pattern in user preference for 4 star rated movies"
            },
            "3_star":{
                "type": "string",
                "description":"Pattern in user preference for 3 star rated movies"
            },
            "2_star":{
                "type": "string",
                "description":"Pattern in user preference for 2 star rated movies"
            },
            "1_star":{
                "type": "string",
                "description":"Pattern in user preference for 1 star rated movies"
            }
        }
    }
}


sample_response = openai.openai_object.OpenAIObject()
sample_response["role"] = "assistant"
sample_response["content"] = None
sample_response["function_call"] = {
    "name": "pattern_in_user_preference",
    "arguments": json.dumps(
        {
            "2_star": "The user gave 2 star rating to the movies with newer release dates, possibly indicating a preference for more recent films.",
            "3_star": "The user gave 3 star rating to the movies with a mix of genres and themes, suggesting moderate enjoyment without strong preferences.",
            "4_star": "The user gave 4 star rating to the movies with romantic or light-hearted themes, offering good entertainment value and possibly resonating emotionally.",
            "5_star": "The user gave 5 star rating to the classic films with timeless appeal, strong narratives, and exceptional performances."
        },
        indent=2,
        ensure_ascii=False,
    ),
}

message2 = """Please do not allow any effect of previously generated output in the current response.    Input and output contains five  category of rating that is 1 star, 2 star, 3 star, 4 star, and 5 star.
    1 star indicates that the user least like to watch that movie, and a rating of 5 star means the user
    most liked to watch that movie. Ratings of 2, 3, and 4 follow the same notion accordingly.    Now, here are the movies with their release dates rated by this single user, categorized by their star ratings.
      Movies name with release date = '[('Casablanca (1942)', '01-Jan-1942'), ('Top Gun (1986)', '01-Jan-1986'), ('Godfather: Part II, The (1974)', '01-Jan-1974')]' and the user rated these movies to 5 star.
      Movies name with release date = '[('Four Weddings and a Funeral (1994)', '01-Jan-1994'), ('Sleepless in Seattle (1993)', '01-Jan-1993')]' and the user rated these movies to 4 star.
      Movies name with release date = '[('Mask, The (1994)', '01-Jan-1994'), ('Raising Arizona (1987)', '01-Jan-1987'), ('My Fellow Americans (1996)', '20-Dec-1996')]' and the user rated these movies to 3 star.
      Movies name with release date = '[('Jungle2Jungle (1997)', '07-Mar-1997'), ('Somewhere in Time (1980)', '01-Jan-1980'), ('Happy Gilmore (1996)', '16-Feb-1996')]' and the user rated these movies to 2 star.

      You have to find rating wise pattern in user preference based on given movies' name, released date and use other
      factors also like movies’ nature, director, casts, theme, storyline, cinematography, movie pace, It factor, genres, tag, message, imdb rating etc.
      And that will help me in predicting the rating value of an unknown movie for the same single user in the future.

      STRICTLY FOLLOW: 1. Don't specify any movie name in the response.
      STRICT FOLLOW 2: If I do not provide which type of movies the user gives a 1-star rating to,
      then you don't need to provide any pattern in the user's preferences for 1-star ratings. If I do not provide
      which type of movies the user gives a 2-star rating to, then you don't need to provide any pattern in the user's
      preferences for 2-star ratings. If I do not provide which type of movies the user gives a 3-star rating to,
      then you don't need to provide any pattern in the user's preferences for 3-star ratings. If I do not provide
      which type of movies the user gives a 4-star rating to, then you don't need to provide any pattern in the
      user's preferences for 4-star ratings. If I do not provide which type of movies the user gives a 5-star rating to,
      then you don't need to provide any pattern in the user's preferences for 5-star ratings."""


for index, row in train_df.iterrows():
    user_id = row['userId']

    # Assuming 'new_df' is your DataFrame
    movies = row['movies']
    ratings = row['ratings']
    top_movies_df = pd.DataFrame({'movie id': movies})

    merged_df = pd.merge(top_movies_df, movies_df, on='movie id')
    movie_name_release_date_pairs = merged_df[['movie title', 'release date']].values.tolist()

    five_star_movies = []
    four_star_movies = []
    three_star_movies = []
    two_star_movies = []
    one_star_movies = []

    for pair, rating in zip(movie_name_release_date_pairs, ratings):
        movie_title, release_date = pair
        if rating == 5:
            five_star_movies.append((movie_title, release_date))
        elif rating == 4:
            four_star_movies.append((movie_title, release_date))
        elif rating == 3:
            three_star_movies.append((movie_title, release_date))
        elif rating == 2:
            two_star_movies.append((movie_title, release_date))
        elif rating == 1:
            one_star_movies.append((movie_title, release_date))


    message1 = f"""Please do not allow any effect of previously generated output in the current response.\
    Input and output contains five  category of rating that is 1 star, 2 star, 3 star, 4 star, and 5 star.
    1 star indicates that the user least like to watch that movie, and a rating of 5 star means the user
    most liked to watch that movie. Ratings of 2, 3, and 4 follow the same notion accordingly.\
    Now, here are the movies with their release dates rated by this single user, categorized by their star ratings.
      Movies name with release date = '{five_star_movies}' and the user rated these movies to 5 star.
      Movies name with release date = '{four_star_movies}' and the user rated these movies to 4 star.
      Movies name with release date = '{three_star_movies}' and the user rated these movies to 3 star.
      Movies name with release date = '{two_star_movies}' and the user rated these movies to 2 star.
      Movies name with release date = '{one_star_movies}' and the user rated these movies to 1 star.\

      You have to find rating wise pattern in user preference based on given movies' name, released date and use other
      factors also like movies’ nature, director, casts, theme, storyline, cinematography, movie pace, It factor, genres, tag, message, imdb rating etc.
      And that will help me in predicting the rating value of an unknown movie for the same single user in the future.

      STRICTLY FOLLOW 1: Don't specify any movie name in the response.
      STRICT FOLLOW 2: If I do not provide which type of movies the user gives a 1-star rating to,
      then you don't need to provide any pattern in the user's preferences for 1-star ratings. If I do not provide
      which type of movies the user gives a 2-star rating to, then you don't need to provide any pattern in the user's
      preferences for 2-star ratings. If I do not provide which type of movies the user gives a 3-star rating to,
      then you don't need to provide any pattern in the user's preferences for 3-star ratings. If I do not provide
      which type of movies the user gives a 4-star rating to, then you don't need to provide any pattern in the
      user's preferences for 4-star ratings. If I do not provide which type of movies the user gives a 5-star rating to,
      then you don't need to provide any pattern in the user's preferences for 5-star ratings."""




    if not five_star_movies:
       message1 = message1.replace(f"Movies name with release date = '{five_star_movies}' and the user rated these movies to 5 star.", "")

    if not four_star_movies:
       message1 = message1.replace(f"Movies name with release date = '{four_star_movies}' and the user rated these movies to 4 star.", "")

    if not three_star_movies:
       message1 = message1.replace(f"Movies name with release date = '{three_star_movies}' and the user rated these movies to 3 star.", "")

    if not two_star_movies:
       message1 = message1.replace(f"Movies name with release date = '{two_star_movies}' and the user rated these movies to 2 star.", "")

    if not one_star_movies:
       message1 = message1.replace(f"Movies name with release date = '{one_star_movies}' and the user rated these movies to 1 star.", "")

    messages2 = [
        {"role": "system", "content": """You are an expert in identifying the pattern in user preference (meaning which type of movies the single
        user most likes to watch and which type of movies the user least like to watch) based on their rating values of movies. Please provide the rating wise pattern
        in user preference and then extract the relevant data to use as arguments to pass into the given function provided.""" },
        {'role': 'user', 'content': message2},
        sample_response,
        {'role': 'user', 'content': message1}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages2,
        functions=[function],
        function_call={"name": "pattern_in_user_preference"}, # this forces calling `function`
        temperature=0.5,
        seed=42
    )


    content = response["choices"][0]["message"]["function_call"]["arguments"]

    train_df.loc[index, 'predicted_rating_pattern'] = content




function ={
    "name": "Give_rating_using_pattern_in_user_preference",
    "description": "A function that gives ratings to movies using patterns in the user preferences.",
    "parameters": {
        "type": "object",
        "properties": {
            "recommended_ratings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "predicted_ratings": {"type": "integer"}
                    },
                    "required": ["predicted_ratings"]
                },
                "description": "List of predicted ratings values for all movies."
            }
        }
    }
}


from openai.openai_object import OpenAIObject

NewSample_response = OpenAIObject()
NewSample_response["role"] = "assistant"
NewSample_response["content"] = None
NewSample_response["function_call"] = {
    "name": "Give_rating_using_pattern_in_user_preference",
    "arguments": json.dumps(
        {
            "recommended_ratings": [
                {"predicted_ratings": 3},
                {"predicted_ratings": 4},
                {"predicted_ratings": 2},
                {"predicted_ratings": 5},
                {"predicted_ratings": 4},
                {"predicted_ratings": 3},
                {"predicted_ratings": 2},
                {"predicted_ratings": 1},
                {"predicted_ratings": 3},
                {"predicted_ratings": 5}
            ]
        },
        indent=2,
        ensure_ascii=False
    )
}

message3 = """Please do not allow any effect of previously generated output in the current response.\
              Now, here is the rating wise pattern in the user preference: {
                  "4_star": "The user gave 4 star rating to the movies with a mix of drama and romance, strong character development, and critically acclaimed.",
                  "3_star": "The user gave 3 star rating to the movies with suspenseful plots, average IMDB ratings, and a focus on psychological themes.",
                  "5_star": "The user gave 5 star rating to the movies with complex storylines, exceptional cinematography, and high IMDB ratings."
              },
              and these are the movies listed below with their release dates: [['Love and Death on Long Island (1997)', '10-Mar-1998'],
              ['Lion King, The (1994)', '01-Jan-1994'], ['Airheads (1994)', '01-Jan-1994'], ['For Richer or Poorer (1997)', '01-Jan-1997'],
              ['Wolf (1994)', '01-Jan-1994'], ["Mary Shelley's Frankenstein (1994)", '01-Jan-1994'], ['Nick of Time (1995)', '01-Jan-1995'],
              ['Richie Rich (1994)', '01-Jan-1994'], ['Around the World in 80 Days (1956)', '01-Jan-1956']].\
              Predict the rating scores for all movies by considering the rating wise pattern in the user preference."""

train_df_predicted_rating_pattern = train_df[['userId','predicted_rating_pattern']]
test_df = pd.merge(test_df, train_df_predicted_rating_pattern, on='userId')


test_df = pd.read_csv("/content/prompt2_output (1).csv")

for index, row in test_df.iterrows():
    user_id = row['userId']
    rating_pattern = row['predicted_rating_pattern']

    # Extracting movie IDs and converting to integers
    movies = row['movies']
    movie_ids = [int(x) for x in movies if isinstance(x, int) or re.match(r'^\d+$', str(x))]

    # Create DataFrame with movie IDs
    top_movies_df = pd.DataFrame({'movie id': movie_ids})

    # Merge with movies_df
    merged_df = pd.merge(top_movies_df, movies_df, on='movie id')

    # Convert 'movie id' column to int
    top_movies_df['movie id'] = top_movies_df['movie id'].astype(int)

    # Get movie name and release date pairs
    movie_name_release_date_pairs = merged_df[['movie title', 'release date']].values.tolist()

    # Construct message
    message1 = f"""Please do not allow any influence of previously generated output in the current response.\
    Now, here is the rating-wise user preference pattern of user generated from user’s watched history: '{rating_pattern}'. Take your time, understand it and keep it in your memory properly.\
    Now, these are the new movies listed below with their release dates: '{movie_name_release_date_pairs}'.\
    Consider these movies, generate their information like Director, Casts, Movies Language, Year of Release, Genre and Tags, Cinematography, Visual Language, Lighting, Setting, Costume design and wardrobe, Movies Pacing, Editing, Entertainment Value, Theme, Emotional beats,  Tag for the Beginning and Climax, Message, Sound Design, and It” factor.\
    Finally compare and analyze these movies generated information with the given rating-wise user preference pattern.\
    After the analysis, predict the appropriate rating for each new movie based on the best match with the user preference pattern."""

    messages2 = [
        {"role": "system", "content":"""You are an expert in understanding rating-wise user preference pattern, generating new given movies information, matching the information with the users preference pattern and after analysis accurately predicting the appropriate ratings for the new movies for the user.\
        Please provide ratings for all the given movies based on the user preferences pattern analysis. \
        Then, extract the relevant data to use as arguments to pass into the provided function."""},
        {'role': 'user', 'content': message3},
        NewSample_response,
        {'role': 'user', 'content': message1}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages2,
        functions=[function],
        function_call={"name": "Give_rating_using_pattern_in_user_preference"},
        temperature=0.5,
        seed=42
    )

    predicted_rating = response["choices"][0]["message"]["function_call"]["arguments"]
    test_df.at[index, 'predicted_rating'] = predicted_rating

import re 
converted_ratings_list = []
index = 0
# Iterate over each row in the 'predicted_rating' column
for row in test_df['predicted_rating']:
    predicted_ratings_match = re.findall(r'"predicted_ratings": (\d+)', row)

    # Convert matched strings to integers
    predicted_ratings = [int(rating) for rating in predicted_ratings_match]

    # Append the predicted ratings to the converted ratings list

    # print("converted_ratings_list = ",predicted_ratings)
    test_df.at[index, 'predicted_rating_list'] = predicted_ratings
    index=index+1

def convert_imdb_list_to_float(imdb_list):
    float_list = []
    index = 1
    while index < len(imdb_list) - 1:
        try:
            num = int(imdb_list[index])
            index += 2
            dec = int(imdb_list[index])
            ratings_list = num + dec / 10.0
            float_list.append(float(ratings_list))
            index += 3
        except (IndexError, ValueError) as e:
            print("Error:", e)
            break
    return float_list

Final_result=test_df.copy()

for index, row in Final_result.iterrows():
    predicted_ratings_list = row['predicted_rating_list']
    actual_ratings_list = row['ratings']
    imdb_ratings_list = row['imdb_rating']
    # Check if both lists are non-empty
    if predicted_ratings_list and actual_ratings_list:
        # Convert JSON strings to lists if needed
        if isinstance(predicted_ratings_list, str):
            predicted_ratings_list = json.loads(predicted_ratings_list)
        if isinstance(actual_ratings_list, str):
            actual_ratings_list = json.loads(actual_ratings_list)

        # Find the common length
        common_length = min(len(predicted_ratings_list), len(actual_ratings_list))

        # Check if common_length is greater than zero
        Final_result.at[index, 'imdb_rating'] = convert_imdb_list_to_float(imdb_ratings_list)[:common_length]
        if common_length > 0:
            # Trim lists to the common length
            predicted_ratings_list = predicted_ratings_list[:common_length]
            actual_ratings_list = actual_ratings_list[:common_length]

            # Set values inplace
            Final_result.at[index, 'predicted_rating_list'] = predicted_ratings_list
            Final_result.at[index, 'ratings'] = actual_ratings_list

# Check for string values in the specified columns
string_columns = ['ratings', 'predicted_rating_list', 'imdb_rating']
rows_with_string_values = Final_result[Final_result[string_columns].applymap(lambda x: isinstance(x, str)).any(axis=1)]

# Drop rows with string values
cleaned_df = Final_result.drop(index=rows_with_string_values.index)


def check_and_remove_invalid_rows(df):
    invalid_rows = []
    for index, row in df.iterrows():
        # Check 'ratings' column
        if any(not isinstance(x, int) for x in row['ratings']):
            invalid_rows.append(index)
            print(f"Invalid 'ratings' at index {index}: {row['ratings']}")
            continue

        # Check 'predicted_rating_list' column
        if any(not isinstance(x, int) for x in row['predicted_rating_list']):
            invalid_rows.append(index)
            print(f"Invalid 'predicted_rating_list' at index {index}: {row['predicted_rating_list']}")
            continue

        # Check 'imdb_rating' column
        if any(not isinstance(x, float) for x in row['imdb_rating']):
            invalid_rows.append(index)
            print(f"Invalid 'imdb_rating' at index {index}: {row['imdb_rating']}")
            continue

    # Remove invalid rows
    df.drop(invalid_rows, inplace=True)
    return df

# Call the function to check and remove invalid rows
cleaned_df = check_and_remove_invalid_rows(Final_result)

actual_ratings = Final_result['ratings']
predicted_ratings = Final_result['predicted_rating_list']
imdb_list = Final_result['imdb_rating']


def order_indices(ratings, imdb_list):
    '''generate the sequece of indices as rank based on the rating and imdb rating values. First rank at first position'''
    # print("cnt = ")
    rating_array = np.array(ratings)
    # print("run = ",rating_array )

    unique_ratings = sorted(list(set(ratings)))[::-1]

    order_indics = []

    for ur in unique_ratings:
        rtng_indices = np.where(rating_array == ur)[0]


        imdb = [imdb_list[r_indx] for r_indx in rtng_indices]
        imdb = list(set(imdb))
        imdb = sorted(imdb, reverse=True)


        for i in imdb:
            for r_indx in rtng_indices:
                if imdb_list[r_indx] == i:
                    order_indics.append(r_indx)

    return order_indics

def dcg_at_K(actual_ratings, indeces_order, K):
    '''generates the dcg on a cut of value of K'''
    k, dcg = 0, 0
    K = K if K<=len(actual_ratings) else len(actual_ratings)

    for indx, rating_indx in enumerate(indeces_order):
        relavance = actual_ratings[rating_indx]
        numerator = pow(2,relavance) - 1
        denomirator = math.log2(indx+2)
        dcg += numerator/denomirator

        k += 1
        if k == K:
            return dcg

def mrr_at_k(ac_order_indc, pr_order_indc, actual_ratings, predicted_ratings, K):
    '''generates mrr@k'''
    k, rr = 0, 0
    K = K if K<=len(predicted_ratings) else len(predicted_ratings)
    for i, indx in enumerate(pr_order_indc):
        if k<K:
            if predicted_ratings[indx] >= 3 and actual_ratings[ac_order_indc[i]] >= 3:
                rr += 1/(i+1)
            elif predicted_ratings[indx] < 3 and actual_ratings[ac_order_indc[i]] < 3:
                rr += 1/(i+1)
        k += 1
    MRR_at_k = rr/K
    return(MRR_at_k)

def fcp_at_K(actual_ratings, pr_order_indc, K): #Fraction of Concordant Pairs at K
    K = K if K<=len(actual_ratings) else len(actual_ratings)
    concordant = 0

    top_k_indices = pr_order_indc[:K]

    for i, indx in enumerate(top_k_indices):
        if i < (K-1):
            nxt_top_k_indices = pr_order_indc[i+1:K]
            for nxt_indics in nxt_top_k_indices:
                concordant += 1 if actual_ratings[indx] >= actual_ratings[nxt_indics] else 0

    total_pair = math.comb(K, 2)
    fcp_at_k = concordant/total_pair
    return fcp_at_k

def ndcg_mrr_fpc_k(actual_ratings, predicted_ratings, imdb_list, K):
    '''takes the actual and predicted ratings of movies with imdb list and cutoff k. Returns the ndgc@K, mrr@k and fcp@k'''
    ac_order_indc = order_indices(actual_ratings, imdb_list)
    pr_order_indc = order_indices(predicted_ratings, imdb_list)

    # calculating ndcg
    ndcg_k = dcg_at_K(actual_ratings, pr_order_indc, K)/dcg_at_K(actual_ratings, ac_order_indc, K)

    # calculating mrr
    MRR_at_k = mrr_at_k(ac_order_indc, pr_order_indc, actual_ratings, predicted_ratings, K)

    # calculating fcp
    fcp_at_k = fcp_at_K(actual_ratings, pr_order_indc, K)


    return (ndcg_k, MRR_at_k, fcp_at_k)

import numpy as np

def MAE_MSE_RMSE_R2_at_K(actual_ratings, predicted_ratings, K):
    '''it returns MAE@K, MSE@K, RMSE@K and R2'''

    K = K if K<=len(actual_ratings) else len(actual_ratings)

    k, AE, SE, SS_res, SS_tot = 0, 0, 0, 0, 0
    ac_avg = np.average(np.array(actual_ratings))

    for indx, _ in enumerate(actual_ratings):
        if k < K:
            #  calculate AE
            AE += abs(actual_ratings[indx] - predicted_ratings[indx])

            # calculate SE
            error = (actual_ratings[indx] - predicted_ratings[indx])
            SE += pow(error, 2)

        # calculate R2
        error = (actual_ratings[indx] - predicted_ratings[indx])
        SS_res += pow(error, 2)
        error = actual_ratings[indx] - ac_avg
        SS_tot += pow(error, 2)

        k += 1

    MAE_at_K = AE/K
    MSE_at_K = SE/K
    RMSE_at_K = pow(MSE_at_K, 0.5)

    # Handling division by zero error for R2 calculation
    if SS_tot == 0:
        R2 = 0
    else:
        R2 = 1 - (SS_res/SS_tot)

    return MAE_at_K, MSE_at_K, RMSE_at_K, R2

k_values = range(3, 21)

results = []

for k in k_values:
  ndcg=0
  mrr=0
  fcp=0
  MAE=0
  MSE=0
  RMSE=0
  R2=0
  cnt=0
  for index, row in cleaned_df.iterrows():
      user_id = row['userId']
      movies = row['movies']
      ratings = row['ratings']
      predicted_rating_list = row['predicted_rating_list']
      imdb_rating = row['imdb_rating']

      # # Find the minimum length among the lists
      min_length = min(len(ratings), len(predicted_rating_list), len(imdb_rating))

      # Truncate the lists to the minimum length
      ratings = ratings[:min_length]
      predicted_rating_list = predicted_rating_list[:min_length]
      imdb_rating = imdb_rating[:min_length]

      # Save truncated lists back to the dataframe
      cleaned_df.at[index, 'ratings'] = ratings
      cleaned_df.at[index, 'predicted_rating_list'] = predicted_rating_list
      cleaned_df.at[index, 'imdb_rating'] = imdb_rating

      result1 = ndcg_mrr_fpc_k(ratings, predicted_rating_list, imdb_rating, k)
      result2 = MAE_MSE_RMSE_R2_at_K(ratings, predicted_rating_list, k)


      ndcg+=result1[0]
      mrr+=result1[1]
      fcp+=result1[2]
      MAE+=result2[0]
      MSE+=result2[1]
      RMSE+=result2[2]
      R2+=result2[3]
      cnt+=1

  ndcg/=cnt
  mrr/=cnt
  fcp/=cnt
  MAE/=cnt
  MSE/=cnt
  RMSE/=cnt
  R2/=cnt
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
df = pd.DataFrame(results)

df
