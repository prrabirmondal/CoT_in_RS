import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
import math
import random
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from io import StringIO
from pprint import pprint
import json
import csv
import openai
import warnings
from sklearn.exceptions import ConvergenceWarning
from openai.openai_object import OpenAIObject


test_df = pd.read_csv("/content/test_0.8_0.2.csv")
test_df.drop(columns=['timestamp'],inplace=True)

train_df = pd.read_csv("/content/train_0.8_0.2.csv")
train_df.drop(columns=['timestamp'],inplace=True)

item_df = pd.read_csv("/content/items.csv")


item_df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1',
       'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1', 'IMDb URL',
       'Unnamed: 0.1.1.1.1.1.1','unknown', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40','video release date','No. of ratings',	'YT-Trailer ID','Runtime'],inplace=True)


def MakeDataFrame(new_df):
  grouped_df = new_df.groupby('userId')
  user_movie = grouped_df.count()
  user_movie = user_movie.sort_values(by='movieId', ascending=False)
  user_movie.reset_index(inplace=True)
  user_movie.drop(columns=['movieId','rating'], inplace=True)
  merged_df = pd.merge(new_df, user_movie, on='userId')
  group_df = merged_df.groupby('userId')
  new_data = []
  for user_id, group in group_df:
      movies = group['movieId'].tolist()
      ratings = group['rating'].tolist()
      new_data.append({'userId': user_id, 'movies': movies, 'ratings': ratings})

  new_df = pd.DataFrame(new_data)
  return new_df

test_df = MakeDataFrame(test_df)
train_df = MakeDataFrame(train_df)

openai_key = "API_KEY"
openai.api_key = openai_key


message2 = """Please do not allow any effect of previously generated output in the current response.    Input and output contains five  category of rating that is 1 star, 2 star, 3 star, 4 star, and 5 star.
    1 star indicates that the user least like to watch that movie, and a rating of 5 star means the user
    most liked to watch that movie. Ratings of 2, 3, and 4 follow the same notion accordingly. Now, here are the movies with their release dates, Cast, Director, and Genres, rated by this single user, categorized by their star ratings.
      Movies name with release date, Cast, Director, and Genres = '[('Casablanca (1942)', '01-Jan-1942','['Humphrey Bogart', 'Ingrid Bergman', 'Paul Henreid']','Michael Curtiz', '['Drama','Romance','War']']' and the user rated these movies to 5 star.
      Movies name with release date, Cast, Director, and Genres = '[('Four Weddings and a Funeral (1994)', '01-Jan-1994', '['Hugh Grant', 'Andie MacDowell', 'James Fleet']','Mike Newell','['Comedy', 'Drama', 'Romance']'), ('Sleepless in Seattle (1993)', '01-Jan-1993','['Tom Hanks', 'Meg Ryan', 'Ross Malinger']','Nora Ephron', '['Comedy', 'Drama', 'Romance']')]' and the user rated these movies to 4 star.
      Movies name with release date, Cast, Director, and Genres = '[('Mask, The (1994)', '01-Jan-1994', '['Jim Carrey', 'Cameron Diaz', 'Peter Riegert']','Chuck Russell', '['Comedy', 'Crime', 'Drama']'), ('Raising Arizona (1987)', '01-Jan-1987', '['Nicolas Cage', 'Holly Hunter', 'John Goodman']','['Joel Coen', 'Ethan Coen']', '['Comedy', 'Crime']')]' and the user rated these movies to 3 star.
      Movies name with release date, Cast, Director, and Genres = '[('Jungle2Jungle (1997)', '07-Mar-1997','['Tim Allen', 'Martin Short', 'JoBeth Williams']','John Pasquin', '['Comedy', 'Family']')]' and the user rated these movies to 2 star.

      You have to find rating wise pattern in user preference based on given movies' name, released date, Cast, Director, and Genres, use other factors also like movies’ nature, theme, storyline, cinematography, movie pace, It factor, tag, message etc.
      And that will help me in predicting the rating value of an new movie for the same single user in the future.

      STRICTLY FOLLOW: 1. Don't specify any movie name in the response.
      STRICT FOLLOW 2: If I do not provide which type of movies the user gives a 1-star rating to,
      then you don't need to provide any pattern in the user's preferences for 1-star ratings. If I do not provide
      which type of movies the user gives a 2-star rating to, then you don't need to provide any pattern in the user's
      preferences for 2-star ratings. If I do not provide which type of movies the user gives a 3-star rating to,
      then you don't need to provide any pattern in the user's preferences for 3-star ratings. If I do not provide
      which type of movies the user gives a 4-star rating to, then you don't need to provide any pattern in the
      user's preferences for 4-star ratings. If I do not provide which type of movies the user gives a 5-star rating to,
      then you don't need to provide any pattern in the user's preferences for 5-star ratings.""" 


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


def get_genre_text(row):
    genres = []
    if row['Action'] == 1:
        genres.append('Action')
    if row['Adventure'] == 1:
        genres.append('Adventure')
    if row['Animation'] == 1:
        genres.append('Animation')
    if row["Children's"] == 1:
        genres.append("Children's")
    if row['Comedy'] == 1:
        genres.append('Comedy')
    if row['Crime'] == 1:
        genres.append('Crime')
    if row['Documentary'] == 1:
        genres.append('Documentary')
    if row['Drama'] == 1:
        genres.append('Drama')
    if row['Fantasy'] == 1:
        genres.append('Fantasy')
    if row['Film-Noir'] == 1:
        genres.append('FilmNoir')
    if row['Horror'] == 1:
        genres.append('Horror')
    if row['Musical'] == 1:
        genres.append('Musical')
    if row['Mystery'] == 1:
        genres.append('Mystery')
    if row['Romance'] == 1:
        genres.append('Romance')
    if row['Sci-Fi'] == 1:
        genres.append('SciFi')
    if row['Thriller'] == 1:
        genres.append('Thriller')
    if row['War'] == 1:
        genres.append('War')
    if row['Western'] == 1:
        genres.append('Western')
    return ' , '.join(genres)


item_df1 = item_df.copy()

item_df1['Genres'] = item_df1.apply(get_genre_text, axis=1)



item_df1['Genres'] = item_df1['Genres'].apply(gensim.utils.simple_preprocess)

def convert_string_to_list(input_string):
    if isinstance(input_string, str):
        split_string = input_string.split('|')
        split_string = [name.strip() for name in split_string if name.strip()]
        return split_string
    else:
        return []

item_df1['Cast'] = item_df1['Cast'].apply(convert_string_to_list)

item_df1=item_df1[['movie_id', 'movie_title',	'release date',	'Cast',	'Director',	'Genres']]

def get_movie_name_meta_info_pairs(movie_df, item_df):

    merged_df = pd.merge(movie_df, item_df, on='movie_id')

    movie_name_release_date_pairs = merged_df.values.tolist()

    return movie_name_release_date_pairs



def generate_batches(dataframe, batch_size):
    num_batches = math.ceil(len(dataframe) / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataframe))
        yield dataframe.iloc[start_idx:end_idx]

cnt=0

for batch in generate_batches(train_df, 10):
    responses = []
    for index, row in batch.iterrows():
        user_id = row['userId']
        movies = row['movies']
        ratings = row['ratings']

        five_star_movie_id = []
        four_star_movie_id = []
        three_star_movie_id = []
        two_star_movie_id = []
        one_star_movie_id = []

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

        five_star_movie_df = five_star_movie_id
        four_star_movie_df = four_star_movie_id
        three_star_movie_df = three_star_movie_id
        two_star_movie_df = two_star_movie_id
        one_star_movie_df = one_star_movie_id

        five_star_movies = get_movie_name_meta_info_pairs(five_star_movie_df, item_df1) if not five_star_movie_df.empty else []
        four_star_movies = get_movie_name_meta_info_pairs(four_star_movie_df, item_df1) if not four_star_movie_df.empty else []
        three_star_movies = get_movie_name_meta_info_pairs(three_star_movie_df, item_df1) if not three_star_movie_df.empty else []
        two_star_movies = get_movie_name_meta_info_pairs(two_star_movie_df, item_df1) if not two_star_movie_df.empty else []
        one_star_movies = get_movie_name_meta_info_pairs(one_star_movie_df, item_df1) if not one_star_movie_df.empty else []

        message1 = f"""Please do not allow any effect of previously generated output in the current response.\
        Input and output contains five  category of rating that is 1 star, 2 star, 3 star, 4 star, and 5 star.\
        1 star indicates that the user least like to watch that movie, and a rating of 5 star means the user
        most liked to watch that movie. Ratings of 2, 3, and 4 follow the same notion accordingly.\
        Now, here are the movies with their release dates, Cast, Director, and Genres, rated by this single user, categorized by their star ratings.\
          Movies name with release date, Cast, Director, and Genres = '{five_star_movies}' and the user rated these movies to 5 star.\
          Movies name with release date, Cast, Director, and Genres = '{four_star_movies}' and the user rated these movies to 4 star.\
          Movies name with release date, Cast, Director, and Genres = '{three_star_movies}' and the user rated these movies to 3 star.\
          Movies name with release date, Cast, Director, and Genres = '{two_star_movies}' and the user rated these movies to 2 star.\
          Movies name with release date, Cast, Director, and Genres = '{one_star_movies}' and the user rated these movies to 1 star.\

          You have to find rating wise pattern in user preference based on given movies' name, released date, Cast, Director, and Genres and use other factors also like movies’ nature, theme, storyline, cinematography, movie pace, It factor, tag, message etc.
          And that will help me in predicting the rating value of an new movie for the same single user in the future.

          STRICTLY FOLLOW 1: Don't specify any movie name in the response.
          STRICT FOLLOW 2: If I do not provide which type of movies the user gives a 1-star rating to,
          then you don't need to provide any pattern in the user's preferences for 1-star ratings.\
          If I do not provide which type of movies the user gives a 2-star rating to, then you don't need to provide any
          pattern in the user's preferences for 2-star ratings. If I do not provide which type of movies the user
          gives a 3-star rating to, then you don't need to provide any pattern in the user's preferences for 3-star ratings.\
          If I do not provide which type of movies the user gives a 4-star rating to, then you don't need to provide any pattern in the
          user's preferences for 4-star ratings. If I do not provide which type of movies the user gives a 5-star rating to,
          then you don't need to provide any pattern in the user's preferences for 5-star ratings."""

        if not five_star_movies:
           message1 = message1.replace(f"Movies name with release date, Cast, Director, and Genres = '{five_star_movies}' and the user rated these movies to 5 star.", "")

        if not four_star_movies:
           message1 = message1.replace(f"Movies name with release date, Cast, Director, and Genres = '{four_star_movies}' and the user rated these movies to 4 star.", "")

        if not three_star_movies:
           message1 = message1.replace(f"Movies name with release date, Cast, Director, and Genres = '{three_star_movies}' and the user rated these movies to 3 star.", "")

        if not two_star_movies:
           message1 = message1.replace(f"Movies name with release date, Cast, Director, and Genres = '{two_star_movies}' and the user rated these movies to 2 star.", "")

        if not one_star_movies:
           message1 = message1.replace(f"Movies name with release date, Cast, Director, and Genres = '{one_star_movies}' and the user rated these movies to 1 star.", "")

        messages2 = [
            {"role": "system", "content": """You are an expert in identifying the pattern in user preference (meaning which type of movies the single
            user most likes to watch and which type of movies the user least like to watch) based on their rating values of movies. Please provide the rating wise pattern
            in user preference and then extract the relevant data to use as arguments to pass into the given function provided.""" },
            {'role': 'user', 'content': message2},
            sample_response,
            {'role': 'user', 'content': message1}]
    
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages2,
                functions=[function],
                function_call={"name": "pattern_in_user_preference"}, 
                temperature=0.5,
                seed=42
            )
        except openai.InvalidRequestError as e:
            print("Error processing batch:", e)
            continue  
        content = response["choices"][0]["message"]["function_call"]["arguments"]
        responses.append(content)
        train_df.loc[cnt, 'predicted_rating_pattern'] = content
        cnt+=1



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
              ['Lion King, The (1994)', '01-Jan-1994','Animation|Adventure|Drama','Matthew Broderick|Jeremy Irons|James Earl Jones','Roger Allers|Rob Minkoff'],
              ['Airheads (1994)', '01-Jan-1994','Comedy|Crime|Music','Brendan Fraser|Steve Buscemi|Adam Sandler','Michael Lehmann'],
              ['For Richer or Poorer (1997)', '01-Jan-1997','Comedy','Tim Allen|Kirstie Alley|Jay O. Sanders','Bryan Spicer'],
              ['Wolf (1994)', '01-Jan-1994',Drama|Horror|Romance','Jack Nicholson|Michelle Pfeiffer|James Spader','Mike Nichols'],
              ["Mary Shelley's Frankenstein (1994)", '01-Jan-1994','Drama|Horror|Romance','Robert De Niro|Kenneth Branagh|Helena Bonham Carter','Kenneth Branagh'],
              ['Nick of Time (1995)', '01-Jan-1995','Crime|Drama|Thriller','Johnny Depp|Christopher Walken|Courtney Chase','John Badham'],
              ['Richie Rich (1994)', '01-Jan-1994',Comedy|Family','Macaulay Culkin|John Larroquette|Edward Herrmann','Donald Petrie'],
              ['Around the World in 80 Days (1956)', '01-Jan-1956','Adventure|Comedy|Family','David Niven|Cantinflas|Shirley MacLaine','Michael Anderson']].\
              Predict the rating scores for all movies by considering the rating wise pattern in the user preference."""

train_df_predicted_rating_pattern = train_df[['userId','predicted_rating_pattern']]
test_df = pd.merge(test_df, train_df_predicted_rating_pattern, on='userId')

def generate_batches(dataframe, batch_size):
    num_batches = math.ceil(len(dataframe) / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataframe))
        yield dataframe.iloc[start_idx:end_idx]


for batch in generate_batches(test_df, 10):
    for index, row in batch.iterrows():
        user_id = row['userId']
        rating_pattern = row['predicted_rating_pattern']

        movies = row['movies']
        movie_ids = [int(x) for x in movies if isinstance(x, int) or re.match(r'^\d+$', str(x))]

        top_movies_df = pd.DataFrame({'movie_id': movie_ids})

        # Merge with movies_df
        merged_df = pd.merge(top_movies_df, item_df1, on='movie_id')

        # Convert 'movie id' column to int
        top_movies_df['movie_id'] = top_movies_df['movie_id'].astype(int)

        # Get movie name and release date pairs
        movie_name_release_date_pairs = merged_df[['movie_title', 'release date','Genres','Cast','Director']].values.tolist()

        # Construct message
        message1 = f"""Please do not allow any influence of previously generated output in the current response.\
        Now, here is the rating-wise user preference pattern of user generated from user’s watched history: '{rating_pattern}'. Take your time, understand it and keep it in your memory properly.\
        Now, these are the new movies listed below with their release dates, Genres, Cast and Director : '{movie_name_release_date_pairs}'.\
        Consider these movies, generate their information like Movies Language and Tags, Cinematography, Visual Language, Lighting, Setting, Costume design and wardrobe, Movies Pacing, Editing, Entertainment Value, Theme, Emotional beats,  Tag for the Beginning and Climax, Message, Sound Design, and It” factor.\
        Finally compare and analyze these movies generated information with the given rating-wise user preference pattern.\
        After the analysis, predict the appropriate rating for each new movie based on the best match with the user preference pattern."""

        messages2 = [
            {"role": "system", "content":"""You are an expert in understanding rating-wise user preference pattern, generating new given movies information, matching the information with the users preference pattern and after analysis accurately predicting the appropriate ratings for the new movies for the user.\
            Please provide ratings for all the given movies based on the user preferences pattern analysis. \
            Then, extract the relevant data to use as arguments to pass into the provided function."""},
            {'role': 'user', 'content': message3},
            NewSample_response,
            {'role': 'user', 'content': message1}]


        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages2,
                functions=[function],
                function_call={"name": "Give_rating_using_pattern_in_user_preference"},
                temperature=0.5,
                seed=42
            )
        except openai.InvalidRequestError as e:
            print("Error processing batch:", e)
            continue
        predicted_rating = response["choices"][0]["message"]["function_call"]["arguments"]
        test_df.at[index, 'predicted_rating'] = predicted_rating


converted_ratings_list = []
index = 0
# Iterate over each row in the 'predicted_rating' column
for row in test_df['predicted_rating']:
    try:
        predicted_ratings_match = re.findall(r'"predicted_ratings": (\d+)', row)

    # Convert matched strings to integers
        predicted_ratings = [int(rating) for rating in predicted_ratings_match]

        try:
            length = len(test_df.at[index, 'predicted_rating_list'])
            test_df.at[index, 'predicted_rating_list'] = predicted_ratings[0:length]
        except Exception as e:
            print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    index=index+1



temp_test_df = pd.read_csv("/content/test_0.8_0.2.csv")

temp_test_df.drop(columns=['timestamp'],inplace=True)

temp_item_df = pd.read_csv("/content/items.csv")

temp_item_df = temp_item_df[['movie_id','Rating']]

temp_item_df.rename(columns={'Rating':'imdb_rating','movie_id': 'movieId'}, inplace=True)

temp_test_df = pd.merge(temp_test_df, temp_item_df, on='movieId')


def MakeDataFrame_for_imdb(new_df):
  grouped_df = new_df.groupby('userId')
  user_movie = grouped_df.count()
  user_movie = user_movie.sort_values(by='movieId', ascending=False)
  user_movie.reset_index(inplace=True)
  user_movie.drop(columns=['movieId','rating','imdb_rating'], inplace=True)
  merged_df = pd.merge(new_df, user_movie, on='userId')
  group_df = merged_df.groupby('userId')
  new_data = []
  for user_id, group in group_df:
      movies = group['movieId'].tolist()
      ratings = group['rating'].tolist()
      imdb_ratings = group['imdb_rating'].tolist()
      new_data.append({'userId': user_id, 'movies': movies, 'ratings': ratings,'imdb_rating':imdb_ratings})

  new_df = pd.DataFrame(new_data)
  return new_df

temp_test_df = MakeDataFrame_for_imdb(temp_test_df)

test_df['imdb_rating']=temp_test_df['imdb_rating']


test_df.to_csv('final_result.csv', index=False)

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


cleaned_df=Final_result.copy()

def convert_ratings_to_list(ratings):
    return [int(x) for x in ratings.strip('[]').split(',')]

cleaned_df['ratings'] = cleaned_df['ratings'].apply(convert_ratings_to_list)
cleaned_df['predicted_rating_list'] = cleaned_df['predicted_rating_list'].apply(convert_ratings_to_list)

cleaned_df


actual_ratings = cleaned_df['ratings']
predicted_ratings = cleaned_df['predicted_rating_list']
imdb_list = cleaned_df['imdb_rating']


def order_indices(ratings, imdb_list):
    '''generate the sequece of indices as rank based on the rating and imdb rating values. First rank at first position'''

    rating_array = np.array(ratings)

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

def fcp_at_K(actual_ratings, pr_order_indc, K): 
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
        
    return dcg


def ndcg_mrr_fpc_k(actual_ratings, predicted_ratings, imdb_list, K):
    '''takes the actual and predicted ratings of movies with imdb list and cutoff k. Returns the ndgc@K, mrr@k and fcp@k'''
    ac_order_indc = order_indices(actual_ratings, imdb_list)
    pr_order_indc = order_indices(predicted_ratings, imdb_list)

    # calculating ndcg
    dcg_ac = dcg_at_K(actual_ratings, ac_order_indc, K)
    dcg_pr = dcg_at_K(actual_ratings, pr_order_indc, K)
    ndcg_k = dcg_pr / dcg_ac if dcg_ac != 0 else 0

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

df.to_csv('result_matric.csv', index=False)

df

