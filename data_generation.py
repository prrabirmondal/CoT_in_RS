import json
import pandas as pd
import numpy as np
import os
import re # For a more robust marker finding, if needed
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

if not os.path.exists("./hf_models"):
    os.makedirs("./hf_models", exist_ok=True)

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
cache_dir = "./hf_models"  # Ensure this exists and is writable


# ratings = pd.read_csv("/DATA/punitsingh1801/prabir/prabir/python_program/Earl_workshop/GoT/data/ratings.dat", sep="::", engine="python")
ratings = pd.read_csv(r'./ratings.dat', sep="::", engine="python")
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir = "./hf_models", resume_download=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    load_in_8bit=True,
    cache_dir = "./hf_models",
    resume_download=True
)


# # print(f"total ratings = {len(ratings)}")
# # ratings.head()


def read_json_file(filepath):
    """Reads a JSON file and returns the data as a Python object (dict or list)."""
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None

def check_movie_ratings_in_clusters(filtered_clustered_users_df, ratings_df):
    """
    Checks for each user ID, for each cluster ID, and for each movie in that cluster,
    if the user has rated the movie.

    Args:
        filtered_clustered_users_df (pd.DataFrame): DataFrame with user_id, clusterID,
                                                   and a list of movies.
                                                   Example:
                                                   user_id clusterID                                             movies
                                                   0            1         0  [3186, 1270, 1022, 2340, 1207, 3105, 150, 1028...
        ratings_df (pd.DataFrame): DataFrame with user_id, movie_id, rating, and timestamp.
                                   Example:
                                   user_id  movie_id  rating  timestamp
                                   0              1       661       3  978302109

    Returns:
        pd.DataFrame: A DataFrame with 'user_id', 'clusterID', 'movie_id', and 'rated' columns,
                      where 'rated' is True if the user has rated the movie, False otherwise.
    """
    # Explode the 'movies' column to create a row for each movie in each cluster
    # This transforms the list of movies into individual rows for easier merging.
    exploded_users = filtered_clustered_users_df.explode('movies')

    # Rename the 'movies' column to 'movie_id' to match the column name in the ratings_df.
    # This consistency is crucial for performing an accurate merge.
    exploded_users = exploded_users.rename(columns={'movies': 'movie_id'})
    exploded_users['movie_id'] = exploded_users['movie_id'].apply(lambda x: int(x))

    # Perform a left merge between the exploded_users DataFrame and a subset of the ratings_df.
    # The merge is based on 'user_id' and 'movie_id'.
    # 'how='left'' ensures that all rows from 'exploded_users' are kept,
    # and matching rows from 'ratings_df' are included if they exist.
    # 'indicator=True' adds a special column '_merge' which indicates the source of each row:
    # 'left_only' for rows only in 'exploded_users', 'right_only' for rows only in 'ratings_df' (not applicable with left join on these keys),
    # and 'both' for rows present in both DataFrames.
    merged_df = pd.merge(
        exploded_users,
        ratings_df[['user_id', 'movie_id']], # Only select user_id and movie_id from ratings_df to check for existence
        on=['user_id', 'movie_id'],
        how='left',
        indicator=True
    )

    # Create the 'rated' column.
    # A movie is considered 'rated' if the corresponding row in 'merged_df' came from 'both'
    # DataFrames during the merge, meaning there was a matching entry in the 'ratings_df'.
    merged_df['rated'] = merged_df['_merge'] == 'both'

    # Select and reorder the columns for the final output DataFrame.
    # The '_merge' column is no longer needed after creating 'rated'.
    final_df = merged_df[['user_id', 'clusterID', 'movie_id', 'rated']]

    return final_df

users_rated = check_movie_ratings_in_clusters(filtered_clustered_users, filtered_ratings)
# print(df1)

# for ix,i in filtered_clustered_users.iterrows():
# 	superset = ratings[ratings['user_id']==i['user_id']].movie_id.values
# 	if len(set(i['movies']).intersection(superset)) < 0:
# 	    print(i['user_id'],set(i['movies']).intersection(superset))


# filtered_clustered_users = filtered_clustered_users.loc[0:]

def get_age_map(age_code):
    age_map = {
                '1':  "Under 18",
	            '18':  "18-24",
	            '25':  "25-34",
	            '35':  "35-44",
	            '45':  "45-49",
	            '50':  "50-55",
	            '56':  "56+"
            }
    return age_map.get(age_code)

def get_occupation_map(occupation_code):
    occupation_map = {
        
                        '0': "other or not specified",
                        '1': "academic/educator",
                        '2': "artist",
                        '3': "clerical/admin",
                        '4': "college/grad student",
                        '5': "customer service",
                        '6': "doctor/health care",
                        '7': "executive/managerial",
                        '8': "farmer",
                        '9': "homemaker",
                        '10': "K-12 student",
                        '11': "lawyer",
                        '12': "programmer",
                        '13': "retired",
                        '14': "sales/marketing",
                        '15': "scientist",
                        '16': "self-employed",
                        '17': "technician/engineer",
                        '18': "tradesman/craftsman",
                        '19': "unemployed",
                        '20': "writer"
                    }
    return occupation_map.get(occupation_code)


def get_user_movie_details(user_id, movie_id):
    """
    Retrieves user demographic details and associated movie details.

    Args:
        user_id (int): The ID of the user.

    Returns:
        dict: A dictionary containing user's gender, age, occupation,
              and a list of dictionaries for each movie with 'movie_name' and 'genres'.
              Returns None if user_id is not found.
    """
    user_id_str = str(user_id) # Convert to string to match dictionary keys
    
    if user_id not in filtered_ratings['user_id']:
        print(f"User ID {user_id} not found in user data.")
        return None

    # Get user demographic details
    user_demographics = user_file[user_id_str] # user id is a string in the json file
    gender = user_demographics.get('Gender')
    age_code = user_demographics.get('Age')
    age = get_age_map(age_code)
    occupation_code = user_demographics.get('Occupation')
    occupation = get_occupation_map(occupation_code)

    # Get movie IDs for the user
    movie_details = all_movies[all_movies['movie_id']==movie_id]
    title = movie_details['title'].values[0]
    story_line = movie_details['storyline'].values[0]
    directors = movie_details['directors'].values[0]
    cast = []
    for i in movie_details['cast'].values:
        for j in i:
            for k in eval(j):
                cast.append(k.split("(")[0].strip())
    # cast = movie_details['cast'].values[0]
    genres = movie_details['genres'].values[0]
    writers = movie_details['writers'].values[0]
    
    
    return {
        'gender': gender,
        'age': age,
        'occupation': occupation,
        'title': title,
        'story_line' : story_line,
        'directors' : directors,
        'cast' : cast,
        'genres' : genres,
        'writers' : writers 
    }

def create_user_movie_prompt(gender, age_range, occupation, title, storyline, directors, cast, genres, writers,rating):
    """
    Generates a formatted prompt string with user and movie details using f-strings.

    Args:
        gender (str): The gender of the user (e.g., "Male", "Female").
        age_range (str): The age range of the user (e.g., "18 to 24", "35-44").
        occupation (str): The occupation of the user.
        movie_name (str): The title of the movie.
        genres (str): The genres of the movie, pipe-separated.
        storyline (str): The storyline/synopsis of the movie.

    Returns:
        str: A formatted string containing the user and movie details.
    """
    prompt = f"""
            User Details: 
            gender:  {gender}, age: {age_range}, occupation: {occupation}

            Movie Details:
            movie name: {title}, 
            storyline: {storyline},
            directors : {directors},
            cast : {cast}, 
            genres : {genres}
            writers : {writers}

            Rating Given to the Movie: {rating} where 5 means the user likes the movie the most and 1 means dislikes the movie."

            Here are the user details, movie details and the rating given to the movie by the user have been mentioned. Now please read the information and finally answer the following question by covering all the steps,
            Question: What is the preference of the user?
            Steps to follow to answer my question:
            1. Please analyse the movie features by considering the user's data and find the movie features that suit the user's gender, occupation, and age.
            2. As per the user's age, occupation, please analyse and find out what might be the movie features for which the user has rated {ratings} out of 5? 
            3. As per the user details (age, occupation) please analyse and find out what might be movie features for which the user has not rated 5 out of 5 to the movie.
            4. As per the user details, movie details, the rating given to the movie  and the analysis from the above two points, please analyse and find out the user preference by concerning both user and movie details. Please analyse user preference from both sides like what he likes and what he dislikes.
            5. A higher/ lower rating cannot be a reason for like or dislike. Don't include rating as a reason for like or dislike
            6. Now, finally answer me. Please highlight all the points that describe user preferences in both positive and negative aspects. 

            Answer the above question by following the steps  in the following dictionary format,
            
            {{
                "What the user likes":{{
                    "Feature 1": "one line description",
                    "Feature 2": "one-line description",
                    "Feature 3": "one-line description",
                    .
                    .
                    "Feature n": "one-line description"
                    }},

                "What the user dislikes":{{
                    "Feature 1": "one-line description",
                    "Feature 2": "one-line description",
                    "Feature 3": "one-line description",
                    .
                    .
                    "Feature n": "one-line description"
                    }}
            }}

            Add as many features as required. Please give the appropriate feature name instead of "Feature 1". Use a maximum of two words in the feature name. Only share the final answer.
            Please ensure the reponse is made independently and is not dependent on the previous response.
            """
    return prompt


def query_qwen(prompt):
    full = f"<s>[INST] You are an assisstant that analyzes user movie ratings. {prompt.strip()} [/INST]</s>"
    # out = generator(full, max_new_tokens=500, temperature=0.3, top_p=0.5)
    # return out[0]["generated_text"]
    messages = [
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    # print("content:", content)
    return thinking_content, content



def is_valid_json_string(text_response):
    """
    Checks if a given string contains a valid JSON object embedded within it,
    typically from a markdown-formatted response. It prioritizes extracting
    JSON from '```json\n' and '\n```' markers.

    Args:
        text_response (str): The string response, potentially containing JSON.

    Returns:
        tuple: A tuple containing (True, parsed_json_object) if valid JSON is found,
               or (False, error_message) if not.
    """
    json_str_raw = None
    start_marker = '```json' # Removed '\n' to be more flexible with newline positions
    end_marker = '```'

    # Use re.DOTALL to allow . to match newlines
    # This regex looks for '```json' followed by anything (non-greedy), then '```'
    # It captures the content between them.
    match = re.search(f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", text_response, re.DOTALL)

    if match:
        json_str_raw = match.group(1).strip() # Extract and strip leading/trailing whitespace/newlines
    else:
        # If markdown markers are not found, assume the entire stripped string
        # might be the JSON. This is less robust but handles simple cases.
        json_str_raw = text_response.strip()
        # Optional: Add a check here if you *insist* on markdown formatting
        # if not (json_str_raw.startswith('{') or json_str_raw.startswith('[')):
        #     return False, "No JSON markers found and string does not start with '{' or '['."

    if not json_str_raw: # Handle cases where extraction results in an empty string
        return False, "No content extracted for JSON parsing."

    # 2. Parse it using json.loads()
    try:
        parsed_json = json.loads(json_str_raw)
        return True, parsed_json
    except json.JSONDecodeError as e:
        return False, f"JSON decoding error: {e}. Raw string: '{json_str_raw}'"
    except Exception as e:
        # Catch any other unexpected errors during parsing
        return False, f"An unexpected error occurred: {e}. Raw string: '{json_str_raw}'"

# All Users

user_file = './users.json'
user_file = read_json_file(user_file)
all_users = []
for u_id in list(user_file.keys()):
    dic = {"user_id": u_id, **user_file[u_id]}
    all_users.append(dic)
    
all_users = pd.DataFrame(all_users)
print(f"total users = {len(all_users)}")


# All Movies

movie_file = './movies.json'
movie_file = read_json_file(movie_file)


all_movies = []
for m_id in list(movie_file.keys()):
    dic = {"movie_id": m_id, **movie_file[m_id]}
    all_movies.append(dic)
    
all_movies = pd.DataFrame(all_movies)

print(f"total movies = {len(all_movies)}")


## All Movies

# clusters_file = '/home/ubuntu/prosenjit/GoT/CoT_in_RS-got/data/clusters.json'
clusters_file = './master_cluster_data.json'
clusters_file = read_json_file(clusters_file)

all_cluster = []
for user in clusters_file:
    keys = list(user.keys())
    user_id = int(user[keys[0]])
    user_clusters = user[keys[1]]

    for user_cluster in user_clusters:
       
        cluster = {"user_id": user_id, **user_cluster}
        all_cluster.append(cluster)
        
all_cluster = pd.DataFrame(all_cluster)
all_cluster = all_cluster.sort_values(by=["user_id", "clusterID"])

print(f"total user_clusters = {len(all_cluster)}")

filtered_ratings = pd.read_csv("./filtered_ratings.csv")
filtered_clustered_users = all_cluster[all_cluster['user_id'].isin(filtered_ratings.user_id)]



final_df = pd.DataFrame(columns=['user_id', 'cluster_id', 'movie_id', 'likes', 'dislikes'])
for uid in filtered_clustered_users['user_id']:
    for cluster_id in filtered_clustered_users[filtered_clustered_users['user_id']==uid]['clusterID']:
        for mid in filtered_clustered_users[(filtered_clustered_users['user_id']==uid) & (filtered_clustered_users['clusterID']==cluster_id)]['movies'].values[0]:
            if users_rated[(users_rated['user_id']==uid) & (users_rated['clusterID']==cluster_id)].rated.sum() > 10:
                resp = get_user_movie_details(uid, mid)
                gender, age, occupation, title, storyline, directors, cast, genres, writers = \
                    resp['gender'], resp['age'], resp['occupation'], resp['title'], resp['story_line'], resp['directors'], resp['cast'], resp['genres'], resp['writers']
                filtered_rating = filtered_ratings[(filtered_ratings['user_id']== uid) & (filtered_ratings['movie_id']==int(mid))]['rating']
                if filtered_rating.empty:
                    print(f"skipping for absence of rating for user {uid} cluster {cluster_id} and movie {mid}")
                    continue
                filtered_rating = str(filtered_rating.values[0])
                prompt = create_user_movie_prompt(gender, age, occupation, title, storyline, directors, cast, genres, writers, filtered_rating)
                if not os.path.exists(f'final_output/{model_name}/{uid}_{cluster_id}_{mid}.csv'):
                    _, response= query_qwen(prompt)
                    _, parsed_json = is_valid_json_string(response)
                    data = {
                        'user_id': uid,
                        'cluster_id': cluster_id,
                        'movie_id': mid,
                        'likes': parsed_json['What the user likes'],
                        'dislikes': parsed_json['What the user dislikes']

                    }
                    final_df.loc[len(final_df)] = data
                    df = pd.DataFrame([data.values()], columns=['user_id', 'cluster_id', 'movie_id', 'likes', 'dislikes'])
                    df.to_csv(f'final_output/{model_name}/{uid}_{cluster_id}_{mid}.csv', index=False)
                    print(f"Processed/{uid}_{cluster_id}_{mid}.csv", len(final_df))
                else:
                    print(f"skipped due to less than 10 rated movies for: {uid} with cluster: {cluster_id}")
final_df.to_csv(f'final_output/full_data.csv')
print("Full data Saved successfully")

            
