```markdown
# Movie Recommendation System

This project uses a dataset of movies, user ratings, and other metadata to generate movie recommendations. The recommendations are generated using a combination of TF-IDF vectorization, K-means clustering, and the OpenAI GPT-3.5 API.

## Datasets

The following datasets are used in this project:
- `genome-tags.csv`
- `genome-scores.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`
- `movies.csv`

## Prerequisites

Ensure you have the following Python packages installed:
- pandas
- scikit-learn
- requests
- openai

You can install the required packages using the following command:
```bash
pip install pandas scikit-learn requests openai
```

## API Key

Replace `YOUR_OPENAI_API_KEY` in the `main` function with your actual OpenAI API key.

## How to Run

1. Ensure you have all the required datasets in the specified paths.
2. Run the script by executing the following command:
```bash
python movie_recommendation_system.py
```

## Function Descriptions

### load_datasets
Loads the necessary datasets for the project.

### preprocess_movies
Preprocesses the movies dataset by removing movies with no genres listed and merging with the links dataset.

### cluster_movies
Performs TF-IDF vectorization on movie titles and genres and applies K-means clustering.

### select_movies_from_clusters
Selects one movie from each cluster.

### get_recommendations
Uses the OpenAI API to generate movie recommendations based on the selected movies.

### check_present
Checks if the predicted movie is present in the dataset.

### map_title_to_id
Maps predicted movie titles to their respective movie IDs.

### main
Main function to execute the workflow of loading data, preprocessing, clustering, selecting movies, getting recommendations, and calculating percentages.

## Output

The final output is saved to `output1.csv` which contains the recommended movies and their respective ratings.

## Notes

- The script calculates the percentage of users whose ratings for the recommended movies are within a threshold of 0.5 from their original ratings.
- Ensure the OpenAI API key is kept secure and not shared publicly.
