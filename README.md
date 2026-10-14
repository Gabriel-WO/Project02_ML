## Project 02 - Machine Learning
### Overview
This is a machine learning project that uses YouTube API data (pulled 10.06.2026) to predict view counts for K-Pop videos from a variety of artists, including TWICE, BTS, Blackpink, and more. The model uses a random forest of 100 trees to predict views; the prediction is the average of all the tree outputs. 

Claude was used to help write code. 

### Files
* add_artists.py - merges data from pulled artists and label channels (which often jointly publish videos with the official group channel). 
* app.py - creates a Streamlit interface so the user can input a K-Pop video. 
* build_model - trains the final forest and builds the 80% ranges
* clean_data - removes duplicates and zero-view videos; tags each video as mv, short, behind, etc.
* main.py - empty placeholder
* make_plots.py - generates plots of model performance and correlations
* pull_data.py - goes through each channel's uploads and fetches their stats
* train_model.py - first experiment comparing taking the mean vs linear regression vs random forest
* tune_models.py - runs GridSearchCV on the forest and on HistGradientBoosting 
* plots - contains all the graphs, including for model comparisons
* artist_candidates.csv - list of artists that add_artists.py found but didn't approve
* artist_report.csv - stats of each approved channel
* feature_importance.csv - has each feature and its share of the forest’s importance
* kpop_videos_clean.csv - the file the model trains on; there are about 49,600 rows with the 14 features
* kpop_videos_merged.csv - raw data plus label channel duplicates removed
* kpop_videos_raw.csv - raw API download

### How to use
Run build_model.py to create the model. 

Open app.py and run it. It will open the interface locally on your browser. There, you can choose from a variety of artists, release dates and times, type of video (music video, short, behind the scenes, etc.), and video length. The user can also type in the title of the video. The model will predict the expected amount of views either since the inputted release date or after a certain amount of time, which the user can choose. 

### Limitations
YouTube API data doesn't show anything about the songs themselves. Variables like the style of music, the amount of Korean vs English lyrics, and viral trends aren't accounted for by the model; those variables may be better predictors of view count given that even small groups can sometimes have big hits. This is especially true in K-Pop, where certain music videos may be for major "comebacks" rather than normal songs, so it's view count likely is influenced by months of hype. 

Because the data only pulls from major K-Pop groups, the amount of music videos in particular is quite small, so the model struggles with estimating view counts for music videos. For mega-hits (with 200+ million views) the upper limit that the model gives is often closer to the actual view count. 

By taking the average of the random forest, extreme values are pulled towards the middle, causing the top fifth most viewed videos to be  underestimated. 
