 # Fanbase Social Graph Analysis

## Quick Start Guide

### Step 1: API Setup

Before you begin, ensure you have access to the YouTube Data API and Spotify Web API. Follow these steps to get set up:

#### YouTube Data API
1. Navigate to the [Google Developers Console](https://console.developers.google.com/).
2. Create a new project, enable the YouTube Data API for it, and obtain an API key.
3. Set this API key as an environment variable named YOUTUBE_API_KEY on your machine:
    - **macOS/Linux**: Add `export YOUTUBE_API_KEY='your_api_key_here'` to your `.bash_profile`, `.bashrc`, or `.zshrc` file.
    - **Windows**: Add YOUTUBE_API_KEY as a system environment variable through the System Properties.

#### Spotify Web API
1. Visit the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
2. Log in or sign up, create an app, and note your client ID and client secret. Ensure these credentials are stored securely as they will be used in the scripts.

### Step 2: Initial Script Execution

To understand how the project works, execute these scripts for a sample artist:

1. **Fetch Comments**: Retrieves comments for the artist from YouTube.
    ```bash
    python fetch_comments.py 'artist_name' 'channel_id'
    ```

2. **Clean Comments**: Cleans up the fetched comments for analysis.
    ```bash
    python clean_comments.py 'artist_name'
    ```

3. **Baseline Analysis**: Analyzes the cleaned comments for insights.
    ```bash
    python baseline.py 'artist_name'
    ```

### Step 3: Main

The `main.py` script simplifies the process by chaining together the initial scripts from Step 2. It also identifies artists related to your main artist on Spotify and performs the required analysis for each.

```bash
python main.py <artist_name> <youtube_channel_id> <spotify_artist_id>
```

After execution, you will have 6 output folders containing analyses of the main artist and related artists. The script then compares each fanbase and returns a ranking based on similarity, which is compared with Spotify's ranking to evaluate our system.

Alternatively, if you want to run analysis on multiple artists, you can include a csv file as a single argument in the command line:

```bash
python main.py <artists_to_analyze.csv>
```

The csv must have headers `[name, Youtube_ID, Spotify_ID]` and each artist must follow this schema, with each artist having their own line in the csv file.

### Step 4: Analysis and Documentation

Review the output folders to document findings, investigate trends among the fanbases, and refine the algorithm for improved accuracy.

**Note**: Be cautious of channel IDs and links returned by the scripts; always verify the accuracy of the retrieved data.

## TODO List:

- **fetch_comments.py**:
  - Expand contractions.
  - Perform spell check.

- **baseline.py**:
  - Enhance graph aesthetics and data presentation.

- **main.py**:
  - Implement an IR-Evaluation function to compare fanbase rankings with Spotify's rankings.
  - Calculate metrics like Precision@K and display these.

- **compare_fanbase.py**:
  - Enhance functionality for deeper insights into fanbase social graphs.

## License

This project is licensed under the [MIT License](LICENSE).
