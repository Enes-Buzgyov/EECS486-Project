from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv


def get_video_ids(youtube, channel_id, max_videos=1):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_videos:
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=max_videos
            - len(video_ids),  # Adjust based on remaining videos to fetch
            pageToken=next_page_token,
            type="video",
            order="date",  # Most recent videos
        )
        response = request.execute()
        video_ids.extend([item["id"]["videoId"] for item in response.get("items", [])])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # Break the loop if no more videos are available

    return video_ids[:max_videos]


def get_comments(youtube, video_id):
    comments = []
    next_page_token = None

    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()
            comments.extend(
                [
                    item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    for item in response.get("items", [])
                ]
            )

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as e:
        print(f"Error fetching comments for video {video_id}: {e}")
    return comments


def save_comments_to_csv(comments, filename):
    # Ensure the output directory exists
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for comment in comments:
            writer.writerow([comment])


def fetch_and_save_comments(api_key, channel_id, artist_name):
    youtube = build("youtube", "v3", developerKey=api_key)

    video_ids = get_video_ids(youtube, channel_id, max_videos=1)

    all_comments = []
    for video_id in video_ids:
        comments = get_comments(youtube, video_id)
        all_comments.extend(comments)

    # Adjust filename to include the artist's name
    filename = f"data/{artist_name}_comments.csv"
    save_comments_to_csv(all_comments, filename)
    print(f"Comments for {artist_name} saved to {filename}")
    return len(all_comments)
