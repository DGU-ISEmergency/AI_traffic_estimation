from moviepy.video.io.VideoFileClip import VideoFileClip
import os


def split_video(input_file, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video file
    video = VideoFileClip(input_file)

    # Total duration in seconds
    total_seconds = int(video.duration)

    # Duration of each clip in seconds
    clip_duration = 60

    # Calculate number of full 1-minute clips
    num_clips = total_seconds // clip_duration

    # Iterate through each clip and create it
    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = start_time + clip_duration

        # Create subclip
        subclip = video.subclip(start_time, end_time)
        output_file = os.path.join(output_folder, f'output_{i + 1:03d}.mp4')

        # Write the subclip to file
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # Handle the remaining seconds if there are any
    remaining_seconds = total_seconds % clip_duration
    if remaining_seconds > 0:
        start_time = num_clips * clip_duration
        end_time = total_seconds

        # Create subclip
        subclip = video.subclip(start_time, end_time)
        output_file = os.path.join(output_folder, f'output_{num_clips + 1:03d}.mp4')

        # Write the subclip to file
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")

# Example usage
input_file = 'C:/Users/user/PycharmProjects/emergency/yolov7/inference/images/test1.mp4'
output_folder = 'output_clips'

split_video(input_file, output_folder)

