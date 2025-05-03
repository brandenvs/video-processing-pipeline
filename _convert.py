import subprocess
import os

def convert_mov_to_mp4(input_path, output_path=None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mp4"

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-strict", "experimental",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Conversion complete: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# Example usage:
convert_mov_to_mp4("id_test.MOV")
