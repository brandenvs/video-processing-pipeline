import audio_to_text as att
import db
import os

# TODO: Use relative path and dynamic determination of system absolute path

def main():
    audio_file = att.extract_audio_for_transcription("X:\\development\\adpApp\\data\\good_text_vid.mp4")
    
    # Hard-coded configuration:
    model_name = "base"              # Options: tiny, base, small, medium, large - chose based on your amount of processing power available

    if not os.path.exists(audio_file):
        print(f"Error: The audio file '{audio_file}' does not exist.")
        return

    formatted_original,  analysis = att.direct_transcription(audio_file, model_name)
    
    db.write_to_db(formatted_original, analysis)

if __name__ == "__main__":
    main()