"""
Utility functions for Qwen VL image and video processing
"""

def process_vision_info(messages):
    """
    Process vision information from messages
    
    Args:
        messages (list): List of message dictionaries
        
    Returns:
        tuple: (image_inputs, video_inputs)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if 'content' in message and isinstance(message['content'], list):
            for content_item in message['content']:
                if isinstance(content_item, dict):
                    # Process image items
                    if content_item.get('type') == 'image' and 'image' in content_item:
                        image_path = content_item['image']
                        if image_path.startswith('file://'):
                            image_path = image_path[7:]  # Remove 'file://' prefix
                        image_inputs.append(image_path)
                    
                    # Process video items
                    if content_item.get('type') == 'video' and 'video' in content_item:
                        video_path = content_item['video']
                        if video_path.startswith('file://'):
                            video_path = video_path[7:]  # Remove 'file://' prefix
                        video_inputs.append(video_path)
    
    return image_inputs, video_inputs