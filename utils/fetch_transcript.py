import re
import logging
from typing import List, Optional, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi

# For backward compatibility with existing code
from pytube import YouTube
from pytube.exceptions import VideoUnavailable, RegexMatchError, PytubeError, VideoPrivate, MembersOnly, LiveStreamError, RecordingUnavailable, VideoRegionBlocked

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> str:
    """
    Extract video ID from various YouTube URL formats.
    
    Args:
        url: YouTube URL or video ID
        
    Returns:
        Extracted video ID or original string if not found
    """
    # Handle None or empty string
    if not url or not isinstance(url, str):
        return ""
        
    # Handle direct video ID (11 characters)
    if len(url) == 11 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in url):
        return url
    
    # Handle full URLs
    patterns = [
        r'(?:youtube\.com/.*[?&]v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/watch\?.*v=)([^&\n?#]*)',
        r'(youtube\.com/shorts/)([^?&#/]*)',
        r'(?:v=|/v/|/)([0-9A-Za-z_-]{11}).*',
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, url, re.IGNORECASE)
        if matches:
            # Get the last matching group which should contain the ID
            video_id = matches.group(matches.lastindex or 1)
            # Clean up any extra characters
            video_id = re.sub(r'[^0-9A-Za-z_-]', '', video_id)
            if len(video_id) == 11:  # YouTube IDs are always 11 characters
                return video_id
    
    # If no pattern matched, return the original string stripped of whitespace
    return url.strip()

def get_transcript(video_id: str, language: str = "en") -> str:
    """
    Get transcript for a YouTube video using youtube-transcript-api.
    
    Args:
        video_id: YouTube video ID or URL
        language: Language code for captions (default: "en")
        
    Returns:
        str: Transcript text
        
    Raises:
        ValueError: If video ID is invalid or transcript cannot be retrieved
    """
    try:
        # Extract video ID if URL is provided
        video_id = extract_video_id(video_id)
        logger.info(f"Original input: {video_id}")
        logger.info(f"Extracted video ID: {video_id}")
        
        # Validate video ID
        if not video_id or len(video_id) != 11:
            raise ValueError("⚠️ Invalid YouTube video ID. Please check the URL or ID and try again.")
            
        # Check for valid characters in video ID
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        if not all(c in valid_chars for c in video_id):
            raise ValueError(f"⚠️ Invalid characters in video ID: {video_id}")
        
        logger.info(f"Fetching transcript for video: {video_id}")
        
        # Try to get transcript with fallback languages
        languages = [language, 'en', 'en-US', 'en-GB']
        languages = list(dict.fromkeys(languages))  # Remove duplicates
        
        # First check available transcripts
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_transcripts = [t.language_code for t in transcript_list]
            logger.info(f"Available transcripts: {available_transcripts}")
            
            # Try each preferred language in order
            for lang in languages:
                if lang in available_transcripts:
                    try:
                        logger.info(f"Trying to fetch {lang} transcript...")
                        transcript_data = YouTubeTranscriptApi.get_transcript(
                            video_id,
                            languages=[lang],
                            preserve_formatting=False
                        )
                        
                        if transcript_data and isinstance(transcript_data, list):
                            # Extract text from transcript segments
                            text_segments = [segment.get('text', '') for segment in transcript_data]
                            text = ' '.join(text_segments).strip()
                            if text:
                                logger.info(f"Successfully fetched transcript in {lang}")
                                return text
                    except Exception as e:
                        logger.warning(f"Could not get {lang} transcript: {str(e)}")
            
            # If no preferred language found, try any available transcript
            logger.info("Trying to fetch any available transcript...")
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    languages=available_transcripts,
                    preserve_formatting=False
                )
                
                if transcript_data and isinstance(transcript_data, list):
                    # Extract text from transcript segments
                    text_segments = [segment.get('text', '') for segment in transcript_data]
                    text = ' '.join(text_segments).strip()
                    if text:
                        logger.info("Successfully fetched transcript in available language")
                        return text
                
                raise ValueError("No transcript data available")
                    
            except Exception as e:
                logger.warning(f"Could not get any transcript: {str(e)}")
                
            if available_transcripts:
                raise ValueError(f"Failed to fetch transcript. Available languages: {', '.join(available_transcripts)}")
            else:
                raise ValueError("No transcripts are available for this video")
                
        except Exception as e:
            logger.warning(f"Error listing transcripts: {str(e)}")
            # If listing fails, try direct fetch as fallback
            try:
                logger.info("Trying direct transcript fetch...")
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                if transcript_data and isinstance(transcript_data, list):
                    text_segments = [segment.get('text', '') for segment in transcript_data]
                    text = ' '.join(text_segments).strip()
                    if text:
                        return text
                raise ValueError("No transcript data available")
            except Exception as e:
                logger.error(f"Direct fetch failed: {str(e)}")
                raise ValueError(f"Could not fetch transcript: {str(e)}")
            
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Error in get_transcript: {error_msg}", exc_info=True)
        
        if "no transcript" in error_msg or "no captions" in error_msg:
            raise ValueError("❌ No transcript is available for this video. Please try another video with captions.")
        elif "video unavailable" in error_msg:
            raise ValueError("❌ This video is not available. It may have been removed or made private.")
        elif "members only" in error_msg:
            raise ValueError("❌ This video is for members only.")
        elif "private" in error_msg:
            raise ValueError("❌ This is a private video. Only the uploader can access it.")
        elif "too many requests" in error_msg or "429" in error_msg:
            raise ValueError("⏳ Too many requests. Please wait a moment and try again.")
        elif "400" in error_msg:
            raise ValueError("❌ Bad request. The video ID might be invalid.")
        elif "404" in error_msg:
            raise ValueError("❌ Video not found. Please check the video ID or URL and try again.")
        elif "403" in error_msg:
            raise ValueError("❌ Access denied. The video might have viewing restrictions.")
        elif "age restricted" in error_msg:
            raise ValueError("❌ Age-restricted videos are not supported")
        else:
            raise ValueError(f"❌ Could not fetch transcript: {str(e)}")
