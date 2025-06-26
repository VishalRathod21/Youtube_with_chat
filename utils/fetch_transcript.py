"""
YouTube Transcript Fetcher with Proxy Support

This module provides functionality to fetch YouTube video transcripts with support for proxies,
retry mechanisms, and comprehensive error handling.
"""
import re
import logging
import random
import os
import time
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
    RequestBlocked,
    IpBlocked,
    AgeRestricted
)
from fake_useragent import UserAgent
from dotenv import load_dotenv

# Custom exception classes
class TranscriptError(Exception):
    """Base exception for all transcript-related errors."""
    pass

class VideoAccessError(TranscriptError):
    """Raised when there's an issue accessing the video."""
    pass

class TranscriptNotAvailable(TranscriptError):
    """Raised when no transcript is available for the video."""
    pass

class RateLimitExceeded(TranscriptError):
    """Raised when YouTube rate limits are encountered."""
    pass

class TranscriptFormat:
    """Constants for transcript formats."""
    TEXT = 'text'
    JSON = 'json'
    LINES = 'lines'

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> str:
    """Extract and validate a YouTube video ID from various URL formats.
    
    Args:
        url: A YouTube URL or video ID string.
        
    Returns:
        str: The extracted 11-character video ID, or the original string if no ID found.
        
    Example:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not url or not isinstance(url, str):
        return ""
        
    # Check if already a video ID
    if (len(url) == 11 and 
        all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in url)):
        return url
    
    patterns = [
        # Standard URLs
        r'(?:youtube\.com/.*[?&]v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/watch\?.*v=)([^&\n?#]*)',
        # Shorts URLs
        r'(youtube\.com/shorts/)([^?&#/]*)',
        # Direct video ID
        r'(?:v=|/v/|/)([0-9A-Za-z_-]{11}).*',
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, url, re.IGNORECASE)
        if matches:
            video_id = matches.group(matches.lastindex or 1)
            video_id = re.sub(r'[^0-9A-Za-z_-]', '', video_id)
            if len(video_id) == 11:
                logger.debug(f"Extracted video ID: {video_id} from URL: {url}")
                return video_id
    
    logger.warning(f"Could not extract video ID from URL: {url}")
    return url.strip()

def get_proxy_config() -> Optional[Dict[str, str]]:
    """Get proxy configuration from environment variables.
    
    Reads HTTP_PROXY and HTTPS_PROXY environment variables and returns
    a dictionary suitable for the requests library.
    
    Returns:
        Optional[Dict[str, str]]: Dictionary with proxy configuration or None if not set.
        
    Example:
        >>> os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
        >>> get_proxy_config()
        {'http': 'http://proxy.example.com:8080'}
    """
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    
    if not (http_proxy or https_proxy):
        logger.debug("No proxy configuration found in environment variables")
        return None
        
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
        logger.debug(f"Using HTTP proxy: {http_proxy}")
    if https_proxy:
        proxies['https'] = https_proxy
        logger.debug(f"Using HTTPS proxy: {https_proxy}")
        
    return proxies

def get_random_user_agent() -> str:
    """Generate a random user agent for requests.
    
    Returns:
        str: A random user agent string, falls back to a default Chrome user agent if generation fails.
        
    Example:
        >>> get_random_user_agent()
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ...'
    """
    DEFAULT_USER_AGENT = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/91.0.4472.124 Safari/537.36'
    )
    
    try:
        ua = UserAgent()
        user_agent = ua.random
        logger.debug(f"Generated random user agent: {user_agent}")
        return user_agent
    except Exception as e:
        logger.warning(f"Error generating random user agent: {e}. Using default.")
        return DEFAULT_USER_AGENT

def fetch_transcript_with_retry(
    video_id: str, 
    languages: List[str], 
    proxies: Optional[Dict[str, str]], 
    headers: Dict[str, str], 
    max_retries: int = 3,
    format: str = 'text'
) -> Union[str, List[Dict], None]:
    """Fetch transcript with retry logic and error handling.
    
    Args:
        video_id: YouTube video ID
        languages: List of language codes to try (in order of preference)
        proxies: Optional proxy configuration
        headers: Request headers including User-Agent
        max_retries: Maximum number of retry attempts
        format: Output format ('text', 'json', or 'lines')
        
    Returns:
        Union[str, List[Dict], None]: The transcript in the requested format,
        or None if all attempts fail
        
    Raises:
        TranscriptError: For errors that shouldn't trigger a retry
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching transcript (attempt {attempt + 1}/{max_retries})")
            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id=video_id,
                languages=languages or None,
                preserve_formatting=False,
                proxies=proxies
            )
            
            if not transcript_data or not isinstance(transcript_data, list):
                raise TranscriptError("No transcript data returned")
            
            logger.info(f"Successfully fetched transcript for video {video_id}")
            
            # Format the response based on requested format
            if format == 'json':
                return transcript_data
            elif format == 'lines':
                return [{
                    'text': segment.get('text', '').strip(),
                    'start': segment.get('start', 0),
                    'duration': segment.get('duration', 0)
                } for segment in transcript_data]
            else:  # text format (default)
                return ' '.join(segment.get('text', '').strip() 
                             for segment in transcript_data)
            
        except (VideoUnavailable, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
            raise TranscriptNotAvailable(str(e))
        except (RequestBlocked, IpBlocked) as e:
            raise RateLimitExceeded("YouTube request blocked. Please try again later or use a proxy.") from e
        except AgeRestricted as e:
            raise VideoAccessError("Age-restricted video. Cannot fetch transcript.") from e
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
                continue
    
    logger.error(
        f"Failed to fetch transcript after {max_retries} attempts. "
        f"Last error: {str(last_error)}"
    )
    return None

def get_transcript(
    video_id: str, 
    language: str = "en", 
    max_retries: int = 3,
    format: str = 'text'
) -> Union[str, List[Dict]]:
    """Get transcript for a YouTube video with comprehensive error handling.
    
    Args:
        video_id: YouTube video URL or ID
        language: Preferred language code (default: 'en')
        max_retries: Maximum number of retry attempts (default: 3)
        format: Output format ('text', 'json', or 'lines')
        
    Returns:
        Union[str, List[Dict]]: The transcript in the requested format
        
    Raises:
        ValueError: For invalid inputs or when transcript cannot be retrieved
        TranscriptError: For other transcript-related errors
    """
    # Extract and validate video ID
    video_id = extract_video_id(video_id)
    logger.info(f"Processing video ID: {video_id}")
    
    if not video_id or len(video_id) != 11:
        raise ValueError("Invalid YouTube video ID. Please check the URL or ID and try again.")
        
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    if not all(c in valid_chars for c in video_id):
        raise ValueError(f"Invalid characters in video ID: {video_id}")
    
    # Get proxy and headers
    proxies = get_proxy_config()
    headers = {'User-Agent': get_random_user_agent()}
    
    try:
        # Try to get available transcripts
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(
                video_id, 
                proxies=proxies
            )
            available_transcripts = [t.language_code for t in transcript_list]
            logger.info(f"Available transcripts: {available_transcripts}")
            
            # Try preferred languages in order
            languages = list(dict.fromkeys([language, 'en', 'en-US', 'en-GB']))
            
            for lang in languages:
                if lang in available_transcripts:
                    logger.info(f"Trying to fetch {lang} transcript...")
                    transcript = fetch_transcript_with_retry(
                        video_id, 
                        [lang], 
                        proxies, 
                        headers, 
                        max_retries,
                        format
                    )
                    if transcript:
                        return transcript
            
            # If no preferred language found, try any available transcript
            logger.info("Trying to fetch any available transcript...")
            transcript = fetch_transcript_with_retry(
                video_id, 
                available_transcripts, 
                proxies, 
                headers, 
                max_retries,
                format
            )
            if transcript:
                return transcript
                
            raise TranscriptNotAvailable("No transcript data available in any supported language")
            
        except (TranscriptNotAvailable, RateLimitExceeded) as e:
            # Re-raise these specific exceptions
            raise
            
        except Exception as e:
            logger.warning(f"Error listing transcripts: {str(e)}")
            
            # Try direct fetch as last resort
            logger.info("Trying direct transcript fetch...")
            transcript = fetch_transcript_with_retry(
                video_id, 
                [], 
                proxies, 
                headers, 
                max_retries,
                format
            )
            if transcript:
                return transcript
                
            # Try without proxy if one was configured
            if proxies:
                logger.warning("Proxy may be causing issues. Trying without proxy...")
                transcript = fetch_transcript_with_retry(
                    video_id, 
                    [], 
                    None,  # No proxy
                    headers, 
                    max_retries,
                    format
                )
                if transcript:
                    return transcript
            
            raise TranscriptError("Could not fetch transcript after multiple attempts")
            
    except TranscriptNotAvailable as e:
        raise ValueError(f"❌ {str(e)}")
    except RateLimitExceeded as e:
        raise ValueError("⏳ Too many requests. Please wait a moment and try again.")
    except VideoUnavailable as e:
        raise ValueError("❌ This video is not available. It may have been removed or made private.")
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Error in get_transcript: {error_msg}", exc_info=True)
        
        # Map specific error messages to user-friendly responses
        if any(msg in error_msg for msg in ["no transcript", "no captions"]):
            raise ValueError("❌ No transcript is available for this video. Please try another video with captions.")
        elif "members only" in error_msg:
            raise ValueError("❌ This video is for members only.")
        elif "private" in error_msg:
            raise ValueError("❌ This is a private video. Only the uploader can access it.")
        elif any(msg in error_msg for msg in ["400", "bad request"]):
            raise ValueError("❌ Bad request. The video ID might be invalid.")
        elif "404" in error_msg:
            raise ValueError("❌ Video not found. Please check the video ID or URL and try again.")
        elif "403" in error_msg:
            raise ValueError("❌ Access denied. The video might have viewing restrictions.")
        elif "age restricted" in error_msg:
            raise ValueError("❌ Age-restricted videos are not supported")
        elif any(msg in error_msg for msg in ["proxy", "connection", "timeout"]):
            raise ValueError("❌ Connection to YouTube failed. This might be due to network restrictions. Please try again later or use a different network.")
        else:
            raise ValueError(f"❌ Could not fetch transcript: {str(e)}")
