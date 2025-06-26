import os
import logging
import faiss
from dotenv import load_dotenv
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress FAISS GPU warning
faiss.verbose = False
logging.getLogger('faiss').setLevel(logging.ERROR)

# Suppress other non-critical logs
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="NexusAI - YouTube Intelligence Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import utilities
from utils.fetch_transcript import get_transcript
from utils.embeddings import get_embeddings
from utils.groq_llm import GroqLLM
from utils.vector_store import create_vector_store, get_similar_docs

# Modern Dark Theme CSS
def setup_ui_theme():
    st.markdown("""
        <style>
            /* Modern dark theme with improved contrast */
            :root {
                --primary: #8b5cf6;
                --primary-dark: #7c3aed;
                --primary-light: #a78bfa;
                --secondary: #06b6d4;
                --accent: #f472b6;
                --dark: #0f172a;
                --darker: #020617;
                --dark-gray: #1e293b;
                --medium-gray: #334155;
                --light-gray: #e2e8f0;
                --light: #f8fafc;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --gradient: linear-gradient(135deg, var(--primary), var(--secondary));
                --glass: rgba(15, 23, 42, 0.9);
                --glass-border: rgba(255, 255, 255, 0.08);
                --glass-highlight: rgba(255, 255, 255, 0.05);
                --text-primary: #ffffff;
                --text-secondary: #e2e8f0;
                --bg-primary: #0f172a;
                --bg-secondary: #1e293b;
            }
            
            /* Base styles */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            html, body, #root, .main {
                height: 100%;
                margin: 0;
                padding: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                color: var(--text-primary);
                background-color: var(--bg-primary);
            }
            
            .main {
                background: var(--bg-primary);
                min-height: 100vh;
                padding: 1.5rem 1rem;
            }
            
            /* Streamlit overrides */
            .stApp {
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }
            
            .stTextInput>div>div>input,
            .stTextArea>div>div>textarea {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--glass-border) !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.25rem !important;
                caret-color: var(--primary) !important;  /* Cursor color */
            }
            
            /* Make sure input is visible when enabled */
            .stTextInput>div>div>input:not(:disabled),
            .stTextArea>div>div>textarea:not(:disabled) {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 1px var(--primary) !important;
            }
            
            .stTextInput>div>div>input:focus,
            .stTextArea>div>div>textarea:focus {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
            }
            
            .stButton>button {
                background: var(--gradient) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.75rem !important;
                font-weight: 600 !important;
                transition: all 0.2s ease !important;
            }
            
            .stButton>button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3) !important;
            }
            
            .stContainer {
                margin-bottom: 1.5rem !important;
            }
            
            /* Chat messages */
            .stChatMessage {
                padding: 1.25rem 1.5rem !important;
                border-radius: 12px !important;
                margin: 0.75rem 0 !important;
                max-width: 85% !important;
                backdrop-filter: blur(10px) !important;
                -webkit-backdrop-filter: blur(10px) !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                border: 1px solid var(--glass-border) !important;
                background-color: var(--bg-secondary) !important;
            }
            
            .stChatMessage:has(div[data-testid="stChatMessageUser"]) {
                margin-left: 15% !important;
                border-radius: 16px 16px 4px 16px !important;
            }
            
            .stChatMessage:has(div[data-testid="stChatMessageAssistant"]) {
                margin-right: 15% !important;
                border-radius: 16px 16px 16px 4px !important;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--darker);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--primary);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--secondary);
            }
            
            /* Chat container styles */
            .chat-container {
                background: var(--bg-secondary);
                border: 1px solid var(--glass-border);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                height: 500px;
                overflow-y: auto;
                scroll-behavior: smooth;
            }
            
            .chat-container::-webkit-scrollbar {
                width: 8px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: var(--bg-primary);
                border-radius: 4px;
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background: var(--primary);
                border-radius: 4px;
            }
        </style>
    """, unsafe_allow_html=True)

def generate_response(prompt: str, context: str, api_key: str, model_name: str, temperature: float) -> str:
    """Generate a response using Groq's API."""
    try:
        llm = GroqLLM(api_key=api_key, model_name=model_name)
        system_prompt = f"""You are a helpful AI assistant that answers questions about YouTube videos. 
        Use the following transcript context to answer the user's question. 
        Be concise and accurate in your responses.
        
        Context: {context}"""
        
        response = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=1024,
            top_p=0.9
        )
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def process_video(video_url: str, api_key: str) -> bool:
    """Process a YouTube video URL to extract transcript and create vector store."""
    if not video_url:
        st.error("❌ Please enter a YouTube URL")
        return False
        
    if not api_key:
        st.error("❌ Please enter your Groq API Key in the sidebar")
        return False
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch transcript (60% of progress)
        status_text.info("🔍 Fetching video transcript...")
        progress_bar.progress(20)
        
        try:
            st.session_state.transcript = get_transcript(video_url)
            if not st.session_state.transcript:
                raise ValueError("No transcript available for this video")
                
            progress_bar.progress(60)
            status_text.info("🧠 Creating embeddings and vector store...")
            
            # Step 2: Create embeddings and vector store (40% of progress)
            with st.spinner("Creating embeddings..."):
                embeddings = get_embeddings()
                st.session_state.vector_store = create_vector_store(
                    [st.session_state.transcript],
                    embeddings
                )
            
            progress_bar.progress(100)
            status_text.success("✅ Video processed successfully! You can now ask questions.")
            return True
            
        except ValueError as ve:
            # Specific error messages from get_transcript
            status_text.error(f"❌ {str(ve)}")
            logger.error(f"Validation error processing video: {str(ve)}")
            return False
            
        except Exception as e:
            # Log the full error for debugging
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            status_text.error(f"❌ An error occurred while processing the video. Please try again later.")
            return False
            
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Unexpected error in process_video: {str(e)}", exc_info=True)
        status_text.error("❌ An unexpected error occurred. Please try again or check the logs.")
        return False
        
    finally:
        # Ensure progress bar is complete and cleared
        progress_bar.empty()
        if 'progress_bar' in locals():
            progress_bar.empty()

def initialize_session_state():
    """Initialize session state variables."""
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None

def setup_sidebar():
    """Configure the sidebar with settings and controls."""
    with st.sidebar:
        st.markdown("<h2 style='color: var(--primary);'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        
        # API Key section
        with st.expander("🔑 API Settings", expanded=True):
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=os.getenv("GROQ_API_KEY", ""),
                help="Get your API key from https://console.groq.com/keys",
                label_visibility="collapsed"
            )
            
            if groq_api_key:
                if groq_api_key.startswith('gsk_') and len(groq_api_key) > 40:
                    st.success("✅ Valid API key format")
                else:
                    st.warning("⚠️ API key format appears invalid. It should start with 'gsk_'")
            else:
                st.warning("Please enter your Groq API key", icon="⚠️")
        
        st.markdown("---")
        
        # Model settings
        model_name = st.selectbox(
            "🤖 Choose Model",
            [
                "llama3-8b-8192",      # Llama 3 8B (recommended)
                "llama3-70b-8192",     # Llama 3 70B (larger, more capable)
                "gemma-7b-it",         # Gemma 7B (good balance)
                "mixtral-8x7b-32768"   # Legacy model
            ],
            index=0,
            help="Choose the model to use for generating responses. Recommended: llama3-8b-8192 for most use cases."
        )
        
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness (0.0 to 1.0). Lower values make responses more focused and deterministic."
        )
        
        # Info section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses Groq's API to answer questions about YouTube videos.")
        st.markdown("💡 **Tip:** For best results, use videos with English captions.")
    
    return groq_api_key, model_name, temperature

def extract_video_id(video_url: str) -> str:
    """Extract video ID from YouTube URL."""
    if 'youtube.com/watch?v=' in video_url:
        return video_url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in video_url:
        return video_url.split('youtu.be/')[-1].split('?')[0]
    return video_url  # Assume it's just the ID

def display_video_preview(video_url: str):
    """Display video preview based on the URL."""
    if not video_url:
        return None
        
    try:
        video_id = extract_video_id(video_url)
        
        # Create two columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video player
            st.markdown(f"""
                <div style='background: var(--bg-secondary);
                            border: 1px solid var(--glass-border);
                            border-radius: 12px;
                            padding: 1rem;
                            margin-bottom: 1.5rem;'>
                    <div style='position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 8px;'>
                        <iframe 
                            style='position: absolute; top: 0; left: 0; width: 100%; height: 100%;'
                            src="https://www.youtube.com/embed/{video_id}?modestbranding=1&rel=0" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen>
                        </iframe>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Video info placeholder
            st.markdown(f"""
                <div style='background: var(--bg-secondary);
                            border: 1px solid var(--glass-border);
                            border-radius: 12px;
                            padding: 1.5rem;
                            height: 100%;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;'>
                    <h3 style='color: var(--primary); margin-top: 0;'>Video Information</h3>
                    <p style='color: var(--text-secondary); margin: 0.5rem 0;'>
                        <i class='fas fa-link' style='margin-right: 8px; color: var(--primary);'></i>
                        <span>Video ID: {video_id}</span>
                    </p>
                    <div style='margin-top: 1rem;'>
                        <button onclick='copyVideoUrl()' style='background: var(--primary); color: white; border: none; border-radius: 6px; padding: 0.5rem 1rem; cursor: pointer; font-size: 0.9rem;'>
                            <i class='fas fa-copy' style='margin-right: 6px;'></i> Copy Video URL
                        </button>
                    </div>
                </div>
                <script>
                    function copyVideoUrl() {{
                        navigator.clipboard.writeText(window.location.href.split('?')[0]);
                        alert('Video URL copied to clipboard!');
                    }}
                </script>
            """, unsafe_allow_html=True)
        
        return video_id
        
    except Exception as e:
        st.error(f"Could not load video preview: {str(e)}")
        return None

def display_chat_interface(api_key: str, model_name: str, temperature: float):
    """Display the chat interface and handle user input."""
    # Chat container with minimal styling
    st.markdown("""
        <div style='margin: 1rem 0; padding: 1rem; border-radius: 0.5rem;'>
    """, unsafe_allow_html=True)
    
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], 
                           avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close chat container
    
    # Chat input section
    st.markdown("<div style='margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # Check if video is processed
    is_video_processed = 'video_id' in st.session_state and st.session_state.video_id
    
    # Create a form for the chat input
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([5, 1])
        with cols[0]:
            user_input = st.text_input(
                "Chat input",
                label_visibility="collapsed",
                placeholder="Ask me anything about this video..." if is_video_processed 
                           else "Enter a YouTube URL and click 'Process Video' to start chatting...",
                disabled=not is_video_processed,
                key="chat_input"
            )
        with cols[1]:
            submit_button = st.form_submit_button(
                "➜",
                use_container_width=True,
                disabled=not is_video_processed
            )
    
    # Process the form submission
    if submit_button and user_input.strip() and is_video_processed:
        # Check if this is a duplicate of the last message
        if st.session_state.messages and st.session_state.messages[-1]["content"] == user_input.strip():
            st.warning("Please enter a new message or wait for the current response.")
            return
            
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        
        # Process the message if we have a vector store
        if 'vector_store' in st.session_state and st.session_state.vector_store:
            with st.spinner("💭 Thinking..."):
                try:
                    # Get relevant context from the transcript
                    docs = get_similar_docs(st.session_state.vector_store, user_input, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Generate response using Groq
                    response = generate_response(
                        prompt=user_input,
                        context=context,
                        api_key=api_key,
                        model_name=model_name,
                        temperature=temperature
                    )
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
            
    # Auto-scroll to bottom of chat
    st.markdown("""
        <script>
            // Auto-scroll to bottom of chat
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
    """, unsafe_allow_html=True)

def display_help_section():
    """Display the help section in an expander."""
    with st.expander("ℹ️ How to use NexusAI", expanded=False):
        st.markdown("""
        ### 🚀 Getting Started
        1. Enter a YouTube URL or video ID above
        2. Start chatting with the video content
        3. Ask questions, get summaries, or analyze the content
        
        ### 💡 Pro Tips
        • Ask specific questions for better answers
        • Use "summarize this video" for a quick overview
        • Try different AI models for varied responses
        • Adjust temperature for more focused or creative answers
        
        ### 🔍 Examples
        • "What are the main points of this video?"
        • "Explain the key concepts in simple terms"
        • "Generate a bullet-point summary"
        • "What questions should I ask about this content?"
        """)

def main():
    """Main function to run the Streamlit app."""
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    initialize_session_state()
    
    # Setup UI theme
    setup_ui_theme()
    
    # Setup sidebar and get settings
    groq_api_key, model_name, temperature = setup_sidebar()
    
    # Main header with app name and description
    st.markdown("""
        <div style='margin-bottom: 2rem;'>
            <h1 style='color: var(--primary); margin-bottom: 0.5rem;'>NexusAI</h1>
            <p style='color: var(--text-secondary); font-size: 1.1rem; margin-top: 0;'>
                Intelligent YouTube Video Analysis & Chat
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Video URL input
    video_url = st.text_input(
        "Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a YouTube URL or just the video ID",
        label_visibility="collapsed",
        key="video_url_input"
    )
    
    # Process button
    process_btn = st.button("Process Video", use_container_width=True, key="process_btn")
    
    # Process video if button is clicked
    if process_btn and video_url:
        if process_video(video_url, groq_api_key):
            st.session_state.video_id = extract_video_id(video_url)
            if 'messages' not in st.session_state or not st.session_state.messages:
                st.session_state.messages = [{"role": "assistant", "content": "I've processed the video. What would you like to know?"}]
    
    # Display video preview if URL is provided
    if video_url:
        video_id = display_video_preview(video_url)
        if video_id and 'video_id' not in st.session_state:
            st.session_state.video_id = video_id
    
    # Display chat interface
    display_chat_interface(groq_api_key, model_name, temperature)
    
    # Display help section
    display_help_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: var(--text-secondary); font-size: 0.9rem; margin-top: 2rem;'>
            <p>NexusAI - YouTube Intelligence Assistant</p>
            <p>Powered by Groq & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()