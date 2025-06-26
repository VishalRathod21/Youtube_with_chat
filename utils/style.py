import streamlit as st

def colored_header(label, description=None, color_name="blue-70"):
    """
    Display a colored header with optional description.
    
    Args:
        label: The main header text
        description: Optional description text
        color_name: Color name from the color map (e.g., "blue-70")
    """
    color_map = {
        'blue-70': '#1E88E5',
        'green-70': '#43A047',
        'red-70': '#E53935',
        'orange-70': '#FB8C00',
        'purple-70': '#8E24AA',
        'teal-70': '#00897B',
    }
    
    color = color_map.get(color_name, color_map['blue-70'])
    
    st.markdown(f"""
    <div style='padding: 10px 0 10px 0; border-bottom: 2px solid {color}; margin-bottom: 15px;'>
        <h2 style='color: {color}; margin: 0;'>{label}</h2>
        {f"<p style='color: #666; margin: 5px 0 0 0;'>{description}</p>" if description else ""}
    </div>
    """, unsafe_allow_html=True)

def apply_style():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #1E88E5;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #1565C0;
            color: white;
        }
        
        /* Text input */
        .stTextInput>div>div>input {
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            padding: 0.5rem;
        }
        
        /* Sidebar */
        .css-1d391kg {
            padding: 1.5rem;
            background-color: #f8f9fa;
        }
        
        /* Chat messages */
        .stChatMessage {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .stChatMessage.user {
            background-color: #E3F2FD;
        }
        
        .stChatMessage.assistant {
            background-color: #F5F5F5;
        }
        
        /* Spinner */
        .stSpinner>div {
            border-color: #1E88E5 transparent transparent transparent;
        }
    </style>
    """, unsafe_allow_html=True)

def get_prompt_template():
    """
    Get the default prompt template for the chat.
    
    Returns:
        str: The prompt template
    """
    return """You are a helpful AI assistant that answers questions about YouTube videos. 
    Use the following context from the video transcript to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    Answer:"""
