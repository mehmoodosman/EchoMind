from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from gtts import gTTS
import google.generativeai as genai

# --- App bootstrap --------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="EchoMind – Assistive Communication",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Force light theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f9fbff 0%, #f4f7fb 100%);
    }
    </style>
""", unsafe_allow_html=True)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Set it in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Initialize model lazily with error handling
@st.cache_resource
def get_gemini_model():
    try:
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to initialize Gemini model '{MODEL_NAME}': {e}")
        st.info("Try setting GEMINI_MODEL in .env to 'gemini-pro' or 'gemini-1.5-pro'")
        st.stop()
        return None

CHILD_ID = "demo_child"
CATEGORY_CONFIG: Dict[str, str] = {
    "Body & Needs": "🍎",
    "Feelings & Sensory": "💛",
    "Activities & People": "�",
    "Help & Safety": "🆘",
}


# --- Helper functions ------------------------------------------------------ #


def init_session_state() -> None:
    defaults = {
        "stage": "intro",
        "selected_category": None,
        "latitude": None,
        "longitude": None,
        "location_name": None,
        "gps_requested": False,
        "options": [],
        "last_phrase": None,
        "audio_file": None,
        "recent_phrases": [],  # Store recent phrases for quick access
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_to_recent_phrases(phrase: str, emoji: str) -> None:
    """Add a phrase to recent history (max 5, no duplicates)"""
    recent = st.session_state.recent_phrases
    new_entry = {"text": phrase, "emoji": emoji}
    # Remove if already exists
    recent = [p for p in recent if p["text"] != phrase]
    # Add to front
    recent.insert(0, new_entry)
    # Keep only last 5
    st.session_state.recent_phrases = recent[:5]


def get_current_datetime() -> Dict[str, str]:
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "time_of_day": "morning" if now.hour < 12 else "afternoon" if now.hour < 17 else "evening",
    }


def render_gps_location() -> None:
    """Render GPS location component to request location from browser"""
    if not st.session_state.gps_requested:
        return
    
    # JavaScript to get GPS and store in sessionStorage
    # Since iframe is sandboxed, we rely on sessionStorage and parent window checking
    js_code = """
    <script>
    (function() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    
                    // Store in sessionStorage as strings
                    sessionStorage.setItem('gps_lat', String(lat));
                    sessionStorage.setItem('gps_lng', String(lng));
                    sessionStorage.setItem('gps_timestamp', String(Date.now()));
                    
                    console.log('GPS coordinates stored in sessionStorage:', lat, lng);
                    
                    // Try to send message to parent window
                    try {
                        window.parent.postMessage({
                            type: 'gps_coordinates',
                            lat: lat,
                            lng: lng
                        }, '*');
                    } catch (e) {
                        console.log('postMessage failed, coordinates in sessionStorage');
                    }
                },
                function(error) {
                    console.error('Geolocation error:', error);
                    alert('Unable to get location: ' + error.message + '. Please check your browser permissions.');
                },
                {
                    enableHighAccuracy: true,
                    timeout: 15000,
                    maximumAge: 0
                }
            );
        } else {
            alert('Geolocation is not supported by your browser.');
        }
    })();
    </script>
    """
    components.html(js_code, height=0)


def build_context(category: str) -> Dict[str, str]:
    datetime_info = get_current_datetime()
    location_str = ""
    if st.session_state.latitude and st.session_state.longitude:
        location_str = f"GPS coordinates: {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f}"
        if st.session_state.location_name:
            location_str += f" ({st.session_state.location_name})"
    
    return {
        "child_id": CHILD_ID,
        "category": category,
        "date": datetime_info["date"],
        "time": datetime_info["time"],
        "day_of_week": datetime_info["day_of_week"],
        "time_of_day": datetime_info["time_of_day"],
        "location": location_str if location_str else "Location not available",
        "latitude": str(st.session_state.latitude) if st.session_state.latitude else None,
        "longitude": str(st.session_state.longitude) if st.session_state.longitude else None,
        "last_phrase": st.session_state.get("last_phrase"),
    }


def load_prompt_template() -> str:
    return (
        "You help a non-verbal autistic child communicate using very short first-person "
        "phrases.\n"
        "Context:\n{context}\n\n"
        "Return a JSON object with a key `phrases` that maps to an array of exactly three objects. "
        "Each object must have two keys: `text` (a short literal phrase suitable for text-to-speech) "
        "and `emoji` (a single relevant emoji). The phrases must be literal, concrete, and avoid "
        "question marks, metaphors, or figurative language. Choose emojis that clearly represent "
        "the meaning of each phrase."
    )


def parse_model_output(raw_text: str) -> List[Dict[str, str]]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1]
    
    data = json.loads(cleaned)
    phrases = data.get("phrases", [])
    if not isinstance(phrases, list):
        raise ValueError("Expected 'phrases' to be a list")
    
    if len(phrases) != 3:
        raise ValueError(f"Expected exactly 3 phrases, got {len(phrases)}")
    
    result = []
    for item in phrases:
        if not isinstance(item, dict):
            raise ValueError(f"Expected phrase to be a dict, got {type(item).__name__}")
        
        text = item.get("text", "").strip()
        emoji = item.get("emoji", "").strip()
        
        if not text:
            raise ValueError("Phrase 'text' field is required and cannot be empty")
        if not emoji:
            raise ValueError("Phrase 'emoji' field is required and cannot be empty")
        
        result.append({"text": text, "emoji": emoji})
    
    return result


def generate_ai_options(category: str, context: Dict[str, str]) -> List[Dict[str, str]]:
    model = get_gemini_model()
    if not model:
        st.error("Gemini model is not available.")
        st.stop()
        return []
    
    prompt_template = load_prompt_template()
    context_lines = [
        f"Child ID: {context['child_id']}",
        f"Category: {context['category']}",
        f"Date: {context['date']}",
        f"Time: {context['time']}",
        f"Day of week: {context['day_of_week']}",
        f"Time of day: {context['time_of_day']}",
        f"Location: {context['location']}",
    ]
    if context.get("latitude") and context.get("longitude"):
        context_lines.append(f"GPS coordinates: {context['latitude']}, {context['longitude']}")
    if context.get("last_phrase"):
        context_lines.append(f"Last phrase spoken: {context['last_phrase']}")

    prompt = prompt_template.format(context="\n".join(context_lines))

    try:
        response = model.generate_content(prompt)
        if not getattr(response, "text", "").strip():
            st.error("Gemini returned an empty response. Please try again.")
            st.stop()
            return []
        phrases = parse_model_output(response.text)
        return phrases
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse Gemini response as JSON: {e}")
        st.stop()
        return []
    except ValueError as e:
        st.error(f"Invalid response format from Gemini: {e}")
        st.stop()
        return []
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        st.stop()
        return []


def build_option_payload(category: str, phrases: List[Dict[str, str]]) -> List[Dict[str, str]]:
    options = []
    for idx, phrase_data in enumerate(phrases):
        options.append(
            {
                "id": idx,
                "text": phrase_data["text"],
                "emoji": phrase_data["emoji"],
            }
        )
    return options


def fetch_options(category: str) -> None:
    context = build_context(category)
    phrases = generate_ai_options(category, context)
    if not phrases:
        return  # Error already displayed by generate_ai_options
    st.session_state.options = build_option_payload(category, phrases)
    st.session_state.stage = "phrases"


def synthesize_audio(text: str) -> Optional[str]:
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        gTTS(text=text, lang="en").write_to_fp(tmp)
        tmp.close()
        return tmp.name
    except Exception as exc:  # pragma: no cover - Streamlit surface
        st.warning(f"Unable to generate audio: {exc}")
        return None


def reset_flow() -> None:
    st.session_state.stage = "intro"
    st.session_state.selected_category = None
    st.session_state.options = []
    st.session_state.last_phrase = None
    st.session_state.audio_file = None


def render_loading_animation() -> None:
    """Render a calming loading animation while AI generates phrases"""
    loading_html = """
    <div class="loading-container">
        <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
        <div class="loading-text">Finding words for you...</div>
    </div>
    """
    st.markdown(loading_html, unsafe_allow_html=True)


def render_emergency_button() -> None:
    """Render a floating emergency help button on all screens"""
    if st.session_state.stage == "voice":
        return  # Don't show on voice output screen
    
    # Emergency button positioned absolutely
    if st.button("🆘", key="emergency_help", help="I need help now!"):
        st.session_state.last_phrase = "I need help"
        st.session_state.audio_file = synthesize_audio("I need help")
        add_to_recent_phrases("I need help", "🆘")
        st.session_state.stage = "voice"
        st.rerun()
    
    # Style emergency button
    st.markdown("""
    <style>
    button[data-testid*="emergency_help"] {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        width: 80px !important;
        height: 80px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        color: white !important;
        font-size: 2.5rem !important;
        border: 4px solid rgba(255,255,255,0.3) !important;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4) !important;
        z-index: 1000 !important;
        animation: emergency-pulse 3s infinite !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    @keyframes emergency-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    button[data-testid*="emergency_help"]:hover {
        transform: scale(1.15) !important;
        animation: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- UI Sections ----------------------------------------------------------- #


def inject_custom_css() -> None:
    """Inject custom CSS to match the prototype design - optimized for autistic users"""
    css = """
    <style>
    :root {
        --bg: #f9fbff;
        --card: #ffffff;
        --accent: #60be9b;
        --accent-soft: #d9f5e9;
        --text: #1a202c;
        --muted: #718096;
        --danger: #f56565;
        --radius-lg: 24px;
        --radius-md: 16px;
        --shadow-sm: 0 3px 14px rgba(15, 23, 42, 0.05);
        --shadow-md: 0 4px 18px rgba(15, 23, 42, 0.06);
        --shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    
    /* Reset and base styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        min-height: 100vh !important;
    }
    
    /* Remove Streamlit default styling */
    .main .block-container {
        max-width: 100vw !important;
        padding: 0 !important;
        background: transparent !important;
    }
    
    /* Hide Streamlit UI elements */
    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Main app container */
    .autism-app {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    /* Floating background elements */
    .autism-app::before {
        content: '';
        position: absolute;
        top: -50px;
        left: -50px;
        width: 200px;
        height: 100px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 50px;
        animation: float 25s infinite linear;
        z-index: 1;
    }
    
    .autism-app::after {
        content: '';
        position: absolute;
        top: 30%;
        right: -100px;
        width: 300px;
        height: 150px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 75px;
        animation: float 30s infinite linear reverse;
        z-index: 1;
    }
    
    @keyframes float {
        0% { transform: translateX(-100px); }
        100% { transform: translateX(calc(100vw + 100px)); }
    }
    
    /* Content wrapper */
    .autism-content {
        position: relative;
        z-index: 10;
        width: 100%;
        max-width: 600px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2rem;
    }
    
    /* Header section */
    .autism-header {
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    
    .autism-title {
        font-size: clamp(2rem, 8vw, 3.5rem);
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        letter-spacing: -0.02em;
    }
    
    .autism-subtitle {
        font-size: clamp(1rem, 4vw, 1.3rem);
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Main action button */
    .main-speak-btn {
        width: clamp(200px, 40vw, 280px);
        height: clamp(200px, 40vw, 280px);
        border-radius: 50%;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: 6px solid rgba(255,255,255,0.3);
        box-shadow: 0 20px 60px rgba(79, 172, 254, 0.4);
        font-size: clamp(3rem, 8vw, 5rem);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .main-speak-btn::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 4s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Category grid */
    .category-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        width: 100%;
        max-width: 500px;
        margin: 0 auto;
    }
    
    .category-btn {
        height: clamp(140px, 25vw, 180px);
        border-radius: 20px;
        border: none;
        font-size: clamp(0.9rem, 3vw, 1.2rem);
        font-weight: 700;
        color: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.8rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .category-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .category-btn-emoji {
        font-size: clamp(2.5rem, 6vw, 3.5rem);
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    }
    
    /* Phrase buttons */
    .phrase-list {
        display: flex;
        flex-direction: column;
        gap: 1.2rem;
        width: 100%;
        max-width: 500px;
        margin: 0 auto;
    }
    
    .phrase-btn {
        min-height: 80px;
        border-radius: 16px;
        background: white;
        border: 3px solid rgba(255,255,255,0.8);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        font-size: clamp(1rem, 4vw, 1.3rem);
        font-weight: 600;
        color: #2d3748;
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .phrase-btn-emoji {
        font-size: clamp(2rem, 5vw, 2.5rem);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    /* Voice output celebration */
    .voice-celebration {
        text-align: center;
        color: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2rem;
    }
    
    .voice-icon {
        width: clamp(150px, 30vw, 200px);
        height: clamp(150px, 30vw, 200px);
        border-radius: 50%;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: clamp(3rem, 8vw, 5rem);
        animation: pulse-glow 2s infinite;
        box-shadow: 0 0 60px rgba(79, 172, 254, 0.6);
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            transform: scale(1); 
            box-shadow: 0 0 60px rgba(79, 172, 254, 0.6); 
        }
        50% { 
            transform: scale(1.05); 
            box-shadow: 0 0 80px rgba(79, 172, 254, 0.8); 
        }
    }
    
    .voice-text {
        font-size: clamp(1.5rem, 5vw, 2.5rem);
        font-weight: 800;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        line-height: 1.2;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    /* Navigation buttons */
    .nav-btn {
        background: rgba(255,255,255,0.2);
        color: white;
        border: 2px solid rgba(255,255,255,0.3);
        border-radius: 50px;
        font-size: clamp(0.9rem, 3vw, 1.1rem);
        font-weight: 600;
        padding: 1rem 2rem;
        min-height: 50px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        margin-top: 1rem;
        width: fit-content;
        min-width: 150px;
    }
    
    /* Recent phrases */
    .recent-section {
        margin-top: 2rem;
        text-align: center;
        width: 100%;
    }
    
    .recent-title {
        color: rgba(255,255,255,0.8);
        font-size: clamp(0.9rem, 3vw, 1.1rem);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .recent-grid {
        display: flex;
        gap: 0.8rem;
        justify-content: center;
        flex-wrap: wrap;
        max-width: 100%;
    }
    
    .recent-btn {
        background: rgba(255,255,255,0.15);
        color: white;
        border: 2px solid rgba(255,255,255,0.2);
        border-radius: 25px;
        padding: 0.6rem 1.2rem;
        font-size: clamp(0.8rem, 2.5vw, 1rem);
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    
    /* Accessibility improvements */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    button {
        min-height: 60px !important;
        min-width: 60px !important;
        font-family: inherit !important;
    }
    
    /* Focus indicators for keyboard navigation */
    *:focus-visible {
        outline: 4px solid rgba(255, 255, 255, 0.8) !important;
        outline-offset: 4px !important;
    }
    
    /* Reduce motion for sensitive users */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* Remove default button styling */
    button {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .echomind-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }
    
    .echomind-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .bubble-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--accent-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    
    .echomind-title {
        font-weight: 700;
        font-size: 18px;
        margin: 0;
        color: var(--text);
    }
    
    .echomind-subtitle {
        font-size: 11px;
        color: var(--muted);
        margin: 0;
    }
    
    .pill {
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.8);
        font-size: 10px;
        color: var(--muted);
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    .pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #48bb78;
    }
    
    .badge {
        padding: 2px 6px;
        border-radius: 999px;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        background: var(--accent-soft);
        color: var(--muted);
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .primary-btn-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0;
    }
    
    .primary-btn {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        font-size: 18px;
        font-weight: 700;
        background: var(--accent);
        color: white;
        border: none;
        box-shadow: 0 10px 30px rgba(96, 190, 155, 0.55);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        touch-action: manipulation;
        -webkit-tap-highlight-color: transparent;
    }
    
    .primary-btn:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(96, 190, 155, 0.65);
    }
    
    .primary-btn:active {
        transform: translateY(0px) scale(0.98);
        box-shadow: 0 6px 20px rgba(96, 190, 155, 0.5);
    }
    
    .primary-btn:focus-visible {
        outline: 3px solid rgba(96, 190, 155, 0.5);
        outline-offset: 4px;
    }
    
    .primary-btn-icon {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: rgba(255,255,255,0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
    }
    
    .hint {
        font-size: 12px;
        color: var(--muted);
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .card {
        background: var(--card);
        border-radius: var(--radius-lg);
        padding: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    
    .tile-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 1rem 0;
    }
    
    .tile-btn {
        padding: 18px 14px;
        border-radius: var(--radius-md);
        background: var(--card);
        box-shadow: var(--shadow-md);
        border: 2px solid transparent;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 140px;
        -webkit-tap-highlight-color: transparent;
    }
    
    .tile-btn:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        border-color: var(--accent-soft);
    }
    
    .tile-btn:active {
        transform: translateY(0px) scale(0.98);
        box-shadow: var(--shadow-sm);
    }
    
    .tile-btn:focus-visible {
        outline: 3px solid var(--accent-soft);
        outline-offset: 2px;
    }
    
    .tile-icon-wrap {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 4px;
        text-align: center;
        padding: 4px;
        flex-shrink: 0;
    }
    
    .tile-body .tile-icon-wrap {
        background: #ffeec2;
    }
    
    .tile-feelings .tile-icon-wrap {
        background: #e1e4ff;
    }
    
    .tile-activities .tile-icon-wrap {
        background: #ffd6e8;
    }
    
    .tile-safety .tile-icon-wrap {
        background: #ffe1e1;
    }
    
    .tile-label {
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        color: var(--text);
    }
    
    .tile-sub {
        font-size: 10px;
        color: var(--muted);
        text-align: center;
    }
    
    .suggestion-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .suggestion-btn {
        width: 100%;
        text-align: left;
        padding: 16px 18px;
        border-radius: var(--radius-md);
        background: var(--card);
        border: 2px solid transparent;
        box-shadow: var(--shadow-sm);
        display: flex;
        align-items: center;
        gap: 12px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 60px;
        -webkit-tap-highlight-color: transparent;
    }
    
    .suggestion-btn:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.1);
        border-color: var(--accent-soft);
    }
    
    .suggestion-btn:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: var(--shadow-sm);
    }
    
    .suggestion-btn:focus-visible {
        outline: 3px solid var(--accent-soft);
        outline-offset: 2px;
    }
    
    .suggestion-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--accent-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        flex-shrink: 0;
    }
    
    .suggestion-text-main {
        font-size: 14px;
        font-weight: 600;
        color: var(--text);
    }
    
    .none-btn {
        width: 100%;
        padding: 14px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        color: var(--danger);
        background: rgba(254, 226, 226, 0.7);
        border: 2px solid transparent;
        cursor: pointer;
        margin-top: 0.5rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 48px;
    }
    
    .none-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(245, 101, 101, 0.2);
        border-color: rgba(245, 101, 101, 0.3);
    }
    
    .none-btn:active {
        transform: translateY(0px);
    }
    
    .back-btn {
        width: 100%;
        padding: 12px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.9);
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
        border: 2px solid transparent;
        box-shadow: 0 3px 12px rgba(15,23,42,0.06);
        cursor: pointer;
        margin-top: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 48px;
    }
    
    .back-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 16px rgba(15,23,42,0.1);
        border-color: var(--accent-soft);
    }
    
    .back-btn:active {
        transform: translateY(0px);
    }
    
    .play-card {
        text-align: center;
        margin: 2rem 0;
    }
    
    .play-icon {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: var(--accent-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px;
        font-size: 32px;
    }
    
    .play-phrase {
        font-size: 18px;
        font-weight: 700;
        margin: 1rem 0;
        color: var(--text);
    }
    
    .play-btn {
        margin-top: 10px;
        padding: 12px 20px;
        border-radius: 999px;
        background: var(--accent);
        color: white;
        font-size: 14px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        touch-action: manipulation;
        min-height: 48px;
        box-shadow: 0 4px 12px rgba(96, 190, 155, 0.3);
    }
    
    .play-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(96, 190, 155, 0.4);
    }
    
    .play-btn:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(96, 190, 155, 0.3);
    }
    
    .play-btn:focus-visible {
        outline: 3px solid rgba(96, 190, 155, 0.5);
        outline-offset: 2px;
    }
    
    .section-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text);
    }
    
    /* Accessibility improvements for autistic users */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Improve text readability */
    body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        line-height: 1.6 !important;
    }
    
    /* Better contrast for text */
    .card p, .card h2 {
        color: var(--text) !important;
    }
    
    /* Larger touch targets - minimum 44x44px for accessibility */
    button {
        min-height: 44px !important;
        min-width: 44px !important;
    }
    
    /* Smooth animations - not jarring */
    * {
        transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Remove distracting elements */
    .stApp > div:first-child {
        padding-top: 0 !important;
    }
    
    /* Better spacing for clarity */
    .card {
        margin-bottom: 1.5rem;
    }
    
    /* Ensure emojis are properly sized */
    .tile-icon-wrap, .suggestion-icon, .bubble-icon, .play-icon {
        font-size: inherit !important;
        line-height: 1 !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* Remove default Streamlit styling and ensure border radius */
    .stButton > button {
        width: 100%;
        border-radius: 16px !important;
    }
    
    /* Default button styling - light colors with border radius */
    button:not([kind="primary"]):not([data-testid*="cat-"]):not([data-testid*="phrase-"]):not([data-testid*="back_"]):not([data-testid*="none_"]):not([data-testid*="play_"]):not([data-testid*="back_home"]) {
        color: #1a202c !important;
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 16px !important;
    }
    
    /* Primary buttons keep their accent color */
    button[kind="primary"] {
        color: white !important;
        background: var(--accent) !important;
        border: none !important;
        border-radius: 16px !important;
    }
    
    /* Ensure all buttons have proper border radius as fallback */
    button {
        border-radius: 16px !important;
        user-select: none;
        -webkit-user-select: none;
    }
    
    /* Better focus indicators for keyboard navigation */
    *:focus-visible {
        outline-width: 3px !important;
        outline-style: solid !important;
        outline-offset: 2px !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header() -> None:
    """Render the autism-friendly header"""
    inject_custom_css()
    
    st.markdown("""
    <div class="autism-app">
        <div class="autism-content">
            <div class="autism-header">
                <div class="autism-title">EchoMind</div>
                <div class="autism-subtitle">Tap to speak your thoughts</div>
            </div>
    """, unsafe_allow_html=True)
    
    # Add GPS listener script once per page load
    if "gps_listener_added" not in st.session_state:
        listener_js = """
        <script>
        (function() {
            // Listen for postMessage from iframe
            window.addEventListener('message', function(event) {
                if (event.data && event.data.type === 'gps_coordinates') {
                    const lat = event.data.lat;
                    const lng = event.data.lng;
                    const currentUrl = window.location.href.split('?')[0];
                    const newUrl = currentUrl + '?lat=' + encodeURIComponent(lat) + '&lng=' + encodeURIComponent(lng) + '&gps_updated=true';
                    console.log('Updating URL from postMessage:', newUrl);
                    window.location.href = newUrl;
                }
            });
            
            // Check sessionStorage immediately and periodically as fallback
            function checkSessionStorage() {
                const lat = sessionStorage.getItem('gps_lat');
                const lng = sessionStorage.getItem('gps_lng');
                if (lat && lng && !window.location.search.includes('lat=')) {
                    console.log('Found GPS in sessionStorage, updating URL:', lat, lng);
                    const currentUrl = window.location.href.split('?')[0];
                    const newUrl = currentUrl + '?lat=' + encodeURIComponent(lat) + '&lng=' + encodeURIComponent(lng) + '&gps_updated=true';
                    window.location.href = newUrl;
                    return true;
                }
                return false;
            }
            
            // Check immediately
            checkSessionStorage();
            
            // Check more frequently when GPS might be requested
            setInterval(checkSessionStorage, 300);
        })();
        </script>
        """
        components.html(listener_js, height=0)
        st.session_state.gps_listener_added = True


def render_context_log() -> None:
    """Display GPS and time context information"""
    with st.expander("📊 Context Log (GPS & Time)", expanded=False):
        datetime_info = get_current_datetime()
        
        st.markdown("### Time Context")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Date:** {datetime_info['date']}")
            st.write(f"**Time:** {datetime_info['time']}")
        with col2:
            st.write(f"**Day of Week:** {datetime_info['day_of_week']}")
            st.write(f"**Time of Day:** {datetime_info['time_of_day']}")
        
        st.markdown("### GPS Context")
        if st.session_state.latitude and st.session_state.longitude:
            st.success(f"**Coordinates:** {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f}")
            if st.session_state.location_name:
                st.write(f"**Location Name:** {st.session_state.location_name}")
        else:
            st.warning("**GPS:** Not available")
            if st.session_state.gps_requested:
                st.info("⏳ Waiting for browser location permission...")
        
        # Debug: Show query params if present
        query_params = st.query_params
        if query_params:
            st.markdown("### Debug: Query Parameters")
            st.json(dict(query_params))
        
        # Show what will be sent to Gemini
        st.markdown("### Context Sent to Gemini")
        context = build_context(st.session_state.selected_category or "N/A")
        context_lines = []
        for key, value in context.items():
            if value:
                if key == "last_phrase":
                    context_lines.append(f"- Last phrase spoken: {value}")
                else:
                    context_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        st.code("\n".join(context_lines), language="text")


def render_location_status() -> None:
    """Display GPS location status and button to request location"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.gps_requested:
            st.warning("🔄 Requesting location... Please allow location access in your browser.")
        elif st.session_state.latitude and st.session_state.longitude:
            st.success(f"📍 Location: {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f}")
        else:
            st.info("📍 Location: Not available")
    
    with col2:
        if st.button("📍 Get Location", help="Request GPS location from your device", disabled=st.session_state.gps_requested):
            st.session_state.gps_requested = True
            st.rerun()
    
    # Render GPS component if requested
    render_gps_location()


def render_stage_intro() -> None:
    """Render the intro screen with huge, friendly button"""
    # Main speak button
    if st.button("🎙️", key="speak_main", help="Tap to start speaking"):
        st.session_state.stage = "categories"
        st.rerun()
    
    # Style the main button
    st.markdown("""
    <style>
    button[data-testid*="speak_main"] {
        width: clamp(200px, 40vw, 280px) !important;
        height: clamp(200px, 40vw, 280px) !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        border: 6px solid rgba(255,255,255,0.3) !important;
        box-shadow: 0 20px 60px rgba(79, 172, 254, 0.4) !important;
        font-size: clamp(3rem, 8vw, 5rem) !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 auto !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    button[data-testid*="speak_main"]:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 25px 80px rgba(79, 172, 254, 0.6) !important;
    }
    button[data-testid*="speak_main"]:active {
        transform: scale(0.95) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Recent phrases section
    if st.session_state.recent_phrases:
        st.markdown("""
        <div class="recent-section">
            <div class="recent-title">Recently used</div>
            <div class="recent-grid">
        """, unsafe_allow_html=True)
        
        # Create buttons for recent phrases
        cols = st.columns(min(len(st.session_state.recent_phrases), 3))
        for idx, phrase in enumerate(st.session_state.recent_phrases[:3]):
            with cols[idx % 3]:
                if st.button(f"{phrase['emoji']} {phrase['text']}", key=f"recent-{idx}"):
                    st.session_state.last_phrase = phrase["text"]
                    st.session_state.audio_file = synthesize_audio(phrase["text"])
                    add_to_recent_phrases(phrase["text"], phrase["emoji"])
                    st.session_state.stage = "voice"
                    st.rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Style recent buttons
        st.markdown("""
        <style>
        button[data-testid*="recent-"] {
            background: rgba(255,255,255,0.15) !important;
            color: white !important;
            border: 2px solid rgba(255,255,255,0.2) !important;
            border-radius: 25px !important;
            padding: 0.6rem 1.2rem !important;
            font-size: clamp(0.8rem, 2.5vw, 1rem) !important;
            font-weight: 600 !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.3s ease !important;
            white-space: nowrap !important;
            width: 100% !important;
        }
        button[data-testid*="recent-"]:hover {
            background: rgba(255,255,255,0.25) !important;
            transform: translateY(-2px) !important;
        }
        </style>
        """, unsafe_allow_html=True)


def render_categories() -> None:
    """Render category selection with large, colorful tiles"""
    st.markdown('<div class="category-grid">', unsafe_allow_html=True)
    
    categories = [
        ("Body & Needs", "🍎"),
        ("Feelings & Sensory", "💛"),
        ("Activities & People", "🎨"),
        ("Help & Safety", "🆘")
    ]
    
    col1, col2 = st.columns(2)
    
    for idx, (label, emoji) in enumerate(categories):
        col = col1 if idx % 2 == 0 else col2
        with col:
            if st.button(f"{emoji}\n{label}", key=f"cat-{idx}"):
                st.session_state.selected_category = label
                fetch_options(label)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Style category buttons
    st.markdown("""
    <style>
    button[data-testid*="cat-"] {
        width: 100% !important;
        height: clamp(140px, 25vw, 180px) !important;
        border-radius: 20px !important;
        border: none !important;
        font-size: clamp(0.9rem, 3vw, 1.2rem) !important;
        font-weight: 700 !important;
        color: white !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.8rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        white-space: pre-line !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    button[data-testid*="cat-"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    button[data-testid*="cat-0"] { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%) !important; }
    button[data-testid*="cat-1"] { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important; }
    button[data-testid*="cat-2"] { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%) !important; }
    button[data-testid*="cat-3"] { background: linear-gradient(135deg, #ff8a80 0%, #ff5722 100%) !important; }
    button[data-testid*="cat-"]:hover {
        transform: translateY(-6px) scale(1.02) !important;
    }
    button[data-testid*="cat-"]:active {
        transform: translateY(-2px) scale(0.98) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back", key="back_intro"):
        reset_flow()
    
    st.markdown("""
    <style>
    button[data-testid*="back_intro"] {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 50px !important;
        font-size: clamp(0.9rem, 3vw, 1.1rem) !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
        min-height: 50px !important;
        backdrop-filter: blur(10px) !important;
        margin-top: 1rem !important;
        width: fit-content !important;
        min-width: 150px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
    }
    button[data-testid*="back_intro"]:hover {
        background: rgba(255,255,255,0.3) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_phrase_options() -> None:
    """Render phrase suggestions with large, easy-to-tap buttons"""
    st.markdown('<div class="phrase-list">', unsafe_allow_html=True)
    
    for idx, option in enumerate(st.session_state.options):
        if st.button(f"{option['emoji']}  {option['text']}", key=f"phrase-{idx}"):
            st.session_state.last_phrase = option["text"]
            st.session_state.audio_file = synthesize_audio(option["text"])
            add_to_recent_phrases(option["text"], option["emoji"])
            st.session_state.stage = "voice"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Style phrase buttons
    st.markdown("""
    <style>
    button[data-testid*="phrase-"] {
        width: 100% !important;
        min-height: 80px !important;
        border-radius: 16px !important;
        background: white !important;
        border: 3px solid rgba(255,255,255,0.8) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
        font-size: clamp(1rem, 4vw, 1.3rem) !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        display: flex !important;
        align-items: center !important;
        gap: 1rem !important;
        padding: 1rem 1.5rem !important;
        text-align: left !important;
        transition: all 0.3s ease !important;
        margin-bottom: 1.2rem !important;
    }
    button[data-testid*="phrase-"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15) !important;
        border-color: #4facfe !important;
    }
    button[data-testid*="phrase-"]:active {
        transform: translateY(-1px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation buttons container
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.session_state.selected_category
        if st.button("🔄 Different options", key="none_btn"):
            fetch_options(category)
    
    with col2:
        if st.button("← Back", key="back_categories"):
            st.session_state.stage = "categories"
            st.rerun()
    
    # Style navigation buttons
    st.markdown("""
    <style>
    button[data-testid*="none_btn"], button[data-testid*="back_categories"] {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 50px !important;
        font-size: clamp(0.9rem, 3vw, 1.1rem) !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
        min-height: 50px !important;
        backdrop-filter: blur(10px) !important;
        margin-top: 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    button[data-testid*="none_btn"]:hover, button[data-testid*="back_categories"]:hover {
        background: rgba(255,255,255,0.3) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_voice_output() -> None:
    """Render celebration voice output screen"""
    if not st.session_state.last_phrase:
        st.session_state.stage = "phrases"
        st.rerun()
        return
    
    # Voice celebration display
    st.markdown(f"""
    <div class="voice-celebration">
        <div class="voice-icon">🔊</div>
        <div class="voice-text">"{st.session_state.last_phrase}"</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-play audio
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, autoplay=True)
    
    # Button container
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔊 Say it again", key="play_again"):
            if st.session_state.audio_file:
                st.audio(st.session_state.audio_file, autoplay=True)
    
    with col2:
        if st.button("← Speak something else", key="back_home"):
            reset_flow()
            st.rerun()
    
    # Style buttons
    st.markdown("""
    <style>
    button[data-testid*="play_again"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        border: 4px solid rgba(255,255,255,0.3) !important;
        border-radius: 50px !important;
        font-size: clamp(1rem, 3vw, 1.3rem) !important;
        font-weight: 700 !important;
        padding: 1.2rem 2rem !important;
        min-height: 70px !important;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.4) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    button[data-testid*="play_again"]:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 12px 40px rgba(79, 172, 254, 0.6) !important;
    }
    button[data-testid*="back_home"] {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 50px !important;
        font-size: clamp(0.9rem, 3vw, 1.1rem) !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
        min-height: 50px !important;
        backdrop-filter: blur(10px) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    button[data-testid*="back_home"]:hover {
        background: rgba(255,255,255,0.3) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Main render ----------------------------------------------------------- #


def main() -> None:
    init_session_state()
    
    # Check for GPS data in query parameters (from JavaScript geolocation)
    query_params = st.query_params
    
    # Try to get from query params first
    if "lat" in query_params and "lng" in query_params:
        try:
            lat_str = query_params.get("lat")
            lng_str = query_params.get("lng")
            # Handle both single values and lists
            lat = float(lat_str[0] if isinstance(lat_str, list) else lat_str)
            lng = float(lng_str[0] if isinstance(lng_str, list) else lng_str)
            
            if lat and lng:
                st.session_state.latitude = lat
                st.session_state.longitude = lng
                st.session_state.gps_requested = False
                
                # Clear query params
                new_params = {k: v for k, v in query_params.items() if k not in ["lat", "lng", "gps_updated"]}
                st.query_params.clear()
                for k, v in new_params.items():
                    st.query_params[k] = v
                
                st.rerun()
        except (ValueError, TypeError, IndexError) as e:
            st.warning(f"Error parsing GPS coordinates from URL: {e}")
    
    
    render_header()

    stage = st.session_state.stage
    if stage == "intro":
        render_stage_intro()
    elif stage == "categories":
        render_categories()
    elif stage == "phrases":
        render_phrase_options()
    elif stage == "voice":
        render_voice_output()
    
    # Close the autism containers
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Emergency button on all screens except voice output
    render_emergency_button()


if __name__ == "__main__":
    main()