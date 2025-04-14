"""
Voice mappings for different languages with their best quality male and female voices.
Each language has a tuple of (female_voice, male_voice).
If a gender is not available for a language, None is used.
"""

VOICE_MAPPING = {
    "en-us": ("af_bella", "am_fenrir"),      # American English
    "en-gb": ("bf_emma", "bm_fable"),        # British English
    "ja": ("jf_alpha", "jm_kumo"),           # Japanese
    "zh": ("zf_xiaobei", "zm_yunjian"),      # Mandarin Chinese
    "es": ("ef_dora", "em_alex"),            # Spanish
    "fr": ("ff_siwis", None),                # French
    "hi": ("hf_alpha", "hm_omega"),          # Hindi
    "it": ("if_sara", "im_nicola"),          # Italian
    "pt-br": ("pf_dora", "pm_alex"),         # Brazilian Portuguese
}

def get_voice(language: str, gender: str = "female") -> str:
    """
    Get the best quality voice for a given language and gender.
    
    Args:
        language (str): Language code (e.g., 'en-us', 'ja', 'fr')
        gender (str): Either 'female' or 'male'
    
    Returns:
        str: Voice ID or None if not available
    """
    if language not in VOICE_MAPPING:
        raise ValueError(f"Language '{language}' not supported")
    
    if gender.lower() not in ["female", "male"]:
        raise ValueError("Gender must be either 'female' or 'male'")
    
    voices = VOICE_MAPPING[language]
    return voices[0] if gender.lower() == "female" else voices[1] 