from .screenwriter import Screenwriter
from .storyboard_artist import StoryboardArtist
from .camera_image_generator import CameraImageGenerator
from .character_extractor import CharacterExtractor
from .character_portraits_generator import CharacterPortraitsGenerator
from .reference_image_selector import ReferenceImageSelector

# Agent pipeline order for reference:
# 1. CharacterExtractor      - extract characters from script
# 2. Screenwriter            - generate full screenplay
# 3. StoryboardArtist        - break screenplay into shots
# 4. CharacterPortraitsGenerator - generate character reference images
# 5. ReferenceImageSelector  - pick best reference per shot
# 6. CameraImageGenerator    - render final camera images

__all__ = [
    "Screenwriter",
    "StoryboardArtist",
    "CameraImageGenerator",
    "CharacterExtractor",
    "CharacterPortraitsGenerator",
    "ReferenceImageSelector",
]
