"""Scene description agent for generating detailed textual descriptions of video scenes."""

from dataclasses import dataclass
from typing import Optional
import base64
import logging
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SceneDescription:
    """Holds the result of a scene description analysis."""

    timestamp: float
    description: str
    objects_detected: list[str]
    scene_type: str
    confidence: float


class SceneDescriptionAgent:
    """Agent responsible for generating rich textual descriptions of video frames/scenes.

    Uses a multimodal LLM to analyze individual frames and produce structured
    scene descriptions including detected objects and scene classification.
    """

    SCENE_PROMPT = (
        "Analyze this image and provide:\n"
        "1. A concise but detailed description of the scene (2-3 sentences).\n"
        "2. A comma-separated list of key objects visible.\n"
        "3. A single scene type label (e.g., 'indoor', 'outdoor', 'urban', 'nature', "
        "'crowd', 'traffic', 'office', 'other').\n\n"
        "Respond in the following format exactly:\n"
        "DESCRIPTION: <scene description>\n"
        "OBJECTS: <obj1, obj2, obj3, ...>\n"
        "SCENE_TYPE: <scene type>\n"
    )

    def __init__(self, client, model: str = "gpt-4o"):
        """Initialize the SceneDescriptionAgent.

        Args:
            client: An OpenAI-compatible client instance.
            model: The multimodal model identifier to use for inference.
        """
        self.client = client
        self.model = model

    def _encode_image(self, image: Image.Image, max_size: tuple[int, int] = (768, 768)) -> str:
        """Encode a PIL Image to a base64 JPEG string suitable for API submission.

        Args:
            image: The PIL Image to encode.
            max_size: Maximum (width, height) to resize the image to before encoding.

        Returns:
            Base64-encoded JPEG string.
        """
        image = image.copy()
        image.thumbnail(max_size, Image.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_response(self, text: str) -> tuple[str, list[str], str]:
        """Parse the structured LLM response into its components.

        Args:
            text: Raw response text from the LLM.

        Returns:
            Tuple of (description, objects_list, scene_type).
        """
        description = ""
        objects: list[str] = []
        scene_type = "other"

        for line in text.strip().splitlines():
            if line.startswith("DESCRIPTION:"):
                description = line[len("DESCRIPTION:"):].strip()
            elif line.startswith("OBJECTS:"):
                raw_objects = line[len("OBJECTS:"):].strip()
                objects = [o.strip() for o in raw_objects.split(",") if o.strip()]
            elif line.startswith("SCENE_TYPE:"):
                scene_type = line[len("SCENE_TYPE:"):].strip().lower()

        return description, objects, scene_type

    def describe(self, image: Image.Image, timestamp: float = 0.0) -> Optional[SceneDescription]:
        """Generate a structured description for the given image frame.

        Args:
            image: A PIL Image representing the video frame to describe.
            timestamp: The timestamp (in seconds) of the frame within the video.

        Returns:
            A SceneDescription dataclass instance, or None if the request fails.
        """
        try:
            encoded = self._encode_image(image)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.SCENE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.2,
            )

            raw_text = response.choices[0].message.content or ""
            description, objects, scene_type = self._parse_response(raw_text)

            # Use finish_reason as a rough confidence proxy
            finish_reason = response.choices[0].finish_reason
            confidence = 1.0 if finish_reason == "stop" else 0.7

            return SceneDescription(
                timestamp=timestamp,
                description=description,
                objects_detected=objects,
                scene_type=scene_type,
                confidence=confidence,
            )

        except Exception as exc:
            logger.error("SceneDescriptionAgent failed for timestamp %.2fs: %s", timestamp, exc)
            return None
