"""Orchestrator agent that coordinates the full ViMax pipeline.

This module ties together the video analyzer, scene description agent,
camera image generator, and best image selector into a single cohesive
pipeline for generating optimal camera views from video input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agents.video_analyzer import VideoAnalyzer, VideoAnalysisResult
from agents.scene_description_agent import SceneDescriptionAgent, SceneDescription
from agents.camera_image_generator import CameraImageGenerator
from agents.best_image_selector import select_best_image

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Aggregated result produced by the orchestration pipeline."""

    video_path: str
    analysis: VideoAnalysisResult
    scene_description: SceneDescription
    candidate_image_paths: list[str] = field(default_factory=list)
    best_image_path: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True when a best image was successfully selected."""
        return self.best_image_path is not None


class Orchestrator:
    """High-level coordinator for the ViMax camera-view generation pipeline.

    Typical usage::

        orchestrator = Orchestrator()
        result = orchestrator.run("path/to/video.mp4")
        print(result.best_image_path)
    """

    def __init__(
        self,
        # Bumped num_frames from 8 to 12 — more frames gives noticeably better
        # scene descriptions for longer/more dynamic clips in my testing.
        num_frames: int = 12,
        # Increased candidates from 5 to 8 so the selector has a wider pool to
        # choose from; quality of the best pick improves with more options.
        num_camera_candidates: int = 8,
        output_dir: str = "outputs",
    ) -> None:
        """
        Args:
            num_frames: Number of frames to sample from the video for analysis.
            num_camera_candidates: How many camera-view images to generate before
                selecting the best one.
            output_dir: Directory where generated images will be saved.
        """
        self.num_frames = num_frames
        self.num_camera_candidates = num_camera_candidates
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._video_analyzer = VideoAnalyzer(num_frames=num_frames)
        self._scene_agent = SceneDescriptionAgent()
        self._camera_generator = CameraImageGenerator(output_dir=str(self.output_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, video_path: str) -> PipelineResult:
        """Execute the full pipeline for *video_path*.

        Steps:
            1. Analyse the video to extract key frames and metadata.
            2. Generate a structured scene description from those frames.
            3. Produce multiple candidate camera-view images.
            4. Select the single best image from the candidates.

        Args:
            video_path: Absolute or relative path to the input video file.

        Return
