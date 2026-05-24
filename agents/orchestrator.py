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
        num_frames: int = 8,
        num_camera_candidates: int = 5,
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

        Returns:
            A :class:`PipelineResult` containing all intermediate artefacts
            and the path to the best generated image.
        """
        logger.info("[Orchestrator] Starting pipeline for: %s", video_path)

        # Step 1 — video analysis
        logger.info("[Orchestrator] Step 1/4 — analysing video")
        analysis: VideoAnalysisResult = self._video_analyzer.analyze(video_path)

        # Step 2 — scene description
        logger.info("[Orchestrator] Step 2/4 — generating scene description")
        scene_description: SceneDescription = self._scene_agent.describe(
            frames=analysis.frames,
            metadata=analysis.metadata,
        )

        # Step 3 — camera image generation
        logger.info(
            "[Orchestrator] Step 3/4 — generating %d camera candidates",
            self.num_camera_candidates,
        )
        candidate_paths: list[str] = []
        for i in range(self.num_camera_candidates):
            image_path = self._camera_generator.get_new_camera_image(
                scene_description=scene_description,
                index=i,
            )
            if image_path:
                candidate_paths.append(image_path)
                logger.debug("[Orchestrator] Candidate %d saved to %s", i, image_path)

        # Step 4 — best image selection
        logger.info("[Orchestrator] Step 4/4 — selecting best image")
        best: Optional[str] = None
        if candidate_paths:
            best = select_best_image(
                candidate_paths=candidate_paths,
                scene_description=scene_description,
            )
            logger.info("[Orchestrator] Best image selected: %s", best)
        else:
            logger.warning("[Orchestrator] No candidate images were generated.")

        return PipelineResult(
            video_path=video_path,
            analysis=analysis,
            scene_description=scene_description,
            candidate_image_paths=candidate_paths,
            best_image_path=best,
        )
