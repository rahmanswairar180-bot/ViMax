"""Video analyzer agent for ViMax.

This module provides functionality to analyze video content by extracting
frames and selecting the most representative/informative images using
the BestImageSelector agent.
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
from pydantic import BaseModel

from agents.best_image_selector import BestImageSelector


class VideoAnalysisResult(BaseModel):
    """Result of a video analysis operation."""

    video_path: str
    total_frames: int
    sampled_frames: int
    best_frame_index: int
    best_frame_timestamp: float  # in seconds
    best_frame_b64: str
    analysis_summary: str


class VideoAnalyzer:
    """Analyzes video files by sampling frames and selecting the best one.

    Uses OpenCV for frame extraction and delegates image selection
    to the BestImageSelector agent.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_frames_to_sample: int = 8,
        sample_strategy: str = "uniform",
    ):
        """
        Initialize the VideoAnalyzer.

        Args:
            model: The LLM model to use for image selection.
            max_frames_to_sample: Maximum number of frames to extract from the video.
            sample_strategy: Strategy for sampling frames ('uniform' or 'scene_change').
        """
        self.model = model
        self.max_frames_to_sample = max_frames_to_sample
        self.sample_strategy = sample_strategy
        self.image_selector = BestImageSelector(model=model)

    def _extract_frames_uniform(
        self, video_path: str, num_frames: int
    ) -> list[tuple[int, float, str]]:
        """Extract frames uniformly distributed across the video.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to extract.

        Returns:
            List of tuples: (frame_index, timestamp_seconds, base64_encoded_image)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # fallback default

        # Calculate evenly spaced frame indices
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]

        extracted = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Encode frame to JPEG and then base64
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue

            b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
            timestamp = idx / fps
            extracted.append((idx, timestamp, b64_image))

        cap.release()
        return extracted

    def analyze(
        self,
        video_path: str,
        query: Optional[str] = None,
    ) -> VideoAnalysisResult:
        """Analyze a video file and return the best representative frame.

        Args:
            video_path: Path to the video file to analyze.
            query: Optional query/context to guide frame selection.

        Returns:
            VideoAnalysisResult containing the best frame and metadata.

        Raises:
            ValueError: If the video file cannot be opened or has no frames.
            FileNotFoundError: If the video file does not exist.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Extract frames from video
        frames = self._extract_frames_uniform(video_path, self.max_frames_to_sample)

        if not frames:
            raise ValueError(f"No frames could be extracted from: {video_path}")

        # Get total frame count for metadata
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Use BestImageSelector to pick the most informative frame
        b64_images = [f[2] for f in frames]
        selection_query = query or "Select the most informative and representative frame."

        selector_result = self.image_selector.select(
            images=b64_images,
            query=selection_query,
        )

        best_local_idx = selector_result.best_image_index
        best_frame_idx, best_timestamp, best_b64 = frames[best_local_idx]

        return VideoAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            sampled_frames=len(frames),
            best_frame_index=best_frame_idx,
            best_frame_timestamp=best_timestamp,
            best_frame_b64=best_b64,
            analysis_summary=selector_result.reasoning,
        )
