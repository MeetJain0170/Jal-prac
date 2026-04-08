"""
video/video_processor.py — Video pipeline entry point.

This module is imported by `api.py` as `from video.video_processor import process_video`.

Implementation lives in `outputs/videos/video_processor.py`; this file re-exports
the public API so imports stay stable.
"""

from __future__ import annotations

from outputs.videos.video_processor import process_video

