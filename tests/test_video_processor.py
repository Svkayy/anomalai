import inspect
import video_processor


def test_extract_frames_accepts_frame_interval_param():
    """The VideoProcessor.extract_frames method must accept frame_interval."""
    sig = inspect.signature(video_processor.VideoProcessor.extract_frames)
    assert "frame_interval" in sig.parameters
