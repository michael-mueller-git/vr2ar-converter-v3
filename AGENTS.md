# AGENTS.md

## Architecture
- **Purpose**: Converts VR videos to Passthrough AR videos via background removal.
- **Workflow**: Video $\rightarrow$ Frame Extraction $\rightarrow$ Mask Generation (SAM2) $\rightarrow$ Temporal Propagation (MatAnyone) $\rightarrow$ FFmpeg Merge.
- **Key Components**:
  - `main.py`: Gradio UI and asynchronous Job Queue.
  - `sam2_executor.py`: Initial mask generation via SAM2/GroundingDINO.
  - `data/ArVideoWriter.py`: Custom writer to prevent merge jitter.
- **Models**: MatAnyone/MatAnyone 2. Checkpoints expected in `/app/model`.

## Developer Commands
- **Run**: `python main.py`
- **Install**: `pip install -r requirements.txt`
- **Verify**: Use scripts in `test/` (e.g., `test_mask.py`, `merge.py`) for manual masking/merging checks.

## Operational Gotchas
- **VRAM**: Mask size is critical; 1440px requires $\sim$20GB VRAM.
- **Job Queue**: Uses `.pkl` files in `/jobs/`. Jobs must match `JOB_VERSION` in `main.py` to be processed.
- **Model Paths**: Discrepancy between `main.py` (`/app/model`) and `sam2_executor.py` (`model/`).
- **Projection**: `eq` (equirectangular) is converted to fisheye by default unless `keepEq` is true.

## Conventions
- **Temporary Files**: stored in `process/frames`, `process/masks`, and `process/debug`.
- **Masking**: Grayscale (`L` mode) PIL images.
- **FFmpeg**: Relies on complex filters for projection and alpha merging.
