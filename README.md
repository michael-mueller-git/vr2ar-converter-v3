# VR2AR Converter 3

Convert your adult VR Videos into Passthrough AR Videos using advanced background removal.

## Project Evolution
- **Difference from v1**: v3 solves mask merge jitter problems by implementing a custom `ArVideoWriter`. See [vr2ar-converter](https://github.com/michael-mueller-git/vr2ar-converter).
- **Difference from v2**: v3 uses more modern models for background removal ([MatAnyone](https://github.com/pq-yang/MatAnyone)) and replaces the v2 container. See [vr2ar-converter-v2](https://github.com/michael-mueller-git/vr2ar-converter-v2).

## Deployment

The only supported deployment method is via Docker to ensure all CUDA, FFmpeg, and model dependencies are correctly bundled.

### 1. Setup Configuration
Depending on your Docker/NVIDIA setup (CDI vs standard), you may need to use either the `cuda1` or `cuda2` example. Copy the appropriate one for your hardware:
```bash
cp docker-compose.cuda1.yaml.example docker-compose.yaml
# OR (for CDI setups)
cp docker-compose.cuda2.yaml.example docker-compose.yaml
```

### 2. Run the Application
Deploy using Docker Compose:
```bash
docker compose up -d
```

## Usage

1. **Upload**: Upload your VR video chunks (recommended length < 3 minutes).
2. **Configuration**: Select the source projection (e.g., Equirectangular `eq`) and adjust the mask size based on your available VRAM (e.g., 1440px requires ~20GB).
3. **Masking**: 
   - Extract projection frames.
   - Generate initial masks using text prompts or point selections.
   - Refine masks using the "add" or "subtract" tools.
4. **Process**: Add the video to the job queue. The system will propagate the mask temporally and merge the result.
5. **Download**: Download the converted AR video once processing is complete.

## Requirements
- NVIDIA GPU with CUDA support.
- Docker and NVIDIA Container Toolkit.
