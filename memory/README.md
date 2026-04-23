# Memory

Auto-annotation pipeline for egocentric video clips. Runs five steps on each clip and maintains a cross-clip global memory of speakers, objects, and events.

## Pipeline

| Step | Purpose | Output |
|---|---|---|
| `asr` | Extract audio + transcribe speech | `data/voices/asr/{event_id}.json` |
| `caption` | LLM generates event caption + interaction targets | `data/events/DAY{n}/{event_id}.json` |
| `entity` | Grounding DINO + SAM2 detect, segment, and track interaction objects; global match across clips | `data/entities/event/{event_id}/`, `data/entities/crops/`, updated event JSON |
| `voiceprint` | Extract voiceprint embeddings and match against global voice DB | `data/voices/embedding/`, `data/voices/voiceprint/database.json`, updated event JSON |
| `relation` | LLM reasons about temporal / causal / activity relations with past events | `data/relationships/{event_id}.json`, updated event JSON |

Steps are selected via `pipeline.steps` in the config YAML.

## Dataset

Download the EgoLife clips from [HuggingFace](https://huggingface.co/datasets/lmms-lab/EgoLife/tree/main) and set the local path as `paths.data_root` in `configs/config.yaml`.

## Submodules

```bash
cd submodules/Grounded-SAM-2

# Grounding DINO
pip install -e grounding_dino

# SAM2
pip install -e .
```

WhisperX (optional, only for local ASR; skip if using API-based ASR):

```bash
cd submodules/whisperX
pip install -e .
```

## Download

```bash
# SAM2 checkpoint
mkdir -p models/entity/sam2 && cd models/entity/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ../../..

# Grounding DINO Tiny (HuggingFace)
git lfs install
git clone https://huggingface.co/IDEA-Research/grounding-dino-tiny models/entity/grounding-dino-tiny

# CLIP ViT-B/32
mkdir -p models/clip && cd models/clip
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
cd ../..

# ERes2Net voiceprint (download from https://github.com/modelscope/3D-Speaker)
# Place the .ckpt file at models/voice/pretrained_eres2netv2.ckpt
```

## Running

Single video:
```bash
python scripts/run_single_video.py \
  --video_path /path/to/clip.mp4 \
  --config configs/config.yaml
```

Batch over a directory (filenames `DAY{n}_{HHMMSS}.mp4`):
```bash
python scripts/run_video_dir.py \
  --video_dir /path/to/videos \
  --config configs/config.yaml \
  --start_idx 0 --num_videos 10
```

To run only a subset of steps, pass `--steps asr caption` etc.

## Configs

- `configs/config.yaml` — main pipeline config (paths, API, per-step options)
- `configs/prompts/` — prompt templates for caption / relation / voice diarization

Fill in placeholders (`YOUR_OPENAI_API_KEY_HERE`, `YOUR_GEMINI_API_KEY_HERE`, `YOUR_HF_TOKEN_HERE`) before running.

## Outputs

```
data/
├── events/DAY{n}/              # event JSONs (pipeline's main artifact)
├── entities/event/             # per-event object tracks
├── entities/crops/             # object crop images
├── entities/global_entities.json
├── features/embeddings/        # caption text embeddings
├── voices/asr/                 # ASR transcripts
├── voices/embedding/           # speaker embeddings
├── voices/voiceprint/          # global voice DB
└── relationships/              # event-relation JSONs
```
