"""
Configuration management.

- Loads config from a YAML file
- Exposes typed accessors for individual settings
- Reads prompt template files
"""

import os
import yaml
from typing import Optional, Dict, Any


class Config:
    """Configuration container."""

    def __init__(self, config_dict: dict):
        """
        Build a Config from a parsed YAML dict.

        Args:
            config_dict: parsed YAML
        """
        self._config = config_dict

        # ====================================================================
        # Pipeline
        # ====================================================================
        pipeline_config = config_dict["pipeline"]
        self.pipeline_steps = pipeline_config["steps"]
        self.temp_dir = pipeline_config["temp_dir"]

        # ====================================================================
        # Paths (must initialize first; everything else may use path expansion)
        # ====================================================================
        paths_config = config_dict["paths"]
        self.data_root = paths_config["data_root"]
        self.output_root = paths_config["output_root"]
        self.prompts_dir = paths_config["prompts_dir"]
        # Shared model root (where all model files live)
        self.models_root = paths_config["models_root"]

        # Expand path variables
        self.events_dir = self._expand_path(paths_config["events_dir"])
        self.entities_dir = self._expand_path(paths_config["entities_dir"])
        self.features_dir = self._expand_path(paths_config["features_dir"])
        self.index_dir = self._expand_path(paths_config["index_dir"])

        # ====================================================================
        # Caption
        # ====================================================================
        caption_config = config_dict["caption"]
        self.caption_client = caption_config["client"]

        # Text embedding
        self.use_text_embedding = caption_config["use_text_embedding_client"]

        # Interaction-target filter
        self.min_interaction_duration = caption_config["min_interaction_duration"]

        # Pull sample_fps and max_frames from the active client's config block
        client_name = self.caption_client.lower()
        client_config = caption_config[client_name]
        self.sample_fps = client_config["sample_fps"]
        # max_frames only applies to gpt4o (frame-extraction mode);
        # gemini/gpt5 use direct video compression and don't need it
        self.max_frames = client_config["max_frames"] if "max_frames" in client_config else None

        # ====================================================================
        # Voice
        # ====================================================================
        voice_config = config_dict["voice"]

        # ASR
        asr_config = voice_config["asr"]
        self.asr_client = asr_config["client"]
        self.min_segment_duration = asr_config["min_segment_duration"]
        self.min_word_count = asr_config["min_word_count"]

        # Voiceprint
        voiceprint_config = voice_config["voiceprint"]
        # Expand model-path variables (e.g. ${paths.models_root})
        self.voiceprint_model_path = self._expand_path(voiceprint_config["model_path"])
        self.voiceprint_embedding_size = voiceprint_config["embedding_size"]
        self.voiceprint_fbank_dim = voiceprint_config["fbank_dim"]
        self.voiceprint_sample_rate = voiceprint_config["sample_rate"]
        self.voiceprint_device = voiceprint_config["device"]
        self.voiceprint_reset_database = voiceprint_config.get("reset_database", False)

        # Per-checkpoint architecture parameters dict (look up by model_path).
        # Expand path variables in the dict keys too.
        raw_model_configs = voiceprint_config["model_configs"]
        self.voiceprint_model_configs = {
            self._expand_path(key): value
            for key, value in raw_model_configs.items()
        }

        # Segment processing
        self.voiceprint_min_segment_duration = voiceprint_config["min_segment_duration"]

        # Matching
        self.voiceprint_match_threshold = voiceprint_config["match_threshold"]
        self.voiceprint_top_k = voiceprint_config["top_k"]
        self.voiceprint_max_history_features = voiceprint_config["max_history_features"]

        # EMA update
        self.voiceprint_ema_alpha = voiceprint_config["ema_alpha"]
        self.voiceprint_ema_update_threshold = voiceprint_config["ema_update_threshold"]

        # History feature filter
        self.voiceprint_min_history_duration = voiceprint_config["min_history_duration"]
        self.voiceprint_min_history_match_score = voiceprint_config["min_history_match_score"]

        # Debug
        self.voiceprint_debug_save_segments = voiceprint_config["debug_save_segments"]
        self.voiceprint_debug_save_dir = self._expand_path(voiceprint_config["debug_save_dir"])

        # Voiceprint output paths
        self.voice_database_path = self._expand_path(voiceprint_config["database_path"])
        self.voiceprint_embedding_dir = self._expand_path(voiceprint_config["embedding_dir"])
        self.voiceprint_asr_dir = self._expand_path(voiceprint_config["asr_dir"])
        self.voiceprint_merged_wav_dir = self._expand_path(voiceprint_config["merged_wav_dir"])

        # ====================================================================
        # Prompts
        # ====================================================================
        prompts_config = config_dict["prompts"]
        self.event_annotation_prompt_file = prompts_config["event_annotation"]
        self.voice_diarization_prompt_file = prompts_config["voice_diarization"]
        self.event_relation_prompt_file = prompts_config["event_relation"]

        # ====================================================================
        # Processing
        # ====================================================================
        processing_config = config_dict["processing"]
        self.batch_size = processing_config["batch_size"]
        self.save_intermediate = processing_config["save_intermediate"]
        self.verbose = processing_config["verbose"]
        self.base_date = processing_config["base_date"]

        # ====================================================================
        # Event-relation reasoning
        # ====================================================================
        relation_config = config_dict["relation"]
        self.relation_client = relation_config["client"]  # e.g. "gemini"
        self.relation_window_size = relation_config["window_size"]
        self.relation_time_threshold = relation_config["time_threshold"]

        # ====================================================================
        # Entity tracking and feature management
        # ====================================================================
        entity_config = config_dict["entity"]

        # Model root (expand path variables, e.g. ${paths.models_root})
        self.entity_models_dir = self._expand_path(entity_config["models_dir"])

        # Video preprocessing (used by entity tracking)
        self.entity_sample_fps = entity_config["sample_fps"]
        self.entity_video_max_size = entity_config["video_max_size"]
        self.entity_device = entity_config["device"]

        # Grounded-SAM-2
        grounded_sam2_config = entity_config["grounded_sam2"]

        # SAM2 checkpoint (supports relative or absolute)
        sam2_checkpoint = grounded_sam2_config["sam2_checkpoint"]
        if os.path.isabs(sam2_checkpoint):
            self.grounded_sam2_checkpoint = sam2_checkpoint
        else:
            self.grounded_sam2_checkpoint = os.path.join(self.entity_models_dir, sam2_checkpoint)

        # SAM2 config file (usually a Hydra-style path)
        self.grounded_sam2_config = grounded_sam2_config["sam2_config"]

        # Grounding DINO checkpoint (supports relative or absolute)
        grounding_model = grounded_sam2_config["grounding_model"]
        if os.path.isabs(grounding_model):
            self.grounding_model_id = grounding_model
        else:
            self.grounding_model_id = os.path.join(self.entity_models_dir, grounding_model)

        # Use only local model files (no online download / verification)
        self.local_files_only = grounded_sam2_config["local_files_only"]

        self.detection_threshold = grounded_sam2_config["detection_threshold"]
        self.text_threshold = grounded_sam2_config["text_threshold"]
        self.max_detection_retry_frames = grounded_sam2_config["max_detection_retry_frames"]
        self.first_appearance_buffer = grounded_sam2_config["first_appearance_buffer"]
        self.segment_buffer = grounded_sam2_config["segment_buffer"]

        # Note: detailed feature/models config is read via config._config["entity"]

        # Global object matching
        global_matching_config = entity_config["global_matching"]
        self.entity_global_matching_enabled = global_matching_config["enabled"]
        self.entity_global_matching_match_threshold = global_matching_config["match_threshold"]
        self.entity_global_matching_ema_alpha = global_matching_config["ema_alpha"]
        self.entity_global_matching_max_history_events = global_matching_config["max_history_events"]
        self.entity_global_matching_min_confidence = global_matching_config["min_confidence"]
        self.entity_global_objects_dir = self._expand_path(global_matching_config["global_objects_dir"])
        self.entity_global_features_dir = self._expand_path(global_matching_config["global_features_dir"])

        # Note: text2text and vision2vision details are read via config._config["entity"]["global_matching"]

        # Entity output paths
        entity_output_config = entity_config["output"]
        self.event_entities_dir = self._expand_path(entity_output_config["event_entities_dir"])
        self.global_entities_path = self._expand_path(entity_output_config["global_entities_path"])
        self.event_features_dir = self._expand_path(entity_output_config["event_features_dir"])
        self.global_features_dir = self._expand_path(entity_output_config["global_features_dir"])

        # ====================================================================
        # Entity tracking (used by ObjectTracker)
        # ====================================================================
        if "entity_tracking" in config_dict:
            self.entity_tracking = config_dict["entity_tracking"]
        else:
            # Fall back to None if not provided
            self.entity_tracking = None

    def _expand_path(self, path: str) -> str:
        """
        Expand variable references in a path.

        Supported variable form: ${paths.xxx}

        Args:
            path: path string, possibly containing ${...} variables

        Returns:
            Fully expanded path (all variables resolved).

        Raises:
            ValueError: if the path still contains unknown variable references.
        """
        if path is None:
            return None

        # Keep the original for error messages
        original_path = path

        # Resolve in dependency order.
        # 1. Resolve base paths (output_root and models_root)
        if "${paths.output_root}" in path:
            path = path.replace("${paths.output_root}", self.output_root)

        if "${paths.models_root}" in path:
            path = path.replace("${paths.models_root}", self.models_root)

        # 2. Resolve entities_dir (depends on output_root)
        if "${paths.entities_dir}" in path:
            # If entities_dir hasn't been initialized yet, build it directly
            if not hasattr(self, 'entities_dir'):
                base_entities_dir = os.path.join(self.output_root, "entities")
            else:
                # Already initialized; make sure it is fully expanded
                base_entities_dir = self.entities_dir
                # Recursively expand if entities_dir itself still contains a variable
                if "${paths.output_root}" in base_entities_dir:
                    base_entities_dir = base_entities_dir.replace("${paths.output_root}", self.output_root)
            path = path.replace("${paths.entities_dir}", base_entities_dir)

        # 3. Resolve features_dir (depends on output_root)
        if "${paths.features_dir}" in path:
            # If features_dir hasn't been initialized yet, build it directly
            if not hasattr(self, 'features_dir'):
                base_features_dir = os.path.join(self.output_root, "features")
            else:
                # Already initialized; make sure it is fully expanded
                base_features_dir = self.features_dir
                # Recursively expand if features_dir itself still contains a variable
                if "${paths.output_root}" in base_features_dir:
                    base_features_dir = base_features_dir.replace("${paths.output_root}", self.output_root)
            path = path.replace("${paths.features_dir}", base_features_dir)

        # 4. Final check: ensure no unresolved variables remain (defensive)
        if "${" in path:
            raise ValueError(
                f"Path expansion failed: unresolved variable.\n"
                f"  original: {original_path}\n"
                f"  current:  {path}\n"
                f"  Hint: check that the variable references in your config are valid."
            )

        return path

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load a Config from a YAML file.

        Args:
            yaml_path: path to the YAML config file

        Returns:
            Config instance.
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def get_api_config(self, model_name: str) -> Dict[str, str]:
        """
        Return the API config (base_url + api_key) for a given model.

        Args:
            model_name: e.g. "gpt-4o", "gemini-2.5-pro", "embedding"

        Returns:
            Dict with base_url and api_key.
        """
        api_config = self._config["api"]
        if model_name not in api_config:
            raise ValueError(f"Unknown model name in API config: {model_name}")

        return {
            "base_url": api_config[model_name]["base_url"],
            "api_key": api_config[model_name]["api_key"]
        }

    def get_caption_config(self, client: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the full caption config for a client (API + model parameters merged).

        Args:
            client: client name; defaults to the configured caption client.

        Returns:
            Merged dict of API config + client-specific config.
        """
        if client is None:
            client = self.caption_client

        caption_config = self._config["caption"]
        client_config = caption_config[client]

        # Pull model name
        model_name = client_config["model"]

        # Pull API config
        api_config = self.get_api_config(model_name)

        # Merge: API config + client config
        full_config = {**api_config, **client_config}

        return full_config

    def get_asr_config(self, client: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the full ASR config for a client (API + model parameters merged).

        Args:
            client: client name; defaults to the configured ASR client.
                    Supported: "gpt4o", "gemini", "whisperx"

        Returns:
            Merged dict of API config + client-specific config.
        """
        if client is None:
            client = self.asr_client

        voice_config = self._config["voice"]
        asr_config = voice_config["asr"]
        client_config = asr_config[client]

        # WhisperX runs locally; no API config needed
        if client == "whisperx":
            # Expand path variables (e.g. ${paths.models_root})
            download_root = self._expand_path(client_config["download_root"])
            whisper_model_path = client_config.get("whisper_model_path")
            if whisper_model_path:
                whisper_model_path = self._expand_path(whisper_model_path)
            diarization_model_path = client_config.get("diarization_model_path")
            if diarization_model_path:
                diarization_model_path = self._expand_path(diarization_model_path)

            return {
                "model": client_config["model"],
                "device": client_config["device"],
                "compute_type": client_config["compute_type"],
                "batch_size": client_config["batch_size"],
                "enable_diarization": client_config["enable_diarization"],
                "hf_token": client_config["hf_token"],
                "min_speakers": client_config["min_speakers"],
                "max_speakers": client_config["max_speakers"],
                "download_root": download_root,            # local model download path
                "whisper_model_path": whisper_model_path,  # local Whisper model path
                "diarization_model_path": diarization_model_path,  # local diarization model path
                "kwargs": client_config["kwargs"]
            }

        # GPT-4o and Gemini need API config
        # Pull model name
        model_name = client_config["model"]

        # Pull API config
        api_config = self.get_api_config(model_name)

        # Merge: API config + client config
        full_config = {**api_config, **client_config}

        return full_config

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Return the embedding config (API + model parameters).

        Returns:
            Merged dict of API config + model parameters.
        """
        # Pull API config
        api_config = self.get_api_config("embedding")

        # Add other embedding params here if needed in the future.
        # For now only the API config is needed.
        return api_config

    def get_relation_config(self, client: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the full event-relation config for a client (API + model parameters merged).

        Args:
            client: client name; defaults to the configured relation client.
                    Currently supported: "gemini"

        Returns:
            Merged dict of API config + client-specific config.
        """
        if client is None:
            client = self.relation_client

        relation_config = self._config["relation"]
        client_config = relation_config[client]

        # Pull model name
        model_name = client_config["model"]

        # Pull API config
        api_config = self.get_api_config(model_name)

        # Merge: API config + client config
        full_config = {**api_config, **client_config}

        return full_config

    def get_text_embedding_config(self) -> Dict[str, Any]:
        """
        Return the text-embedding config (used for caption embeddings).

        Returns:
            Merged dict of API config + text_embedding_client config.
        """
        caption_config = self._config["caption"]
        text_emb_config = caption_config["text_embedding_client"]

        # Pull model name
        model_name = text_emb_config["model"]

        # Pull API config (keyed by model name)
        api_config = self.get_api_config(model_name)

        # Merge: API config + text_embedding_client config
        full_config = {**api_config, **text_emb_config}

        return full_config

    def get_prompt(self, prompt_name: str, client: Optional[str] = None) -> tuple:
        """
        Read a prompt template (.txt or .py), with optional per-client variants.

        Args:
            prompt_name: one of "event_annotation", "voice_diarization", "event_relation"
            client: client name; defaults to the configured client for that prompt.

        Returns:
            (system_message, user_prompt) tuple.
            - .txt files return (None, file_contents)
            - .py files return (SYSTEM_MESSAGE_XXX, USER_PROMPT_XXX)
        """
        filename_map = {
            "event_annotation": self.event_annotation_prompt_file,
            "voice_diarization": self.voice_diarization_prompt_file,
            "event_relation": self.event_relation_prompt_file
        }

        filename = filename_map[prompt_name]

        prompt_path = os.path.join(self.prompts_dir, filename)

        # Pick loader based on extension
        if filename.endswith('.py'):
            # Import the prompt variables from a .py file
            import importlib.util
            spec = importlib.util.spec_from_file_location("prompt_module", prompt_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Pick the per-client prompt variable
            if client is None:
                # Pick a default client based on prompt_name
                if prompt_name == "event_annotation":
                    client = self.caption_client
                elif prompt_name == "voice_diarization":
                    client = self.asr_client
                elif prompt_name == "event_relation":
                    client = self.relation_client  # event_relation uses its own configured client
                else:
                    client = "gpt4o"  # default

            if client.lower() == "gemini":
                system_message = getattr(module, 'SYSTEM_MESSAGE_GEMINI', None)
                user_prompt = getattr(module, 'USER_PROMPT_GEMINI', '')
            else:  # gpt4o or anything else
                system_message = getattr(module, 'SYSTEM_MESSAGE_GPT4O', getattr(module, 'SYSTEM_MESSAGE', None))
                user_prompt = getattr(module, 'USER_PROMPT_GPT4O', getattr(module, 'USER_PROMPT', ''))

            return (system_message, user_prompt)
        else:
            # Read from a .txt file (legacy format)
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return (None, content)

    def get_voiceprint_model_params(self) -> Dict[str, Any]:
        """
        Return the architecture parameters for the active voiceprint checkpoint.

        Returns:
            Dict with baseWidth, scale, expansion. Falls back to a default
            (baseWidth=26, scale=2, expansion=2) if the checkpoint isn't listed.
        """
        model_path = self.voiceprint_model_path

        # Look up by model_path in the model_configs dict
        if model_path in self.voiceprint_model_configs:
            return self.voiceprint_model_configs[model_path]

        # Not found: fall back to defaults and warn
        default_params = {
            "baseWidth": 26,
            "scale": 2,
            "expansion": 2
        }
        print(f"⚠️  Warning: model path {model_path} has no architecture parameters in the config.")
        print(f"    Falling back to defaults: {default_params}")
        return default_params

    def get_embeddings_path(self, event_id: str) -> str:
        """Return the embedding feature file path."""
        return os.path.join(self.features_dir, "embeddings", f"{event_id}_emb.npy")

    def get_vision_feature_path(self, event_id: str) -> str:
        """Return the vision feature file path."""
        return os.path.join(self.features_dir, "vision", f"{event_id}_vision.npy")

    def get_audio_feature_path(self, event_id: str) -> str:
        """Return the audio feature file path."""
        return os.path.join(self.features_dir, "audio", f"{event_id}_audio.npy")

    def get_event_json_path(self, event_id: str, day: int) -> str:
        """Return the event JSON file path."""
        return os.path.join(self.events_dir, f"DAY{day}", f"{event_id}.json")

    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        os.makedirs(self.events_dir, exist_ok=True)
        os.makedirs(self.entities_dir, exist_ok=True)
        os.makedirs(os.path.join(self.features_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(self.features_dir, "vision"), exist_ok=True)
        os.makedirs(os.path.join(self.features_dir, "audio"), exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        # Create voice-related directories if configured
        if self.voiceprint_embedding_dir:
            os.makedirs(self.voiceprint_embedding_dir, exist_ok=True)
        if self.voiceprint_asr_dir:
            os.makedirs(self.voiceprint_asr_dir, exist_ok=True)
        if self.voice_database_path:
            os.makedirs(os.path.dirname(self.voice_database_path), exist_ok=True)
        if self.voiceprint_debug_save_segments and self.voiceprint_debug_save_dir:
            os.makedirs(self.voiceprint_debug_save_dir, exist_ok=True)
