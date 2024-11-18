import os
import tokenizers
import ctranslate2
import numpy as np
import torch

from .trt_model import WhisperTRT
from .tokenizer import Tokenizer
from .hf_utils import download_model
from .engine_builder import build_trt_engine, TRTBuilderConfig, load_trt_build_config


from .. import WhisperModel
from ...configs import *


SPECIAL_TOKENS_MAP = {
    50257: "<|unknown|>",
    50258: "<|unknown|>",
    50259: "<|en|>",
    50260: "<|zh|>",
    50261: "<|de|>",
    50262: "<|es|>",
    50263: "<|ru|>",
    50264: "<|ko|>",
    50265: "<|fr|>",
    50266: "<|ja|>",
    50267: "<|pt|>",
    50268: "<|tr|>",
    50269: "<|pl|>",
    50270: "<|ca|>",
    50271: "<|nl|>",
    50272: "<|ar|>",
    50273: "<|sv|>",
    50274: "<|it|>",
    50275: "<|id|>",
    50276: "<|hi|>",
    50277: "<|fi|>",
    50278: "<|vi|>",
    50279: "<|he|>",
    50280: "<|uk|>",
    50281: "<|el|>",
    50282: "<|ms|>",
    50283: "<|cs|>",
    50284: "<|ro|>",
    50285: "<|da|>",
    50286: "<|hu|>",
    50287: "<|ta|>",
    50288: "<|no|>",
    50289: "<|th|>",
    50290: "<|ur|>",
    50291: "<|hr|>",
    50292: "<|bg|>",
    50293: "<|lt|>",
    50294: "<|la|>",
    50295: "<|mi|>",
    50296: "<|ml|>",
    50297: "<|cy|>",
    50298: "<|sk|>",
    50299: "<|te|>",
    50300: "<|fa|>",
    50301: "<|lv|>",
    50302: "<|bn|>",
    50303: "<|sr|>",
    50304: "<|az|>",
    50305: "<|sl|>",
    50306: "<|kn|>",
    50307: "<|et|>",
    50308: "<|mk|>",
    50309: "<|br|>",
    50310: "<|eu|>",
    50311: "<|is|>",
    50312: "<|hy|>",
    50313: "<|ne|>",
    50314: "<|mn|>",
    50315: "<|bs|>",
    50316: "<|kk|>",
    50317: "<|sq|>",
    50318: "<|sw|>",
    50319: "<|gl|>",
    50320: "<|mr|>",
    50321: "<|pa|>",
    50322: "<|si|>",
    50323: "<|km|>",
    50324: "<|sn|>",
    50325: "<|yo|>",
    50326: "<|so|>",
    50327: "<|af|>",
    50328: "<|oc|>",
    50329: "<|ka|>",
    50330: "<|be|>",
    50331: "<|tg|>",
    50332: "<|sd|>",
    50333: "<|gu|>",
    50334: "<|am|>",
    50335: "<|yi|>",
    50336: "<|lo|>",
    50337: "<|uz|>",
    50338: "<|fo|>",
    50339: "<|ht|>",
    50340: "<|ps|>",
    50341: "<|tk|>",
    50342: "<|nn|>",
    50343: "<|mt|>",
    50344: "<|sa|>",
    50345: "<|lb|>",
    50346: "<|my|>",
    50347: "<|bo|>",
    50348: "<|tl|>",
    50349: "<|mg|>",
    50350: "<|as|>",
    50351: "<|tt|>",
    50352: "<|haw|>",
    50353: "<|ln|>",
    50354: "<|ha|>",
    50355: "<|ba|>",
    50356: "<|jw|>",
    50357: "<|su|>",
    50358: "<|yue|>",
    50359: "<|unknown|>",
    50360: "<|unknown|>",
    50361: "<|unknown|>",
    50362: "<|unknown|>",
    50363: "<|unknown|>",
    50364: "<|unknown|>",
}


FAST_ASR_OPTIONS = {
    "beam_size": 1,
    "best_of": 1,  # Placeholder
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4,  # Placeholder
    "log_prob_threshold": -1.0,  # Placeholder
    "no_speech_threshold": 0.5,  # Placeholder
    "prefix": None,  # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False,  # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": "tiny",
}


BEST_ASR_CONFIG = {
    "beam_size": 2,
    "best_of": 1,  # Placeholder
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4,  # Placeholder
    "log_prob_threshold": -1.0,  # Placeholder
    "no_speech_threshold": 0.5,  # Placeholder
    "prefix": None,  # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False,  # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": "tiny",
}


class WhisperModelTRT(WhisperModel):
    def __init__(
        self,
        model_name_or_path: str,
        cpu_threads=4,
        num_workers=1,
        device="cuda",
        device_index=0,
        compute_type="bfloat16",
        max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
        asr_options={},
        **model_kwargs,
    ):

        # ASR Options
        self.asr_options = BEST_ASR_CONFIG
        self.asr_options.update(asr_options)

        # Get local model path or build a new engine
        if os.path.isdir(model_name_or_path):
            self.model_path = model_name_or_path
            trt_build_args = load_trt_build_config(self.model_path)
        else:
            trt_build_args = model_kwargs.get("trt_build_args", None)
            if trt_build_args is None:
                print(
                    f"'trt_build_args' not provided in model_kwargs, using default configs."
                )
                trt_build_args = TRTBuilderConfig(
                    max_output_len=max_text_token_len,
                    max_beam_width=self.asr_options["beam_size"],
                )

            self.model_path = build_trt_engine(
                model_name=model_name_or_path, args=trt_build_args
            )

        if "trt_build_args" in model_kwargs:
            del model_kwargs["trt_build_args"]

        self.trt_build_args = trt_build_args

        # Update params according to TRT Build Args
        if max_text_token_len > self.trt_build_args.max_output_len:
            print(
                f"'max_text_token_len' cannot be larger than 'self.trt_build_args.max_output_len'. Setting 'max_text_token_len' to {self.trt_build_args.max_output_len}."
            )
            max_text_token_len = self.trt_build_args.max_output_len

        if self.asr_options["beam_size"] > self.trt_build_args.max_beam_width:
            print(
                f"'beam_size' cannot be larger than 'self.trt_build_args.max_beam_width'. Setting 'beam_size' to {self.trt_build_args.max_beam_width}."
            )
            self.asr_options["beam_size"] = self.trt_build_args.max_beam_width

        # Load model
        self.model = WhisperTRT(self.model_path)

        # Load tokenizer
        tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
        tokenizer = Tokenizer(
            tokenizers.Tokenizer.from_file(tokenizer_file), self.model.is_multilingual
        )

        if self.asr_options["word_timestamps"]:
            self.aligner_model_path = download_model(
                self.asr_options["word_aligner_model"]
            )
            self.aligner_model = ctranslate2.models.Whisper(
                self.aligner_model_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                intra_threads=cpu_threads,
                inter_threads=num_workers,
            )

        self.generate_kwargs = {
            "end_id": tokenizer.eot,
            "pad_id": tokenizer.eot,
            "max_new_tokens": max_text_token_len,
            "length_penalty": self.asr_options["length_penalty"],
            "repetition_penalty": self.asr_options["repetition_penalty"],
            "num_beams": self.asr_options["beam_size"],
            "stop_words_list": self.asr_options["suppress_blank"],
            "bad_words_list": self.asr_options["suppress_tokens"],
            "temperature": self.asr_options["sampling_temperature"],
        }

        super().__init__(
            tokenizer=tokenizer,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs,
        )

    def update_generation_kwargs(self, params={}):
        self.generate_kwargs.update(params)

        if "max_text_token_len" in params:
            self.update_params(
                params={"max_text_token_len": params["max_text_token_len"]}
            )

    def encode(self, features):
        """
        [Not Used]
        """

        return self.model.encode(features)

    def assign_word_timings(self, alignments, text_token_probs, words, word_tokens):
        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])

        if len(word_tokens) <= 1:
            return []

        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return []

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * TIME_PRECISION
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probs = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]

        return [
            dict(
                word=word, start=round(start, 2), end=round(end, 2), prob=round(prob, 2)
            )
            for word, start, end, prob in zip(words, start_times, end_times, word_probs)
        ]

    def align_words(
        self, features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata
    ):
        lang_codes = [_["lang_code"] for _ in seg_metadata]
        word_tokens = self.tokenizer.split_to_word_tokens_batch(
            texts, text_tokens, lang_codes
        )

        start_seq_wise_req = {}
        for _idx, _sot_seq in enumerate(sot_seqs):
            try:
                # print(_sot_seq)
                start_seq_wise_req[_sot_seq].append(_idx)
            except:
                start_seq_wise_req[_sot_seq] = [_idx]

        token_alignments = [[] for _ in seg_metadata]
        for start_seq, req_idx in start_seq_wise_req.items():
            res = self.aligner_model.align(
                ctranslate2.StorageView.from_array(features[req_idx]),
                start_sequence=list(start_seq),
                text_tokens=[text_tokens[_] for _ in req_idx],
                num_frames=list(seq_lens[req_idx].detach().cpu().numpy()),
                median_filter_width=7,
            )

            for _res, _req_idx in zip(res, req_idx):
                token_alignments[_req_idx] = _res

        word_timings = []
        for _idx, _seg_metadata in enumerate(seg_metadata):
            _word_timings = self.assign_word_timings(
                token_alignments[_idx].alignments,
                token_alignments[_idx].text_token_probs,
                word_tokens[_idx][0],
                word_tokens[_idx][1],
            )

            stitched_seg = _seg_metadata["stitched_seg"]

            current_seg_idx = 0
            current_offset = _seg_metadata["start_time"]

            for w in _word_timings:
                while (w["start"] + current_offset) >= stitched_seg[current_seg_idx][1]:
                    current_seg_idx += 1
                    current_offset += (
                        stitched_seg[current_seg_idx][0]
                        - stitched_seg[current_seg_idx - 1][1]
                    )

                w["start"] += current_offset
                w["end"] += current_offset

            word_timings.append(_word_timings)

        return word_timings

    def generate_segment_batched(
        self, features, prompts, seq_lens, seg_metadata, lang_codes
    ):
        # TODO: when proper queue is implemented, this should be fixed
        if lang_codes[0] is not None:
            result = self.model.generate(
                features,
                prompts,
                **self.generate_kwargs,
            )
        else:
            # print("len here!", len(features))
            result = self.model.generate(
                features,
                prompts,
                decoder_input_ids=torch.tensor(
                    [[50258] for _ in range(features.shape[0])]
                ),
                **self.generate_kwargs,
            )
        # print(result)

        texts = self.tokenizer.decode_batch([x[0] for x in result])

        # 50257 - start of special tokens

        # logits = self.model.generate(
        #     features,
        #     prompts,
        #     decoder_input_ids=torch.tensor([[50258] for _ in range(features.shape[0])]),
        #     **self.generate_kwargs,
        # )
        # print(logits)

        response = []
        for idx, r in enumerate(result):
            response.append(
                {
                    "text": texts[idx].strip(),
                    "tokens": [x for x in result[idx][0] if x < 50257],
                    "temperature": self.asr_options["sampling_temperature"],
                    "language": (
                        SPECIAL_TOKENS_MAP[result[idx][0][1]]
                        .replace("<|", "")
                        .replace("|>", "")
                        if lang_codes[0] is None
                        else lang_codes[0]
                    ),
                    "seek": 0,
                    "avg_logprob": self.asr_options["log_prob_threshold"],
                    "compression_ratio": self.asr_options[
                        "compression_ratio_threshold"
                    ],
                    "no_speech_prob": self.asr_options["no_speech_threshold"],
                }
            )

        if self.asr_options["word_timestamps"]:
            text_tokens = [
                [_t for _t in x[0] if _t < self.tokenizer.eot] + [self.tokenizer.eot]
                for x in result
            ]
            sot_seqs = [tuple(_[-4:]) for _ in prompts]
            word_timings = self.align_words(
                features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata
            )

            for _response, _word_timings in zip(response, word_timings):
                _response["word_timestamps"] = _word_timings

        return response
