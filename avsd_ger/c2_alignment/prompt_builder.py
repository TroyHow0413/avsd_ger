"""Prompt rendering and generated-transcript normalization for GER."""
from __future__ import annotations

import re
from typing import Any


class GERPromptBuilder:
    AV_PLACEHOLDER = "<AV_CTX>"

    def __init__(self, cfg: dict[str, Any], tokenizer: Any):
        self.template = str(cfg["prompt_template"])
        self.speaker_token = str(
            cfg.get("speaker_special_token", "[Speaker: ID_i]")
        )
        self.tokenizer = tokenizer

    def render(
        self,
        speaker_id: str | None,
        nbest: list[str],
        lip_hyp: str,
        *,
        mode: str = "av",
        use_av_context: bool = True,
    ) -> str:
        del speaker_id  # identity is carried by the learned special-token bias
        asr_block = " | ".join(item.strip() for item in nbest if item.strip())
        if mode == "audio_only":
            content = (
                f"{self.speaker_token}\nAudio hypothesis: {asr_block or '<none>'}\n"
                "Correct the transcript using the audio hypothesis as the main source.\n"
                "Return only the corrected transcript text, with no explanation and no quoted instruction.\n"
                'Do not output the words "speaker label".\nOutput:\n'
            )
        elif mode == "visual_only":
            content = (
                f"{self.speaker_token}\nVisual hypothesis: {lip_hyp or '<none>'}\n"
                "Correct the transcript using the visual hypothesis as the main source.\n"
                "Return only the corrected transcript text, with no explanation and no quoted instruction.\n"
                'Do not output the words "speaker label".\nOutput:\n'
            )
        elif mode == "av":
            content = self.template.format(
                speaker_tag=self.speaker_token,
                asr_nbest=asr_block or "<none>",
                lip_hyp=lip_hyp or "<none>",
            )
            if use_av_context:
                if self.AV_PLACEHOLDER not in content:
                    raise ValueError("AV prompt_template must contain <AV_CTX>")
            else:
                content = content.replace(
                    f"Aligned feature context: {self.AV_PLACEHOLDER}\n", ""
                ).replace(self.AV_PLACEHOLDER, "")
        else:
            raise ValueError(f"Unsupported GER prompt mode: {mode!r}")

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return content

    @staticmethod
    def clean_generated_text(text: str) -> str:
        value = text.strip()
        if not value:
            return value
        for pattern in (
            r"(?is)^the corrected transcript text is\s*:?\s*",
            r"(?is)^corrected transcript\s*:?\s*",
            r"(?is)^transcript\s*:?\s*",
            r"(?is)^output\s*:?\s*",
        ):
            value = re.sub(pattern, "", value).strip()
        quoted = re.findall(r'"([^"\n]{1,300})"', value)
        if quoted:
            value = quoted[0].strip()
        value = re.split(
            r"(?i)\b(the audio hypothesis|the visual hypothesis|the speaker|is saying|repeatedly)\b",
            value,
            maxsplit=1,
        )[0].strip()
        if "|" in value:
            parts = [part.strip(" \t\r\n\"'") for part in value.split("|") if part.strip()]
            if parts:
                value = parts[0]
        chunks = [chunk.strip() for chunk in re.findall(r"[^.!?]+[.!?]?", value) if chunk.strip()]
        deduped: list[str] = []
        previous = None
        for chunk in chunks:
            key = re.sub(r"\W+", "", chunk).lower()
            if key and key == previous:
                continue
            deduped.append(chunk)
            previous = key
        if deduped:
            keys = [re.sub(r"\W+", "", chunk).lower() for chunk in deduped]
            for unit in range(1, len(keys) // 2 + 1):
                if len(keys) % unit == 0 and all(
                    keys[index:index + unit] == keys[:unit]
                    for index in range(0, len(keys), unit)
                ):
                    deduped = deduped[:unit]
                    break
            value = " ".join(deduped)
        return re.sub(r"\s+", " ", value).strip(" \t\r\n\"'")
