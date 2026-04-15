from __future__ import annotations

import asyncio
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

# ==============================================================================
#  Configuration
# ==============================================================================

load_dotenv()

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv(
    "OPENAI_BASE_URL", "https://api.deepseek.com/v1"
)
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "deepseek-chat")

# ==============================================================================
#  Tone-of-Voice Styles
# ==============================================================================

STYLE_VIRAL: str = "viral"
STYLE_PROFESSIONAL: str = "professional"
STYLE_PHILOSOPHICAL: str = "philosophical"

STYLE_LABELS: dict[str, str] = {
    STYLE_VIRAL: "🔥 Viral / Aggressive",
    STYLE_PROFESSIONAL: "💼 Professional / Expert",
    STYLE_PHILOSOPHICAL: "🧠 Deep / Philosophical",
}

STYLE_DESCRIPTIONS: dict[str, str] = {
    STYLE_VIRAL: (
        "Bold, provocative, polarizing. Short sentences. Power words. "
        "Designed to stop the scroll and trigger shares."
    ),
    STYLE_PROFESSIONAL: (
        "Credible, data-aware, authoritative. Structured arguments. "
        "Suitable for LinkedIn and executive audiences."
    ),
    STYLE_PHILOSOPHICAL: (
        "Contemplative, pattern-seeking, first-principles thinking. "
        "Connects the topic to deeper universal truths."
    ),
}

# ==============================================================================
#  Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  In-Memory User State
# ==============================================================================

_TELEGRAM_LIMIT = 4096
_CHUNK_MARGIN = 200
_CHUNK_LIMIT = _TELEGRAM_LIMIT - _CHUNK_MARGIN


@dataclass
class UserState:
    """Per-user session data."""
    style: str = STYLE_PROFESSIONAL
    last_request_at: float = 0.0


user_states: dict[int, UserState] = {}


def get_user_state(user_id: int) -> UserState:
    """Return (and lazily create) the state for *user_id*."""
    if user_id not in user_states:
        user_states[user_id] = UserState()
    return user_states[user_id]


# ==============================================================================
#  YouTube Video ID Extraction
# ==============================================================================

_YT_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/.*[?&]v=)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:m\.youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})"),
]


def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID from any common URL format.

    Supports: desktop, mobile (m.), shorts, live, embed, youtu.be, /v/.
    """
    url = url.strip()
    for pattern in _YT_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


# ==============================================================================
#  Rate Limiter
# ==============================================================================

COOLDOWN_SECONDS: int = 5


def is_rate_limited(user_id: int) -> bool:
    """Return True if *user_id* is still inside the cooldown window."""
    now = time.monotonic()
    state = get_user_state(user_id)
    if now - state.last_request_at < COOLDOWN_SECONDS:
        return True
    state.last_request_at = now
    return False


# ==============================================================================
#  Custom Exceptions
# ==============================================================================

class TranscriptError(Exception):
    """Raised when a transcript cannot be retrieved."""


class AIProcessingError(Exception):
    """Raised when the AI API fails or returns garbage."""


# ==============================================================================
#  Language Detection (from transcript metadata)
# ==============================================================================

_LANG_CODE_TO_NAME: dict[str, str] = {
    "en": "English",
    "en-US": "English",
    "en-GB": "English",
    "ru": "Russian",
    "ru-RU": "Russian",
    "uk": "Ukrainian",
    "uk-UA": "Ukrainian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-Hans": "Chinese",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
}


def detect_transcript_language(transcript_obj) -> str:
    """Infer the human-readable language name from a transcript object."""
    code = getattr(transcript_obj, "language_code", "") or ""
    return _LANG_CODE_TO_NAME.get(code, "English")


# ==============================================================================
#  Step 1: Transcript Extraction
# ==============================================================================

async def get_transcript(video_id: str) -> tuple[str, str]:
    """Fetch a YouTube video transcript.

    Returns:
        A tuple of (transcript_text, detected_language_name).

    Raises:
        TranscriptError: On any retrieval failure.
    """
    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except TranscriptsDisabled:
        raise TranscriptError("Subtitles are disabled for this video.")
    except VideoUnavailable:
        raise TranscriptError("The video is unavailable or has been removed.")
    except Exception as exc:
        raise TranscriptError(f"Failed to load transcripts: {exc}")

    # Prefer English, then Ukrainian, then Russian, then any
    preferred = ["en", "uk", "ru", "en-US", "ru-RU", "uk-UA", "en-GB"]
    transcript = None
    detected_language = "English"

    for lang in preferred:
        try:
            transcript = transcript_list.find_transcript([lang])
            detected_language = detect_transcript_language(transcript)
            break
        except NoTranscriptFound:
            continue

    if transcript is None:
        # Fallback: auto-generated -> manually created
        try:
            transcript = transcript_list.find_generated_transcript([])
            detected_language = detect_transcript_language(transcript)
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_manually_created_transcript([])
                detected_language = detect_transcript_language(transcript)
            except NoTranscriptFound:
                raise TranscriptError("No transcript available in any language.")

    entries = transcript.fetch()
    text = " ".join(entry.text for entry in entries)
    return text, detected_language


# ==============================================================================
#  Step 2: AI Content Generation - The Golden Prompt
# ==============================================================================

def _build_system_prompt(style: str, language: str) -> str:

    tone_instructions = {
        STYLE_VIRAL: textwrap.dedent("""\
            TONE: Aggressive, provocative, polarizing.
              - Use short, punchy sentences.
              - Lead with controversy or a bold claim.
              - Power words, no hedging.
              - Designed to stop the scroll and trigger shares."""),
        STYLE_PROFESSIONAL: textwrap.dedent("""\
            TONE: Professional, authoritative, credible.
              - Structured arguments with clear logic.
              - Reference data or frameworks where possible.
              - Suitable for LinkedIn and executive audiences."""),
        STYLE_PHILOSOPHICAL: textwrap.dedent("""\
            TONE: Deep, contemplative, first-principles thinking.
              - Connect the topic to universal human truths.
              - Use pattern recognition and mental models.
              - Thought-provoking, not preachy."""),
    }

    tone_block = tone_instructions.get(style, tone_instructions[STYLE_PROFESSIONAL])

    return textwrap.dedent(f"""\
        You are a Senior Content Strategist & Growth Hacker. Your specialty is
        cross-platform content distillation - turning long-form video transcripts
        into viral, platform-optimized distribution packs.

        {tone_block}

        LANGUAGE: All body content MUST be written in {language}.
        Keep section HEADERS in English for a professional look.

        TASK: Deconstruct the raw transcript below and generate these 8 sections.
        Separate every section with a line of three dashes: ---

        SECTION 1 - ATOMIC INSIGHTS
          Identify 3-5 Atomic Insights. Format each as a blockquote:
          > *Insight title*: one-sentence explanation

        SECTION 2 - LINKEDIN POST
          Expert-level thought leadership post.
          Hook -> Insights -> Actionable takeaway -> Call-to-action.
          Use line breaks for readability.

        SECTION 3 - TELEGRAM POST
          Punchy, direct, utility-focused.
          Use emojis strategically (not spammy).
          Bold key terms. Keep it under 800 characters.

        SECTION 4 - X / TWITTER THREAD
          Exactly 7 tweets.
          [1/7]: Irresistible hook.
          [2/6]: Each delivers one atomic insight, self-contained.
          [7/7]: Summary + call-to-action.

        SECTION 5 - TIKTOK / REELS SCRIPT
          High-energy, spoken style, 30-60 second runtime.
          Include [Visual Cues] in brackets for every scene transition.

        SECTION 6 - VIRAL HOOKS (5x)
          Generate 5 MrBeast-style headline hooks for this content.
          Each should be curiosity-driven, under 60 characters.
          Format:
          1. Hook text here
          2. Hook text here
          ...

        SECTION 7 - AI IMAGE PROMPTS (3x)
          Generate 3 precise image prompts for Midjourney or DALL-E 3.
          Style: Photorealistic or Modern Abstract.
          Each prompt should visually represent one core insight.
          Format:
          [Image Prompt 1]: description
          [Image Prompt 2]: description
          [Image Prompt 3]: description

        SECTION 8 - DISTRIBUTION CHECKLIST
          A short 4-5 item checklist with the best times/channels to post.
          Format as bullet points with a checkmark emoji.

        GLOBAL RULES:
          - Active voice only. No cliches. No filler.
          - Maintain the original nuance and intent of the source.
          - Do NOT add disclaimers, meta-commentary, or "Here is your content."
          - Start directly with SECTION 1 header.
          - Separate every section with a --- divider line.""")


async def generate_ai_content(
    transcript: str,
    style: str,
    language: str,
) -> str:
    """Send *transcript* to the AI and return the generated content pack.

    Args:
        transcript: Raw video transcript text.
        style: One of the STYLE_* constants.
        language: Detected language name for the output.

    Raises:
        AIProcessingError: On API failure or empty response.
    """
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    system_prompt = _build_system_prompt(style, language)

    user_message = (
        f"Raw transcript ({language}):\n\n---\n{transcript}\n---\n\n"
        "Apply the system instructions and generate the full distribution pack."
    )

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
    except Exception as exc:
        err_lower = str(exc).lower()
        if any(t in err_lower for t in ("auth", "api key", "401")):
            raise AIProcessingError(
                "Invalid API key or unauthorized access.\n"
                "Check your OPENAI_API_KEY and OPENAI_BASE_URL."
            )
        if any(t in err_lower for t in ("context", "token", "429", "rate limit")):
            raise AIProcessingError(
                "Context window exceeded or rate limited.\n"
                "Try a shorter video or retry in a moment."
            )
        raise AIProcessingError(f"AI API Error: {exc}")

    content = response.choices[0].message.content
    if not content:
        raise AIProcessingError("AI returned an empty response. Please try again.")

    return content


# ==============================================================================
#  Step 3: Section-Aware Message Splitting
# ==============================================================================

def split_message(text: str) -> list[str]:
    """Split *text* at section boundaries (---), not mid-sentence.

    Telegram limit: 4096 chars. We use _CHUNK_LIMIT (3896) for safety.
    """
    if len(text) <= _CHUNK_LIMIT:
        return [text]

    # Split on section dividers
    sections = re.split(r"\n?---\n?", text)
    sections = [s.strip() for s in sections if s.strip()]

    if not sections:
        return [text]

    chunks: list[str] = []
    current = ""

    for section in sections:
        # If a single section exceeds the limit, hard-split it
        if len(section) > _CHUNK_LIMIT:
            if current:
                chunks.append(current.strip())
                current = ""
            sub_chunks = _hard_split(section)
            chunks.extend(sub_chunks)
            continue

        if len(current) + len(section) + 20 > _CHUNK_LIMIT:
            chunks.append(current.strip())
            current = section
        else:
            if current:
                current += "\n\n---\n\n" + section
            else:
                current = section

    if current:
        chunks.append(current.strip())

    return chunks


def _hard_split(text: str) -> list[str]:
    """Fallback: split by paragraphs when a single section is too large."""
    if len(text) <= _CHUNK_LIMIT:
        return [text]

    parts: list[str] = []
    while len(text) > _CHUNK_LIMIT:
        split_at = text.rfind("\n\n", 0, _CHUNK_LIMIT)
        if split_at == -1:
            split_at = text.rfind("\n", 0, _CHUNK_LIMIT)
        if split_at == -1:
            split_at = _CHUNK_LIMIT

        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        parts.append(text)

    return parts


async def send_split_message(message: Message, text: str) -> None:
    """Send *text* to the user, splitting into parts if necessary.

    Uses Telegram Markdown (parse_mode=Markdown) - the legacy format
    that supports *bold*, _italic_, and `code` WITHOUT the aggressive
    escaping that MarkdownV2 requires.
    """
    chunks = split_message(text)
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if total == 1:
            header = "✅ *Content Ready!*\n\n"
        else:
            header = f"✅ *Content Ready!* (part {idx}/{total})\n\n"

        full_text = header + chunk

        try:
            await message.answer(full_text, parse_mode=ParseMode.MARKDOWN)
        except TelegramBadRequest as exc:
            logger.warning(
                "Markdown send failed for chunk %d, retrying plain text: %s",
                idx, exc,
            )
            # Strip markdown for plain-text fallback
            clean = chunk.replace("*", "").replace("_", "")
            if total == 1:
                await message.answer(f"Content Ready!\n\n{clean}")
            else:
                await message.answer(
                    f"Content Ready! (part {idx}/{total})\n\n{clean}"
                )

        if total > 1 and idx < total:
            await asyncio.sleep(0.5)


# ==============================================================================
#  Inline Keyboard Builder
# ==============================================================================

def build_style_keyboard() -> InlineKeyboardMarkup:
    """Build the tone-of-voice selection inline keyboard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=STYLE_LABELS[STYLE_VIRAL],
                    callback_data=f"style:{STYLE_VIRAL}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=STYLE_LABELS[STYLE_PROFESSIONAL],
                    callback_data=f"style:{STYLE_PROFESSIONAL}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=STYLE_LABELS[STYLE_PHILOSOPHICAL],
                    callback_data=f"style:{STYLE_PHILOSOPHICAL}",
                ),
            ],
        ]
    )


# ==============================================================================
#  Bot Handlers
# ==============================================================================

dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Show a professional SaaS-style welcome message + style selector."""
    style_kb = build_style_keyboard()

    await message.answer(
        "🏭 *Content Refinery - AI Bot*\n"
        "---\n"
        "Turn any YouTube video into a complete *cross-platform content pack*.\n\n"
        "📝 LinkedIn post (thought leadership)\n"
        "✈️ Telegram post (punchy & direct)\n"
        "🐦 X/Twitter thread (7 viral tweets)\n"
        "🎬 TikTok/Reels script (with visual cues)\n"
        "🪝 5 Viral Hooks (MrBeast-style)\n"
        "🖼 3 AI Image prompts (Midjourney / DALL-E)\n"
        "✅ Distribution checklist\n\n"
        "---\n"
        "_Step 1: Choose your tone of voice below._",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=style_kb,
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Show usage instructions."""
    await message.answer(
        "📖 *How to use*\n"
        "---\n"
        "1. Choose a tone of voice (/start or use the buttons)\n"
        "2. Paste any YouTube video URL\n"
        "3. Wait ~15 seconds for your content pack\n\n"
        "*Supported URL formats:*\n"
        "`youtube.com/watch`\n"
        "`youtu.be`\n"
        "`youtube.com/shorts`\n"
        "`m.youtube.com/watch`",
        parse_mode=ParseMode.MARKDOWN,
    )


@dp.callback_query(F.data.startswith("style:"))
async def cb_select_style(callback: CallbackQuery) -> None:
    """Handle tone-of-voice selection from the inline keyboard."""
    if not callback.data or not callback.message:
        return

    style_key = callback.data.split(":", 1)[1]

    if style_key not in STYLE_LABELS:
        await callback.answer("Invalid style.", show_alert=True)
        return

    user_id = callback.from_user.id
    state = get_user_state(user_id)
    state.style = style_key

    label = STYLE_LABELS[style_key]
    description = STYLE_DESCRIPTIONS[style_key]

    confirmation_text = (
        f"✅ *Style selected: {label}*\n"
        f"---\n"
        f"{description}\n\n"
        f"_Now send me a YouTube video URL to get started._"
    )

    try:
        await callback.message.edit_text(
            confirmation_text,
            parse_mode=ParseMode.MARKDOWN,
        )
    except TelegramBadRequest:
        await callback.message.answer(confirmation_text, parse_mode=ParseMode.MARKDOWN)

    await callback.answer(f"Style: {label}")


@dp.message()
async def handle_video(message: Message) -> None:
    """Process a YouTube URL - the main 3-step pipeline."""
    if not message.text:
        return

    user_id = message.from_user.id
    state = get_user_state(user_id)

    # -- Rate limit --
    if is_rate_limited(user_id):
        await message.answer(
            f"⏸️ Please wait {COOLDOWN_SECONDS} seconds between requests."
        )
        return

    # -- Validate link --
    url = message.text.strip()
    video_id = extract_video_id(url)

    if not video_id:
        await message.answer(
            "❌ *Invalid Link*\n\n"
            "This doesn't look like a valid YouTube URL. Example:\n"
            "`https://www.youtube.com/watch?v=dQw4wRFFgW4`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    style_label = STYLE_LABELS[state.style]

    # -- [1/3] Extract transcript --
    status_msg = await message.answer(
        "📥 *[1/3]* Distilling transcript..."
    )

    try:
        transcript, language = await get_transcript(video_id)
        logger.info(
            "Transcript fetched: %s (%d chars, %s)",
            video_id, len(transcript), language,
        )
    except TranscriptError as exc:
        await status_msg.edit_text(f"❌ {exc}")
        return

    # -- [2/3] AI processing --
    await status_msg.edit_text(
        f"🧠 *[2/3]* AI is thinking in {style_label} mode..."
    )

    try:
        content = await generate_ai_content(
            transcript, state.style, language
        )
        logger.info("AI generation complete: %s", video_id)
    except AIProcessingError as exc:
        await status_msg.edit_text(f"❌ {exc}")
        return

    # -- [3/3] Deliver result --
    await status_msg.edit_text("🧠 *[3/3]* Done!")

    try:
        await send_split_message(message, content)
    except TelegramBadRequest as exc:
        logger.error("Failed to send result to user: %s", exc)
        await message.answer(
            "❌ An error occurred while sending the result. Please try again."
        )


# ==============================================================================
#  Entry Point
# ==============================================================================

async def main() -> None:
    """Validate configuration, initialise the bot, and start long-polling."""
    if not TELEGRAM_TOKEN:
        logger.error(
            "TELEGRAM_TOKEN is not set.\n"
            "Create a .env file with:\n"
            "  TELEGRAM_TOKEN=...\n"
            "  OPENAI_API_KEY=...\n"
            "  OPENAI_BASE_URL=https://api.deepseek.com/v1\n"
            "  OPENAI_MODEL=deepseek-chat"
        )
        return

    if not OPENAI_API_KEY:
        logger.error(
            "OPENAI_API_KEY is not set.\n"
            "Add it to your .env file."
        )
        return

    bot = Bot(token=TELEGRAM_TOKEN)
    logger.info(
        "Bot started - model=%s base_url=%s", OPENAI_MODEL, OPENAI_BASE_URL
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C).")
    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
