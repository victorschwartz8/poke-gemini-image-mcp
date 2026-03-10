import base64
import os

import httpx
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

mcp = FastMCP(
    name="gemini-image",
    instructions=(
        "Use generate_image to create images from text prompts via Google Gemini. "
        "Defaults to gemini-3.1-flash-image-preview. "
        "Supports different aspect ratios (1:1, 3:4, 4:3, 9:16, 16:9)."
    ),
)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


# -- Shared helpers ----------------------------------------------------------


async def gemini_generate_image(
    prompt: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
) -> tuple[bytes, str]:
    """POST to Gemini generateContent API and return (image_bytes, mime_type)."""
    url = f"{API_BASE}/{model}:generateContent"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")
        data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("No candidates in Gemini API response.")

    for part in candidates[0].get("content", {}).get("parts", []):
        if "inlineData" in part:
            inline = part["inlineData"]
            return base64.b64decode(inline["data"]), inline.get("mimeType", "image/png")

    raise RuntimeError("No image data in Gemini API response.")


# -- Tools -------------------------------------------------------------------


@mcp.tool
async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    model: str = DEFAULT_MODEL,
) -> Image:
    """Generate an image from a text prompt using Google Gemini.

    Args:
        prompt: Text description of the image to generate.
        aspect_ratio: Aspect ratio — 1:1, 3:4, 4:3, 9:16, or 16:9.
        model: Model to use. Defaults to gemini-3.1-flash-image-preview.
               Also supports gemini-3-pro-image-preview or gemini-2.5-flash-image.
    """
    image_bytes, mime_type = await gemini_generate_image(prompt, model, aspect_ratio)
    fmt = "png" if "png" in mime_type else "jpeg"
    return Image(data=image_bytes, format=fmt)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
