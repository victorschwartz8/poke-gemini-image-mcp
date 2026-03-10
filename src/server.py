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
        "Use generate_image to create images from text prompts via Google Imagen 4. "
        "Supports different aspect ratios (1:1, 3:4, 4:3, 9:16, 16:9) and quality tiers "
        "(standard, fast, ultra)."
    ),
)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# -- Shared helpers ----------------------------------------------------------


async def imagen_generate(
    prompt: str,
    model: str = "imagen-4.0-generate-001",
    aspect_ratio: str = "1:1",
    person_generation: str = "allow_adult",
) -> bytes:
    """POST to Imagen API and return decoded PNG bytes."""
    url = f"{API_BASE}/{model}:predict"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json={
                "instances": [{"prompt": prompt}],
                "parameters": {
                    "sampleCount": 1,
                    "aspectRatio": aspect_ratio,
                    "personGeneration": person_generation,
                },
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Imagen API error ({resp.status_code}): {resp.text}")
        data = resp.json()

    predictions = data.get("predictions")
    if not predictions or "bytesBase64Encoded" not in predictions[0]:
        raise RuntimeError("No image data in Imagen API response.")

    return base64.b64decode(predictions[0]["bytesBase64Encoded"])


# -- Tools -------------------------------------------------------------------


@mcp.tool
async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    model: str = "imagen-4.0-generate-001",
    person_generation: str = "allow_adult",
) -> Image:
    """Generate an image from a text prompt using Google Imagen 4.

    Args:
        prompt: Text description of the image to generate.
        aspect_ratio: Aspect ratio — 1:1, 3:4, 4:3, 9:16, or 16:9.
        model: Model to use — imagen-4.0-generate-001 (standard),
               imagen-4.0-fast-generate-001 (faster), or
               imagen-4.0-ultra-generate-001 (highest quality).
        person_generation: Person generation policy — dont_allow, allow_adult, or allow_all.
    """
    image_bytes = await imagen_generate(prompt, model, aspect_ratio, person_generation)
    return Image(data=image_bytes, format="png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
