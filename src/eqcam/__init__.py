"""
eqcam: rapid survey of camera videos after an earthquake with vision-language models

  eqcam.frames    the motion of each video, and a tile of its frames before, during and after the shaking
  eqcam.schema    the findings (structural damage, landslides, ground failure, fire, shaking intensity, ...)
  eqcam.backends  Claude (Anthropic API or Vertex AI), Gemini (Vertex AI), Qwen3-VL (OpenAI-compatible server)
  eqcam.review    the findings of every camera, in parallel, with an optional second stage
  eqcam.report    the table and the maps
"""

__version__ = '0.1.0'
