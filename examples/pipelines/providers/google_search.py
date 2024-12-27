"""
title: Google GenAI Manifold Pipeline
author: Marc Lopez (refactor by justinh-rahb)
date: 2024-06-06
version: 1.3
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: google-generativeai
"""

from typing import List, Union, Iterator
import os

from pydantic import BaseModel, Field

from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

client = genai.Client()
model_id = "gemini-2.0-flash-exp"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

response = client.models.generate_content(
    model=model_id,
    contents="When is the next total solar eclipse in the United States?",
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)
# Example response:
# The next total solar eclipse visible in the contiguous United States will be on ...

# To get grounding metadata as web content.
print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)
