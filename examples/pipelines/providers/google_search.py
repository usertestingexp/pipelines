"""
title: Google GenAI Search Tool Pipeline
author: Your Name
date: 2024-12-27
version: 1.1
license: MIT
description: A pipeline for generating text using Google's GenAI models, utilizing the Search Tool function.
requirements: google-generativeai
environment_variables: GOOGLE_API_KEY
"""

from typing import List, Union, Iterator
import os
import json

from pydantic import BaseModel, Field

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, Tool, FunctionDeclaration


class Pipeline:
    """Google GenAI pipeline with Search Tool integration"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_API_KEY: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.id = "google_genai_search_tool"
        self.name = "Google Search Tool: "

        self.valves = self.Valves(**{
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "USE_PERMISSIVE_SAFETY": False
        })
        self.pipelines = []
        self.google_search_tool = Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="google_search",
                        description="Performs a google search and retrieves the results.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query"}
                            },
                            "required": ["query"]
                        }
                    )
                ]
            )
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            self.update_pipelines()
            print("Pipeline initialized successfully.")
        except Exception as e:
            print(f"Error during pipeline initialization: {e}")


    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            self.update_pipelines()
        except Exception as e:
            print(f"Error during on_startup: {e}")

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""

        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""

        print(f"on_valves_updated:{__name__}")
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            self.update_pipelines()
        except Exception as e:
            print(f"Error during on_valves_updated: {e}")


    def update_pipelines(self) -> None:
        """Update the available models from Google GenAI"""

        if self.valves.GOOGLE_API_KEY:
            try:
                models = genai.list_models()
                self.pipelines = [
                    {
                        "id": model.name[7:],
                        "name": model.display_name,
                    }
                    for model in models
                    if "generateContent" in model.supported_generation_methods
                    if model.name[:7] == "models/"
                ]
            except Exception as e:
                print(f"Error updating pipelines: {e}")
                self.pipelines = [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Google, please update the API Key in the valves.",
                    }
                ]
        else:
            self.pipelines = []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"

        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)

            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")

            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)

            contents = []
            for message in messages:
                if message["role"] != "system":
                    if isinstance(message.get("content"), list):
                        parts = []
                        for content in message["content"]:
                            if content["type"] == "text":
                                parts.append({"text": content["text"]})
                            elif content["type"] == "image_url":
                                image_url = content["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    image_data = image_url.split(",")[1]
                                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})
                                else:
                                    parts.append({"image_url": image_url})
                        contents.append({"role": message["role"], "parts": parts})
                    elif message.get("function_call"):
                        contents.append({
                            "role": message["role"],
                            "parts": [{
                                "function_call": message["function_call"]
                            }]
                        })
                    else:
                        contents.append({
                            "role": "user" if message["role"] == "user" else "model",
                            "parts": [{"text": message["content"]}]
                        })

            if "gemini-1.5" in model_id:
                model = genai.GenerativeModel(model_name=model_id, system_instruction=system_message, tools=[self.google_search_tool])
            else:
                if system_message:
                   contents.insert(0, {"role": "user", "parts": [{"text": f"System: {system_message}"}]})
                model = genai.GenerativeModel(model_name=model_id, tools=[self.google_search_tool])

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")
            
            response = model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=body.get("stream", False),
            )

            if response.candidates and response.candidates[0].content.parts and response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                function_name = function_call.name
                function_parameters = json.loads(function_call.args)

                if function_name == "google_search":
                    query = function_parameters["query"]
                    
                    search_contents = contents + [{
                                "role": "user",
                                "parts": [{
                                    "text": query
                                }]
                            }]
                    
                    search_response = model.generate_content(
                        search_contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        stream=body.get("stream", False),
                    )

                    search_results = ""
                    if search_response.candidates and search_response.candidates[0].content.parts:
                        for part in search_response.candidates[0].content.parts:
                            search_results += part.text

                    contents.append({
                        "role": "function",
                        "parts": [{
                            "text": search_results
                        }]
                    })

                    response = model.generate_content(
                        contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        stream=body.get("stream", False),
                    )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}")

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                yield chunk.text
