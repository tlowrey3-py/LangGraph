import os
import hashlib
import json
import time
from openai import AzureOpenAI
from dotenv import dotenv_values, load_dotenv
from datetime import datetime
from collections import deque
import tiktoken
import itertools
from typing import Dict, List

from typing import Optional
from database.db_connection import SQLServerConnection
from database.tracker import create_session, insert_tracker, update_tracker, Tracker
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify, session

import logging
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType, VectorizableTextQuery
# import cohere
import requests

logger = logging.getLogger("flask-backend")

class IndexService:
    def __init__(self, config):
        is_local_debug = config["IS_LOCAL_DEBUG"].lower() == "true"
        self.credential = AzureKeyCredential(config["AZURE_SEARCH_ADMIN_KEY"]) if is_local_debug else DefaultAzureCredential()
        self.endpoint = config["AZURE_SEARCH_SERVICE_ENDPOINT"]
        # self.cohere_api_key = config["AZURE_COHERE_RERANKER_API_KEY"]
        # self.cohere_uri = config["AZURE_COHERE_RERANKER_URI"]
        self.index_name = config["AZURE_SEARCH_INDEX"]
    
    # TODO needs a graceful fallback for when the Azure Search API fails
    def azure_ai_search(self, query: str, max_context: int = 10, final_top_n: int = 5):
        try:
            # Initialize the Azure Search client
            search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
            vector_query = VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=max_context,
                fields="contentVector",
                exhaustive=True
            )
            # Perform the search query
            results = search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "content_type", "title", "content", "abbreviated_title", "implementation_date"],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name='my-semantic-config',
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=max_context
            )

            # Process the results into a list of dictionaries
            context_json_list = []
            for result in results:
                context_json = {
                    "content": result.get("content"),
                    "title": result.get("title"),
                    "abbreviated_title": result.get("abbreviated_title"),
                    "implementation_date": result.get("implementation_date")
                }
                context_json_list.append(context_json)

            # Handle cases where fewer results are returned
            if len(context_json_list) < final_top_n:
                logger.warning("Fewer results than expected in azure_ai_search", extra={
                    "function": "azure_ai_search",
                    "query": query[:100],
                    "results_returned": len(context_json_list),
                    "expected_minimum": final_top_n,
                    "max_context": max_context
                })
                return context_json_list

            # Return the top N results
            final_results = context_json_list[:final_top_n]

            logger.info("Successful search in azure_ai_search", extra={
                "function": "azure_ai_search",
                "query": query[:100],
                "results_returned": len(final_results),
                "max_context": max_context,
                "final_top_n": final_top_n
            })

            return final_results

        except Exception as ex:
            logging.exception(f"Exception occurred: {str(ex)}")
            # Log the exception with additional context
            logger.exception(f"Azure AI Search failed: {ex}", extra={
                "function": "azure_ai_search",
                "query": query[:100],  # Don't log huge queries
                "max_context": max_context,
                "final_top_n": final_top_n
            })
            # Fallback: Return an empty list instead of crashing
            return []

    # Index Service may move to functions later
    def process_context(self, context_raw):
        """
        Process context from request.form.get("context").
        Accepts either a single JSON object or a list of JSON objects.
        Returns a list of dictionaries formatted consistently.

        :param context_raw: str, raw JSON string from the "context" field
        :return: list of dictionaries
        """
        try:
            # Parse the raw context input
            context_parsed = json.loads(context_raw)

            # Ensure it's a list of dictionaries
            if isinstance(context_parsed, dict):
                context_parsed = [context_parsed]  # Wrap single object in a list

            if not isinstance(context_parsed, list) or not all(isinstance(obj, dict) for obj in context_parsed):
                raise ValueError("Context must be a JSON object or a list of JSON objects.")

            # Format each object in the desired structure
            formatted_context = []
            for item in context_parsed:
                formatted_context.append({
                    "id": item.get("id"),
                    "content_type": item.get("content_type"),
                    "content": item.get("content"),
                    "title": item.get("title"),
                    "abbreviated_title": item.get("abbreviated_title"),
                    "implementation_date": item.get("implementation_date")  # Optional, might not exist in input
                })

            return formatted_context

        except json.JSONDecodeError:
            logger.exception("Invalid JSON format in process_context", extra={
                "function": "process_context",
                "context_preview": context_raw[:300]  # Don't dump the entire payload in case it's huge
            })
            raise ValueError("Invalid JSON format provided for context.")
    
        except Exception as e:
            logger.exception(f"Unexpected error in process_context: {str(e)}", extra={
                "function": "process_context",
                "context_preview": context_raw[:300]
            })
            raise ValueError(f"Error processing context")
