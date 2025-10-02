import re
import os
import csv
import logging
import json
from pprint import pprint
from tqdm import tqdm
import litellm
from litellm import completion
from dotenv import load_dotenv
from datetime import datetime
import argparse

from litellm import supports_response_schema

from pydantic import BaseModel
from typing import Literal

# litellm._turn_on_debug()


# API keys will be automatically loaded from .env into environment variables
# Make sure .env file contains:
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key
# GOOGLE_API_KEY=your-google-key
# MISTRAL_API_KEY=your-mistral-key
load_dotenv()


# Organize models by family with variants for each
# For each family, we start by running on a full size model
# If cost and time allows, we will also try lite models, and potentially a powerful reasoning model from each provider where possible
# Model names correpsond to the LiteLLM documentation, e.g. https://docs.litellm.ai/docs/providers/openai
model_families = {
    "OpenAI": [
        "gpt-5-nano-2025-08-07"
        # Large model
        # "gpt-4.1",
        # Lite model
        # "gpt-4.1-mini",
        # Most powerful reasoning model
        # "o3"
    ],
    "Anthropic": [
        # Large model
        "claude-3-5-sonnet",
        # Lite model
        # "claude-3-haiku",
        # Most powerful reasoning model
        # "anthropic/claude-3.7-sonnet"
    ],
    "Google": [
        # Large model
        "gemini/gemini-2.0-flash",
        # Lite model
        # "gemini/gemini-2.0-flash-lite"
        # Most powerful reasoning model
        # "gemini/gemini-2.5-pro"
    ],
    # "Mistral": [
    #     # Large model
    #     "mistral/mistral-medium-2505"
    #     # Lite model
    #     #"mistral/mistral-small-2503"
    #     # Most powerful reasoning model
    #     #"mistral/mistral-large-2411",
    # ],
    # "Qwen": [
    #     # Large model
    #     "ollama/qwen-72b",
    #     # Liter model
    #     "ollama/qwen-7b"
    #     # No Qwen reasoning models
    # ],
    # "DeepSeek": [
    #     # Most powerful reasoning model
    #     "ollama/deepseek-v3",
    # ],
    # "Meta": [
    #     # Larger model
    #     "ollama/llama-4-maverick",
    #     # Lite model
    #     "ollama/llama-4-scout",
    #     # Most powerful reasoning model
    #     # ???
    # ]
}


# Configure logging - with filter for HTTP logs
class HTTPFilter(logging.Filter):
    def filter(self, record):
        # Filter out logs containing "HTTP Request"
        return "HTTP Request" not in record.getMessage()


# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_log.log"), logging.StreamHandler()],
)
logger = logging.getLogger()

# Add HTTP filter to the root logger to suppress HTTP request logs
http_filter = HTTPFilter()
logger.addFilter(http_filter)

# Also suppress logs from the urllib3 and requests libraries
for log_name in ["urllib3", "requests", "httpx"]:
    logging.getLogger(log_name).setLevel(logging.WARNING)

MODEL = model_families["OpenAI"][0]
# MODEL = "gpt-4o-mini"  # Update this to your desired model
NUM_RETRIES = 20  # Number of times to retry a response if the output is malformed
NUM_RESPONSES = 1000  # Number of independent conversation threads to run
START_RESPONSE_OFFSET = 0  # Offset to start the conversation threads from


class MultiChoiceResponse(BaseModel):
    """Type class for structured model responses when they are supported"""

    discussion_of_options: str
    response: Literal["A", "B", "C"]


def run_experiment(model: str, num_responses: int):
    global MODEL, NUM_RESPONSES
    MODEL = model
    NUM_RESPONSES = num_responses

    if supports_response_schema(model=MODEL):
        print(f"Model {MODEL} supports response schema validation.")
        RESPONSE_FORMATTING = True
        litellm.enable_json_schema_validation = True

        print(
            f"Running experiment with {MODEL} in conversation mode with structured response prompting."
        )

    else:
        print(f"Model {MODEL} does NOT support response schema validation.")
        RESPONSE_FORMATTING = False

        print(
            f"Running experiment with {MODEL} in conversation mode with Chain-Of-Thought prompting."
        )

    # Load the contents of prompt files
    try:
        with open("prompt-prefix.txt", "r") as file:
            prompt_prefix = file.read()

        prompt_suffix = ""
        if RESPONSE_FORMATTING:
            with open("prompt-suffix-structured.txt", "r") as file:
                prompt_suffix = file.read()
        else:
            with open("prompt-suffix-cot.txt", "r") as file:
                prompt_suffix = file.read()
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {e}")
        raise

    # Prepare CSV output file
    csv_filename = f"{MODEL.replace('/', '-')}-responses-conversation-{"structured" if RESPONSE_FORMATTING else "cot"}.csv"
    columns = ["model", "conversation_thread", "scenario_id", "choice", "discussion"]

    # Check if file exists
    file_exists = os.path.isfile(csv_filename)

    # Create or open the CSV file
    try:
        if file_exists:
            print(f"Found existing file {csv_filename}.")

            # Count existing responses
            with open(csv_filename, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header

                conversation_thread_column_index = columns.index("conversation_thread")
                conversation_threads = set(
                    int(row[conversation_thread_column_index])
                    for row in reader
                    if row[conversation_thread_column_index].isdigit()
                )
                existing_responses = len(conversation_threads)

            # Update offset and remaining responses
            START_RESPONSE_OFFSET = existing_responses
            NUM_RESPONSES = max(0, NUM_RESPONSES - existing_responses)
            print(f"Found {existing_responses} existing responses.")
            print(
                f"Will generate {NUM_RESPONSES} additional responses, starting from offset {START_RESPONSE_OFFSET}."
            )

        # Open file for appending
        csv_file = open(csv_filename, "a", newline="")
        writer = csv.writer(csv_file)

        # Write header only if the file is newly created
        if not file_exists:
            print(f"Created new CSV file {csv_filename}.")

            writer.writerow(columns)
            csv_file.flush()

            print(
                f"Will generate {NUM_RESPONSES} additional responses, starting from offset {START_RESPONSE_OFFSET}."
            )

    except Exception as e:
        logger.error(f"Failed to open CSV file: {e}")
        raise

    try:
        # Process each conversation thread
        for conversation_thread in tqdm(
            range(START_RESPONSE_OFFSET, START_RESPONSE_OFFSET + NUM_RESPONSES)
        ):
            thread_answers = []  # Store answers for this thread
            thread_success = True  # Track if this thread completed successfully

            # Reset message history for this thread
            messages = []

            # Iterate over the scenarios
            for scenario_id in [1, 2, 3, 4, 5, 6, 7, 8]:
                try:
                    # Load this scenario
                    with open(f"scenario-0{scenario_id}.txt", "r") as file:
                        scenario_text = file.read()

                    # Prepare the prompt, using a different prompt for the first part of a conversation
                    if scenario_id == 1:
                        prompt = prompt_prefix + scenario_text + prompt_suffix
                    else:
                        prompt = scenario_text + prompt_suffix

                    # Add it to the conversation thread
                    message = {"role": "user", "content": prompt}
                    messages.append(message)

                    output_choice = None
                    output_reason = None
                    scenario_output = None

                    # Try to get a valid response
                    for retry in range(NUM_RETRIES):

                        # We're not asking for reasons in the CoT version
                        output_discussion = ""

                        try:
                            output_choice = None

                            # Get response using LiteLLM
                            if RESPONSE_FORMATTING:
                                response = completion(
                                    model=MODEL,
                                    messages=messages,
                                    response_format=MultiChoiceResponse,
                                )
                                scenario_output = response.choices[0].message.content
                                structured_output_dict = json.loads(scenario_output)
                                output_choice = structured_output_dict.get("response")
                                output_discussion = structured_output_dict.get(
                                    "discussion_of_options"
                                )
                            else:
                                response = completion(
                                    model=MODEL,
                                    messages=messages,
                                )
                                scenario_output = response.choices[0].message.content

                                # Parse output
                                output_choice = re.search(
                                    r"Choice:\s*(.*)", scenario_output
                                )

                                # Extract just the part after the colon
                                if output_choice:
                                    output_choice = output_choice.group(1).strip()
                                else:
                                    output_choice = None

                                output_discussion = scenario_output

                            # If we got valid output, break out of retry loop
                            if output_choice is not None:
                                break

                            logger.warning(
                                f"Thread {conversation_thread}, Scenario {scenario_id}: Retry {retry+1}/{NUM_RETRIES} - Failed to parse response"
                            )
                            logger.warning(f"Response: {scenario_output}")

                        except Exception as e:
                            logger.warning(
                                f"Thread {conversation_thread}, Scenario {scenario_id}: Retry {retry+1}/{NUM_RETRIES} - Exception: {e}"
                            )
                            # Continue with next retry

                    # After all retries, check if we got valid output
                    if output_choice is not None and output_discussion is not None:
                        # Add this message to the conversation history
                        messages.append(
                            {
                                "role": "assistant",
                                "content": scenario_output,
                            }
                        )

                        # Add parsed answer to this thread's answers
                        thread_answers.append(
                            [
                                MODEL,
                                conversation_thread,
                                scenario_id,
                                output_choice,
                                output_discussion,
                            ]
                        )
                    else:
                        # Failed to get valid output after all retries
                        logger.error(
                            f"Thread {conversation_thread}, Scenario {scenario_id}: Failed to parse response after {NUM_RETRIES} retries"
                        )
                        logger.error(f"Last response: {scenario_output}")
                        thread_success = False
                        # We'll add a placeholder for this failed scenario
                        thread_answers.append(
                            [
                                MODEL,
                                conversation_thread,
                                scenario_id,
                                "PARSING_FAILED",
                                "PARSING_FAILED",
                            ]
                        )
                        # Continue to next scenario rather than stopping

                except Exception as e:
                    logger.error(
                        f"Thread {conversation_thread}, Scenario {scenario_id}: Unexpected error: {e}"
                    )
                    thread_success = False
                    thread_answers.append(
                        [
                            MODEL,
                            conversation_thread,
                            scenario_id,
                            "ERROR",
                            str(e)[:100],
                        ]
                    )
                    # Continue to the next scenario

            # After all scenarios in this thread, save results to CSV
            try:
                writer.writerows(thread_answers)
                csv_file.flush()  # Ensure data is written to disk
                completed_msg = (
                    "completed successfully"
                    if thread_success
                    else "completed with errors"
                )
                logger.info(
                    f"Thread {conversation_thread} {completed_msg}. Saved {len(thread_answers)} results."
                )
            except Exception as e:
                logger.error(
                    f"Failed to write results for thread {conversation_thread}: {e}"
                )
                # Continue to next thread even if saving failed

        # All threads completed
        logger.info(f"Experiment completed. Results saved to {csv_filename}")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user. Partial results have been saved.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        # Close the CSV file
        if "csv_file" in locals():
            csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM experiment.")
    parser.add_argument(
        "model", type=str, help="The model string, e.g., 'gpt-5-nano-2025-08-07'."
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        help="The number of responses to generate.",
        default=1000,
        required=False,
    )

    args = parser.parse_args()

    run_experiment(args.model, args.num_responses)
