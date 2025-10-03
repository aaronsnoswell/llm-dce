import re
import os
import csv
import json
import logging
import argparse
from datetime import datetime
from pprint import pprint
from enum import Enum

from tqdm import tqdm
from dotenv import load_dotenv
import litellm
from litellm import completion
from litellm import supports_response_schema

from pydantic import BaseModel
from typing import Literal, Tuple, Optional, Dict
import uuid

# API keys will be automatically loaded from .env into environment variables
load_dotenv()


class ResponseFormat(Enum):
    """Enum for response formatting types."""

    STRUCTURED = "structured"
    CHAIN_OF_THOUGHT = "cot"


# Organize models by family with variants for each
model_families = {
    "OpenAI": ["gpt-5-nano-2025-08-07"],
    "Anthropic": [
        "claude-4-5-sonnet",
    ],
    "Google": [
        "gemini/gemini-2.0-flash",
    ],
}

# Generate unique log filename
log_filename = f"experiment_{uuid.uuid4()}.log"
print(f"Logging to: {log_filename}")

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger()


class MultiChoiceResponse(BaseModel):
    """Type class for structured model responses when they are supported"""

    discussion_of_options: str
    response: Literal["A", "B", "C"]


def load_prompts(response_format: ResponseFormat) -> Tuple[str, str]:
    """Load prompt files based on response formatting type.

    Args:
        response_format: The response format type (STRUCTURED or CHAIN_OF_THOUGHT)

    Returns:
        Tuple of (prompt_prefix, prompt_suffix)
    """
    try:
        with open("prompt-prefix.txt", "r") as file:
            prompt_prefix = file.read()

        if response_format == ResponseFormat.STRUCTURED:
            with open("prompt-suffix-structured.txt", "r") as file:
                prompt_suffix = file.read()
        else:
            with open("prompt-suffix-cot.txt", "r") as file:
                prompt_suffix = file.read()

        return prompt_prefix, prompt_suffix
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {e}")
        raise


def setup_csv_file(
    model: str, response_format: ResponseFormat, num_responses: int
) -> Tuple[object, object, int, int]:
    """Set up CSV file for writing results, handling resume logic.

    Args:
        model: Model identifier string
        response_format: The response format type (STRUCTURED or CHAIN_OF_THOUGHT)
        num_responses: Total number of responses desired

    Returns:
        Tuple of (csv_file, writer, start_offset, remaining_responses)
    """
    csv_filename = f"{model.replace('/', '-').replace(':', '-')}-responses-conversation-{response_format.value}.csv"
    columns = ["model", "conversation_thread", "scenario_id", "choice", "discussion"]

    file_exists = os.path.isfile(csv_filename)
    start_offset = 0
    remaining_responses = num_responses

    try:
        if file_exists:
            print(f"Found existing file {csv_filename}.")

            # Count existing responses
            with open(csv_filename, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                row_count = sum(1 for row in reader)
                existing_responses = row_count // 8

            start_offset = existing_responses
            remaining_responses = max(0, num_responses - existing_responses)
            print(f"Found {existing_responses} existing responses.")
            print(
                f"Will generate {remaining_responses} additional responses, starting from offset {start_offset}."
            )

        # Open file for appending
        csv_file = open(csv_filename, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)

        if not file_exists:
            print(f"Created new CSV file {csv_filename}.")
            writer.writerow(columns)
            csv_file.flush()
            print(
                f"Will generate {remaining_responses} responses, starting from offset {start_offset}."
            )

        return csv_file, writer, start_offset, remaining_responses

    except Exception as e:
        logger.error(f"Failed to open CSV file: {e}")
        raise


def parse_response(
    scenario_output: str, response_format: ResponseFormat
) -> Tuple[Optional[str], str]:
    """Parse model response to extract choice and discussion.

    Args:
        scenario_output: Raw output from the model
        response_format: The response format type (STRUCTURED or CHAIN_OF_THOUGHT)

    Returns:
        Tuple of (choice, discussion)
    """
    if response_format == ResponseFormat.STRUCTURED:
        structured_output_dict = json.loads(scenario_output)
        output_choice = structured_output_dict.get("response")
        output_discussion = structured_output_dict.get("discussion_of_options")
    else:
        # Parse CoT output
        final_line = scenario_output.splitlines()[-1]
        if "Choice: A" in final_line:
            output_choice = "A"
        elif "Choice: B" in final_line:
            output_choice = "B"
        elif "Choice: C" in final_line:
            output_choice = "C"
        else:
            raise ValueError(f"Failed to parse choice from CoT response: {final_line}")
        # Output discussion is all but the final line
        output_discussion = "\n".join(scenario_output.splitlines()[:-1]).strip()

    return output_choice, output_discussion


def get_response_with_retry(
    model: str,
    messages: list,
    response_format: ResponseFormat,
    conversation_thread: int,
    scenario_id: int,
    num_retries: int,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get response from model with retry logic.

    Args:
        model: Model identifier
        messages: Conversation history
        response_format: The response format type (STRUCTURED or CHAIN_OF_THOUGHT)
        conversation_thread: Thread number for logging
        scenario_id: Scenario number for logging
        num_retries: Number of retry attempts

    Returns:
        Tuple of (raw_output, choice, discussion) or (None, None, None) on failure
    """
    for retry in range(num_retries):
        try:
            if response_format == ResponseFormat.STRUCTURED:
                response = completion(
                    model=model,
                    messages=messages,
                    response_format=MultiChoiceResponse,
                )
                scenario_output = response.choices[0].message.content
            else:
                response = completion(
                    model=model,
                    messages=messages,
                )
                scenario_output = response.choices[0].message.content

            # Parse the response
            output_choice, output_discussion = parse_response(
                scenario_output, response_format
            )

            # If we got valid output, return it
            if output_choice is not None:
                return scenario_output, output_choice, output_discussion

            logger.warning(
                f"Thread {conversation_thread}, Scenario {scenario_id}: "
                f"Retry {retry+1}/{num_retries} - Failed to parse response"
            )
            logger.warning(f"Response: {scenario_output}")

        except Exception as e:
            logger.warning(
                f"Thread {conversation_thread}, Scenario {scenario_id}: "
                f"Retry {retry+1}/{num_retries} - Exception: {e}"
            )

    # Failed after all retries
    return None, None, None


def process_scenario(
    model: str,
    messages: list,
    scenario_id: int,
    prompt_prefix: str,
    prompt_suffix: str,
    response_format: ResponseFormat,
    conversation_thread: int,
    num_retries: int,
) -> Tuple[list, list]:
    """Process a single scenario in the conversation.

    Args:
        model: Model identifier
        messages: Conversation history (will be modified)
        scenario_id: Scenario number (1-8)
        prompt_prefix: Prefix for first scenario only
        prompt_suffix: Suffix for all scenarios
        response_format: The response format type (STRUCTURED or CHAIN_OF_THOUGHT)
        conversation_thread: Thread number for logging
        num_retries: Number of retry attempts

    Returns:
        Tuple of (updated_messages, answer_row)
    """
    try:
        # Load this scenario
        with open(f"scenario-0{scenario_id}.txt", "r") as file:
            scenario_text = file.read()

        # Prepare the prompt (first scenario includes prefix)
        if scenario_id == 1:
            prompt = prompt_prefix + scenario_text + prompt_suffix
        else:
            prompt = scenario_text + prompt_suffix

        # Add to conversation thread
        messages.append({"role": "user", "content": prompt})

        # Get response with retry logic
        scenario_output, output_choice, output_discussion = get_response_with_retry(
            model,
            messages,
            response_format,
            conversation_thread,
            scenario_id,
            num_retries,
        )

        # Handle the result
        if output_choice is not None and output_discussion is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": scenario_output,
                }
            )

            answer_row = [
                model,
                conversation_thread,
                scenario_id,
                output_choice,
                output_discussion.replace("\n", " "),
            ]
        else:
            # Failed to get valid output after all retries
            logger.error(
                f"Thread {conversation_thread}, Scenario {scenario_id}: "
                f"Failed to parse response after {num_retries} retries"
            )
            logger.error(f"Last response: {scenario_output}")

            answer_row = [
                model,
                conversation_thread,
                scenario_id,
                "PARSING_FAILED",
                "PARSING_FAILED",
            ]

        return messages, answer_row

    except Exception as e:
        logger.error(
            f"Thread {conversation_thread}, Scenario {scenario_id}: "
            f"Unexpected error: {e}"
        )
        answer_row = [
            model,
            conversation_thread,
            scenario_id,
            "ERROR",
            str(e)[:100],
        ]
        return messages, answer_row


def run_experiment(model: str, num_responses: int, num_retries: int):
    """Run the main experiment loop.

    Args:
        model: Model identifier string
        num_responses: Number of conversation threads to run
        num_retries: Number of retry attempts for failed responses
    """
    # Check if model supports structured responses
    if supports_response_schema(model=model):
        print(f"Model {model} supports response schema validation.")
        response_format = ResponseFormat.STRUCTURED
        litellm.enable_json_schema_validation = True
        print(
            f"Running experiment with {model} in conversation mode with structured response prompting."
        )
    else:
        print(f"Model {model} does NOT support response schema validation.")
        response_format = ResponseFormat.CHAIN_OF_THOUGHT
        print(
            f"Running experiment with {model} in conversation mode with Chain-Of-Thought prompting."
        )

    # Load prompts
    prompt_prefix, prompt_suffix = load_prompts(response_format)

    # Set up CSV file
    csv_file, writer, start_offset, remaining_responses = setup_csv_file(
        model, response_format, num_responses
    )

    try:
        # Process each conversation thread
        for conversation_thread in tqdm(
            range(start_offset, start_offset + remaining_responses)
        ):
            thread_answers = []
            thread_success = True
            messages = []

            # Iterate over the scenarios
            for scenario_id in [1, 2, 3, 4, 5, 6, 7, 8]:
                messages, answer_row = process_scenario(
                    model,
                    messages,
                    scenario_id,
                    prompt_prefix,
                    prompt_suffix,
                    response_format,
                    conversation_thread,
                    num_retries,
                )

                thread_answers.append(answer_row)

                # Track if we hit any failures
                if answer_row[3] in ["PARSING_FAILED", "ERROR"]:
                    thread_success = False

            # Save results to CSV after each thread
            try:
                writer.writerows(thread_answers)
                csv_file.flush()
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

        logger.info(f"Experiment completed. Results saved to CSV file.")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user. Partial results have been saved.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM experiment.")

    parser.add_argument(
        "--num_responses",
        type=int,
        help="The number of responses to generate",
        default=1000,
        required=False,
    )

    parser.add_argument(
        "--num_retries",
        type=int,
        help="The number of times to re-try generating responses if there is a parsing/API/network/etc. error",
        default=20,
        required=False,
    )

    parser.add_argument(
        "model",
        type=str,
        help="The 'provider/model' string, e.g., 'openai/gpt-5-nano-2025-08-07'.",
    )

    args = parser.parse_args()

    run_experiment(args.model, args.num_responses, args.num_retries)
