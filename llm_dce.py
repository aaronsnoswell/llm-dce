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
import portalocker

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


def load_scenarios() -> Dict[int, str]:
    """Load all scenario files once at startup.

    Returns:
        Dictionary mapping scenario_id (1-8) to scenario text
    """
    scenarios = {}
    for scenario_id in range(1, 9):
        try:
            with open(f"scenario-0{scenario_id}.txt", "r") as file:
                scenarios[scenario_id] = file.read()
            logger.info(f"Loaded scenario {scenario_id}")
        except FileNotFoundError as e:
            logger.error(f"Scenario file not found: scenario-0{scenario_id}.txt")
            raise

    print(f"Loaded {len(scenarios)} scenario files into memory")
    return scenarios


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


def get_csv_filename(model: str, response_format: ResponseFormat) -> str:
    """Generate standardized CSV filename for a model.

    Args:
        model: Model identifier string
        response_format: The response format type

    Returns:
        CSV filename string
    """
    return f"{model.replace('/', '-').replace(':', '-')}-responses-conversation-{response_format.value}.csv"


def count_existing_responses(csv_filename: str) -> int:
    """Count how many complete response sets (8 scenarios) exist in CSV.

    Args:
        csv_filename: Path to the CSV file

    Returns:
        Number of complete response sets (conversation threads)
    """
    if not os.path.isfile(csv_filename):
        return 0

    try:
        with portalocker.Lock(
            csv_filename, "r", timeout=30, encoding="utf-8", errors="replace"
        ) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row_count = sum(1 for row in reader)
            # Each conversation thread has 8 scenarios
            return row_count // 8
    except portalocker.exceptions.LockException:
        logger.warning(
            f"Could not acquire lock to count responses in {csv_filename}, assuming 0"
        )
        return 0
    except Exception as e:
        logger.warning(f"Error counting responses in {csv_filename}: {e}, assuming 0")
        return 0


def atomic_append_with_thread_id(csv_filename: str, rows: list, model: str) -> int:
    """Atomically read max conversation_thread, update rows, and append to CSV.

    This function performs all operations within a single file lock to ensure atomicity:
    1. Acquires exclusive lock
    2. Reads existing data to find max conversation_thread
    3. Updates the provided rows with the next conversation_thread value
    4. Appends the updated rows
    5. Releases lock

    Args:
        csv_filename: Path to the CSV file
        rows: List of rows to append (conversation_thread values will be updated)
        model: Model identifier (for logging)

    Returns:
        The conversation_thread number that was assigned to these rows
    """
    columns = ["model", "conversation_thread", "scenario_id", "choice", "discussion"]

    try:
        # Check if file exists
        file_exists = os.path.isfile(csv_filename)

        # Open with exclusive lock (use r+ if exists, w+ if new)
        mode = "r+" if file_exists else "w+"

        with portalocker.Lock(
            csv_filename,
            mode,
            timeout=30,
            encoding="utf-8",
            newline="",
            errors="replace",
        ) as f:
            # If file exists, read to find max conversation_thread
            next_thread = 0
            if file_exists:
                try:
                    reader = csv.DictReader(f)
                    max_thread = -1
                    for row in reader:
                        try:
                            thread_num = int(row["conversation_thread"])
                            max_thread = max(max_thread, thread_num)
                        except (ValueError, KeyError):
                            continue
                    next_thread = max_thread + 1
                except Exception as e:
                    logger.warning(
                        f"Error reading existing threads, assuming next is 0: {e}"
                    )
                    next_thread = 0

                # Move to end of file for appending
                f.seek(0, 2)
            else:
                # New file - write header
                writer = csv.writer(f)
                writer.writerow(columns)
                logger.info(f"Created new CSV file: {csv_filename}")

            # Update all rows with the assigned conversation_thread
            updated_rows = []
            for row in rows:
                # row format: [model, conversation_thread, scenario_id, choice, discussion]
                # Update the conversation_thread (index 1)
                row[1] = next_thread
                updated_rows.append(row)

            # Write all updated rows
            writer = csv.writer(f)
            writer.writerows(updated_rows)
            f.flush()

            logger.info(
                f"Assigned conversation_thread={next_thread} and wrote {len(updated_rows)} rows"
            )
            return next_thread

    except portalocker.exceptions.LockException:
        logger.error(f"Could not acquire lock to write to {csv_filename}")
        raise
    except Exception as e:
        logger.error(f"Error in atomic append operation for {csv_filename}: {e}")
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

        # The models can be a bit markdown-happy, so we allow for that
        if "Choice: A" in final_line or "**Choice:** A" in final_line:
            output_choice = "A"
        elif "Choice: B" in final_line or "**Choice:** B" in final_line:
            output_choice = "B"
        elif "Choice: C" in final_line or "**Choice:** C" in final_line:
            output_choice = "C"
        else:
            raise ValueError(f"Failed to parse choice from CoT response: {final_line}")
        # Output discussion is all but the final line
        output_discussion = scenario_output

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
    scenario_text: str,
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
        scenario_text: The text content of the scenario
        prompt_prefix: Prefix for first scenario only
        prompt_suffix: Suffix for all scenarios
        response_format: The response format type (STRUCTURED or CHAIN_OF_THOUGHT)
        conversation_thread: Thread number for logging (placeholder, will be updated later)
        num_retries: Number of retry attempts

    Returns:
        Tuple of (updated_messages, answer_row)
    """
    try:
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
                None,  # Placeholder - will be filled by atomic_append_with_thread_id
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
                None,  # Placeholder - will be filled by atomic_append_with_thread_id
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
            None,  # Placeholder - will be filled by atomic_append_with_thread_id
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

    # Load all scenarios once at startup
    scenarios = load_scenarios()

    # Load prompts
    prompt_prefix, prompt_suffix = load_prompts(response_format)

    # Get CSV filename
    csv_filename = get_csv_filename(model, response_format)
    print(f"Output CSV file: {csv_filename}")

    try:
        # Create progress bar for full expected number of responses
        pbar = tqdm(total=num_responses, desc="Generating responses")

        iteration = 0
        while iteration < num_responses:
            # Check how many responses already exist
            existing_responses = count_existing_responses(csv_filename)

            # Update progress bar to reflect existing work
            if pbar.n < existing_responses:
                pbar.update(existing_responses - pbar.n)

            # Exit if we already have enough responses
            if existing_responses >= num_responses:
                logger.info(
                    f"Target of {num_responses} responses already reached. Exiting."
                )
                break

            # Generate responses for this thread (conversation_thread will be assigned during write)
            thread_answers = []
            thread_success = True
            messages = []

            # Iterate over the scenarios
            for scenario_id in [1, 2, 3, 4, 5, 6, 7, 8]:
                messages, answer_row = process_scenario(
                    model,
                    messages,
                    scenario_id,
                    scenarios[scenario_id],
                    prompt_prefix,
                    prompt_suffix,
                    response_format,
                    pbar.n + 1,
                    num_retries,
                )

                thread_answers.append(answer_row)

                # Track if we hit any failures
                if answer_row[3] in ["PARSING_FAILED", "ERROR"]:
                    thread_success = False

            # Atomically append results to CSV (thread ID assigned here)
            try:
                assigned_thread = atomic_append_with_thread_id(
                    csv_filename, thread_answers, model
                )
                completed_msg = (
                    "completed successfully"
                    if thread_success
                    else "completed with errors"
                )
                logger.info(
                    f"Thread {assigned_thread} {completed_msg}. Saved {len(thread_answers)} results."
                )

                # Update progress bar
                pbar.update(1)

            except Exception as e:
                logger.error(f"Failed to write results: {e}")

            iteration += 1

        pbar.close()
        logger.info(f"Experiment completed. Results saved to {csv_filename}")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user. Partial results have been saved.")
        if "pbar" in locals():
            pbar.close()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        if "pbar" in locals():
            pbar.close()


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
        "model", type=str, help="The model string, e.g., 'gpt-5-nano-2025-08-07'."
    )

    args = parser.parse_args()

    run_experiment(args.model, args.num_responses, args.num_retries)
