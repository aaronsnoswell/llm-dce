import re
import os
import csv
import logging
from pprint import pprint
from tqdm import tqdm
from ollama import Client
from datetime import datetime


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

MODEL = "llama3.2"
NUM_RETRIES = 20  # Number of times to retry a response if the output is malformed
NUM_RESPONSES = 1000  # Number of independent conversation threads to run
START_RESPONSE_OFFSET = 0  # Offset to start the conversation threads from

print(f"Running experiment with {MODEL} in conversation mode with Chain-Of-Thought prompting.")
print(f"Number of responses: {NUM_RESPONSES}")
print(f"Starting from response offset: {START_RESPONSE_OFFSET}")

# Load the contents of prompt files
try:
    with open("prompt-prefix.txt", "r") as file:
        prompt_prefix = file.read()

    with open("prompt-suffix-cot.txt", "r") as file:
        prompt_suffix = file.read()
except FileNotFoundError as e:
    logger.error(f"Prompt file not found: {e}")
    raise

# Prepare CSV output file
csv_filename = f"{MODEL}-responses-conversation-cot.csv"
columns = ["model", "conversation_thread", "scenario_id", "choice", "reason"]

# Check if file exists to determine if we need to write headers
file_exists = os.path.isfile(csv_filename)

# Create or open the CSV file
try:
    csv_file = open(csv_filename, "a", newline="")
    writer = csv.writer(csv_file)
    # Write header only if the file is newly created
    if not file_exists:
        writer.writerow(columns)
        csv_file.flush()
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
                    try:
                        # Get response
                        response = Client().chat(model=MODEL, messages=messages)
                        scenario_output = response.message.content

                        # Parse output
                        output_choice = re.search(r"Choice:\s*(.*)", scenario_output)
                        output_reason = "N/A"

                        # Extract just the part after the colon
                        if output_choice:
                            output_choice = output_choice.group(1).strip()
                        else:
                            output_choice = None

                        # If we got valid output, break out of retry loop 
                        if output_choice is not None:
                            break

                        logger.warning(
                            f"Thread {conversation_thread}, Scenario {scenario_id}: Retry {retry+1}/{NUM_RETRIES} - Failed to parse response"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Thread {conversation_thread}, Scenario {scenario_id}: Retry {retry+1}/{NUM_RETRIES} - Exception: {e}"
                        )
                        # Continue with next retry

                # After all retries, check if we got valid output
                if output_choice is not None and output_reason is not None:
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
                            output_reason,
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
                "completed successfully" if thread_success else "completed with errors"
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
