import logging
import os
import json
import asyncio
from pathlib import Path
import time # Added for potential retries

import pyarrow as pa
from dora import DoraStatus

# --- Dependency Imports ---
try:
    # For LLM call (replace with your actual client/wrapper if different)
    from openai import OpenAI, AsyncOpenAI
    # For summarizing
    from transformers import GPT2Tokenizer
except ImportError as e:
    logging.error(f"Failed to import dependencies (openai, transformers): {e}. Please install them.")
    raise

try:
    # Use MoFA utils if available
    from mofa.utils.files.read import read_yaml
    from mofa.utils.files.dir import get_relative_path
    from mofa.kernel.utils.util import create_agent_output, load_node_result
except ImportError:
    # Provide simple fallbacks if mofa utils are not in the path
    import yaml
    logging.warning("MoFA utilities not found. Using basic fallback implementations.")
    def get_relative_path(current_file, sibling_directory_name, target_file_name):
        base_dir = Path(current_file).parent
        # Adjust path assuming script is in 'scripts' and config in 'configs'
        return base_dir.parent / sibling_directory_name / target_file_name

    def read_yaml(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error reading YAML file {file_path}: {e}")
            return None

    def create_agent_output(agent_name, agent_result, dataflow_status=False):
        return {"agent_name": agent_name, "agent_result": agent_result, "dataflow_status": dataflow_status}

    def load_node_result(input_data):
         if isinstance(input_data, dict) and "agent_result" in input_data:
             return input_data["agent_result"]
         return input_data # Return as is if format is unexpected


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuestionGenOperator")

# --- Helper Functions (from original Step 2 script) ---

def get_summary(context: str, tokenizer, total_tokens: int = 2000, min_tokens_for_summary=50):
    """Generates a summary by taking start and end tokens."""
    try:
        tokens = tokenizer.tokenize(context)
        if len(tokens) < min_tokens_for_summary:
            logger.warning(f"Context too short ({len(tokens)} tokens) for meaningful summary. Returning original.")
            # Return original context or a fixed message, maybe empty string?
            return context[:total_tokens*5] # Limit length roughly

        half_tokens = total_tokens // 2

        # Adjusted slicing to be safer for short contexts
        start_slice_end = min(half_tokens, len(tokens))
        end_slice_start = max(0, len(tokens) - half_tokens)

        start_tokens = tokens[:start_slice_end]
        end_tokens = tokens[end_slice_start:]

        # Avoid duplicating the middle part if slices overlap significantly
        if end_slice_start < start_slice_end:
             logger.debug("Token slices overlap, using combined slice.")
             summary_tokens = tokens[:min(total_tokens, len(tokens))] # Just take the beginning up to limit
        else:
             summary_tokens = start_tokens + end_tokens

        summary = tokenizer.convert_tokens_to_string(summary_tokens)
        return summary
    except Exception as e:
        logger.error(f"Error during tokenization/summary: {e}", exc_info=True)
        # Fallback: return truncated original context
        return context[:total_tokens*5] # Rough limit


async def call_llm_for_questions(client: AsyncOpenAI, model: str, prompt: str, max_retries=3, delay=5):
    """Calls the LLM with retries to generate questions."""
    messages = [{"role": "user", "content": prompt}]
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Sending request to LLM ({model})... (Attempt {retries + 1})")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                # Add other params like temperature if needed from config
            )
            result = response.choices[0].message.content
            logger.info("LLM response received successfully.")
            return result
        except Exception as e:
            retries += 1
            logger.warning(f"LLM call failed (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                logger.info(f"Retrying LLM call in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error("LLM call failed after maximum retries.")
                return None # Indicate failure

# --- DORA Operator Class ---

class Operator:
    """
    DORA Operator for generating questions based on dataset summaries using an LLM.
    """
    def __init__(self):
        """Initialize the operator."""
        self.config = None
        self.processed = False
        self.tokenizer = None
        self.llm_client = None
        logger.info("QuestionGenOperator initialized.")

    def load_config(self):
        """Loads configuration from YAML."""
        try:
            yaml_file_path = get_relative_path(__file__, 'configs', 'question_gen_config.yml')
            logger.info(f"Loading configuration from: {yaml_file_path}")
            config = read_yaml(yaml_file_path)
            if config and "QUESTION_GEN" in config:
                self.config = config["QUESTION_GEN"]
                logger.info("Configuration loaded successfully.")
                # Validate required keys
                required_keys = [
                    "INPUT_CONTEXTS_FILENAME", "SUMMARY_TOTAL_TOKENS",
                    "OUTPUT_QUESTIONS_FILE", "LLM_API_KEY", "LLM_BASE_URL",
                    "LLM_MODEL_NAME", "TOKENIZER_MODEL"
                ]
                if not all(key in self.config for key in required_keys):
                    missing = [key for key in required_keys if key not in self.config]
                    logger.error(f"Configuration missing required keys: {missing}")
                    self.config = None
                    return False

                # Resolve output file path (relative to project root/script parent's parent)
                script_dir_parent = Path(__file__).parent.parent
                self.config["OUTPUT_QUESTIONS_FILE"] = str(script_dir_parent / self.config["OUTPUT_QUESTIONS_FILE"])
                # Ensure output directory exists
                output_dir = Path(self.config["OUTPUT_QUESTIONS_FILE"]).parent
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Resolved OUTPUT_QUESTIONS_FILE: {self.config['OUTPUT_QUESTIONS_FILE']}")

                return True
            else:
                logger.error("'QUESTION_GEN' section not found in YAML or config is empty.")
                return False
        except Exception as e:
            logger.error(f"Failed to load or process configuration: {e}", exc_info=True)
            return False

    def _setup_dependencies(self):
        """Initializes tokenizer and LLM client based on config."""
        if not self.config:
            logger.error("Cannot setup dependencies without configuration.")
            return False
        try:
            logger.info(f"Loading tokenizer: {self.config['TOKENIZER_MODEL']}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['TOKENIZER_MODEL'])
            logger.info("Tokenizer loaded.")

            logger.info("Initializing AsyncOpenAI client...")
            self.llm_client = AsyncOpenAI(
                api_key=self.config['LLM_API_KEY'],
                base_url=self.config['LLM_BASE_URL'],
            )
            logger.info("AsyncOpenAI client initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to setup dependencies: {e}", exc_info=True)
            self.tokenizer = None
            self.llm_client = None
            return False

    async def _generate_questions_task(self, unique_contexts_dir: str):
        """The core async task for loading data, summarizing, and generating questions."""
        if not self.config or not self.tokenizer or not self.llm_client:
            logger.error("Operator not properly configured or dependencies not set up.")
            return None # Indicate failure

        try:
            # Construct the path to the specific context file needed
            context_file_name = self.config["INPUT_CONTEXTS_FILENAME"]
            context_file_path = os.path.join(unique_contexts_dir, context_file_name)
            logger.info(f"Attempting to load contexts from: {context_file_path}")

            if not os.path.exists(context_file_path):
                logger.error(f"Required context file not found: {context_file_path}")
                return None

            # Load contexts from the specified file
            with open(context_file_path, mode="r", encoding="utf-8") as f:
                unique_contexts = json.load(f)

            if not isinstance(unique_contexts, list) or not unique_contexts:
                 logger.error(f"Context file {context_file_path} is empty or not a list.")
                 return None

            # Generate summaries (as per original script: uses all contexts from the file)
            logger.info(f"Generating summaries for {len(unique_contexts)} contexts...")
            summary_tokens = self.config.get("SUMMARY_TOTAL_TOKENS", 2000)
            summaries = [get_summary(ctx, self.tokenizer, summary_tokens) for ctx in unique_contexts]
            total_description = "\n\n".join(summaries)
            # Optional: Truncate total_description if it's excessively long for the LLM prompt
            # max_desc_length = 10000 # Example limit
            # if len(total_description) > max_desc_length:
            #    logger.warning(f"Truncating total description from {len(total_description)} to {max_desc_length} characters.")
            #    total_description = total_description[:max_desc_length]

            logger.info(f"Total description length: {len(total_description)} characters.")

            # Construct the prompt
            prompt = f"""
Given the following description generated from summaries of a dataset's contexts:

{total_description}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would likely perform using this dataset. Then, for each (user, task) combination, generate 5 distinct questions that require a high-level understanding or analysis of the entire dataset, not just retrieval of specific snippets. Ensure the questions probe deeper insights, comparisons, or overarching themes present in the data.

Output the results strictly in the following structure:
- User 1: [Concise user description, e.g., AI Researcher]
    - Task 1: [Concise task description, e.g., Analyzing bias in language models]
        - Question 1: [Generated Question]
        - Question 2: [Generated Question]
        - Question 3: [Generated Question]
        - Question 4: [Generated Question]
        - Question 5: [Generated Question]
    - Task 2: [Concise task description]
            ...
    - Task 5: [Concise task description]
- User 2: [Concise user description]
    ...
- User 5: [Concise user description]
    ...
"""
            # Call the LLM to generate questions
            logger.info("Requesting question generation from LLM...")
            generated_content = await call_llm_for_questions(
                client=self.llm_client,
                model=self.config['LLM_MODEL_NAME'],
                prompt=prompt
            )

            if not generated_content:
                logger.error("Failed to get response from LLM for question generation.")
                return None

            # Save the result
            output_file_path = self.config["OUTPUT_QUESTIONS_FILE"]
            try:
                with open(output_file_path, "w", encoding="utf-8") as file:
                    file.write(generated_content)
                logger.info(f"Generated questions successfully saved to: {output_file_path}")
                return output_file_path # Return the path on success
            except IOError as e:
                logger.error(f"Failed to write questions to file {output_file_path}: {e}")
                return None

        except FileNotFoundError:
             logger.error(f"Context file not found during processing: {context_file_path}")
             return None
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON from {context_file_path}: {e}")
             return None
        except Exception as e:
            logger.error(f"An unhandled error occurred during question generation task: {e}", exc_info=True)
            return None # Indicate failure

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        """Handles DORA input events to trigger question generation."""
        if dora_event["type"] == "INPUT":
            event_id = dora_event["id"]
            logger.debug(f"Received input event on ID: {event_id}")

            # Expecting input from the data prep operator (same as Step 1 input)
            if event_id == "unique_contexts_dir":
                if self.processed:
                    logger.info("Question generation already processed. Ignoring further inputs.")
                    return DoraStatus.CONTINUE

                logger.info("Received unique contexts directory path. Starting question generation...")

                # Load configuration
                if not self.config:
                    if not self.load_config():
                        logger.error("Failed to load configuration. Stopping.")
                        return DoraStatus.STOP

                # Setup tokenizer and LLM client
                if not self.tokenizer or not self.llm_client:
                    if not self._setup_dependencies():
                        logger.error("Failed to setup dependencies. Stopping.")
                        return DoraStatus.STOP

                # --- Run the Main Async Task ---
                async def run_main_task():
                    input_data = dora_event["value"][0].as_py()
                    unique_contexts_dir = load_node_result(input_data) # Extract path

                    if not unique_contexts_dir or not os.path.isdir(unique_contexts_dir):
                        logger.error(f"Invalid input: Expected a valid directory path for 'unique_contexts_dir', got '{unique_contexts_dir}'. Stopping.")
                        return None # Indicate failure to the sync wrapper

                    return await self._generate_questions_task(unique_contexts_dir)

                # Execute the async task
                loop = asyncio.get_event_loop()
                output_file_path = None
                try:
                    output_file_path = loop.run_until_complete(run_main_task())
                except RuntimeError as e:
                     logger.warning(f"Asyncio loop issue: {e}. Attempting direct run (check context).")
                     # Fallback if loop is managed differently by DORA

                # --- Handle Result ---
                if output_file_path:
                    logger.info("Question generation completed successfully.")
                    # Send the path to the generated questions file
                    output_data = create_agent_output(
                        agent_name="question_generator",
                        agent_result=output_file_path,
                        dataflow_status=False
                    )
                    send_output(
                        "generated_questions_file", # Output ID for the next node
                        pa.array([output_data]),
                        dora_event['metadata']
                    )
                    logger.info(f"Sent generated questions file path: {output_file_path}")
                else:
                    logger.error("Question generation failed. No output sent.")
                    self.processed = True # Mark as processed (failed)
                    return DoraStatus.STOP # Stop due to failure

                self.processed = True # Mark processing as complete
                logger.info("QuestionGenOperator processing finished.")
                return DoraStatus.STOP # Stop after successful run

            else:
                logger.debug(f"Ignoring input from ID: {event_id}")


        elif dora_event["type"] == "STOP":
            logger.info("Received STOP event. Shutting down QuestionGenOperator.")
        elif dora_event["type"] == "ERROR":
             logger.error(f"Received ERROR event: {dora_event['error']}")
        else:
            logger.debug(f"Received unknown event type: {dora_event['type']}")

        return DoraStatus.CONTINUE