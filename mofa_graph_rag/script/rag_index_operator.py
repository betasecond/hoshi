import logging
import os
import json
import time
import asyncio
import glob
import numpy as np
from pathlib import Path

import pyarrow as pa
from dora import DoraStatus

# --- Dependency Imports ---
try:
    from lightrag import LightRAG
    from lightrag.utils.types import EmbeddingFunc
    # Note: lightrag.llm.openai might need adjustment if using different providers or custom caching
    from lightrag.llm.openai import openai_complete, openai_embed
    # from lightrag.kg.shared_storage import initialize_pipeline_status # Maybe needed? Check lightrag example context
except ImportError as e:
    logging.error(f"Failed to import LightRAG components: {e}. Ensure lightrag is installed.")
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
        return base_dir.parent / sibling_directory_name / target_file_name # Assumes script is in 'scripts'

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
         # Basic extraction assuming the structure from create_agent_output
         if isinstance(input_data, dict) and "agent_result" in input_data:
             return input_data["agent_result"]
         return input_data # Return as is if format is unexpected


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RagIndexOperator")

# --- DORA Operator Class ---

class Operator:
    """
    DORA Operator for building the LightRAG index from unique context files.
    """
    def __init__(self):
        """Initialize the operator."""
        self.config = None
        self.rag_instance = None
        self.processed = False
        self.llm_model_func = None
        self.embedding_func_wrapper = None
        logger.info("RagIndexOperator initialized.")

    def load_config(self):
        """Loads configuration from YAML."""
        try:
            yaml_file_path = get_relative_path(__file__, 'configs', 'rag_index_config.yml')
            logger.info(f"Loading configuration from: {yaml_file_path}")
            config = read_yaml(yaml_file_path)
            if config and "RAG_INDEX" in config:
                self.config = config["RAG_INDEX"]
                logger.info("Configuration loaded successfully.")
                # Validate required keys
                required_keys = [
                    "WORKING_DIR", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME",
                    "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL_NAME",
                    "EMBEDDING_DIM", "EMBEDDING_MAX_TOKEN_SIZE", "EMBEDDING_BATCH_NUM",
                    "INPUT_CONTEXTS_FILENAME_PATTERN"
                ]
                if not all(key in self.config for key in required_keys):
                    missing = [key for key in required_keys if key not in self.config]
                    logger.error(f"Configuration missing required keys: {missing}")
                    self.config = None
                    return False

                # Resolve relative paths (make absolute relative to script dir parent)
                script_dir_parent = Path(__file__).parent.parent
                self.config["WORKING_DIR"] = str(script_dir_parent / self.config["WORKING_DIR"])
                logger.info(f"Resolved WORKING_DIR: {self.config['WORKING_DIR']}")
                os.makedirs(self.config["WORKING_DIR"], exist_ok=True) # Ensure working dir exists

                return True
            else:
                logger.error("'RAG_INDEX' section not found in YAML or config is empty.")
                return False
        except Exception as e:
            logger.error(f"Failed to load or process configuration: {e}", exc_info=True)
            return False

    def _setup_async_funcs(self):
        """Sets up the async LLM and embedding functions based on config."""
        if not self.config:
            logger.error("Cannot setup async functions without configuration.")
            return False

        # Define LLM function within this scope to capture config
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
            logger.debug(f"Calling LLM: {self.config['LLM_MODEL_NAME']}")
            # Using basic openai_complete, add caching/retries if needed
            return await openai_complete(
                model=self.config['LLM_MODEL_NAME'],
                prompt=prompt,
                system_prompt=system_prompt,
                # history_messages=history_messages, # Pass if needed by lightrag structure
                api_key=self.config['LLM_API_KEY'],
                base_url=self.config['LLM_BASE_URL'],
                 # timeout=kwargs.get("timeout", 60), # Example: add timeout
                **kwargs,
            )
        self.llm_model_func = llm_model_func

        # Define Embedding function within this scope
        async def embedding_func(texts: list[str]) -> np.ndarray:
            logger.debug(f"Calling Embedding: {self.config['EMBEDDING_MODEL_NAME']} for {len(texts)} texts")
            return await openai_embed(
                texts=texts,
                model=self.config['EMBEDDING_MODEL_NAME'],
                api_key=self.config['EMBEDDING_API_KEY'],
                base_url=self.config['EMBEDDING_BASE_URL'],
                 # timeout=60, # Example: add timeout
            )

        # Wrap embedding func with LightRAG's EmbeddingFunc structure
        self.embedding_func_wrapper = EmbeddingFunc(
            embedding_dim=self.config['EMBEDDING_DIM'],
            max_token_size=self.config['EMBEDDING_MAX_TOKEN_SIZE'],
            func=embedding_func
        )
        logger.info("Async LLM and Embedding functions configured.")
        return True

    async def _initialize_rag_instance(self):
        """Initializes the LightRAG instance."""
        if not self.config or not self.llm_model_func or not self.embedding_func_wrapper:
            logger.error("Cannot initialize RAG: Missing config or async functions.")
            return False
        try:
            logger.info("Initializing LightRAG instance...")
            self.rag_instance = LightRAG(
                working_dir=self.config['WORKING_DIR'],
                llm_model_func=self.llm_model_func,
                embedding_func=self.embedding_func_wrapper,
                embedding_batch_num=self.config['EMBEDDING_BATCH_NUM']
            )
            logger.info("LightRAG instance created. Initializing storages...")
            await self.rag_instance.initialize_storages()
            logger.info("LightRAG storages initialized.")

            # Initialize pipeline status if required by your lightrag version/setup
            # await initialize_pipeline_status()
            # logger.info("Pipeline status initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}", exc_info=True)
            self.rag_instance = None
            return False

    async def _insert_contexts_from_file(self, file_path: str):
        """Reads contexts from a JSON file and inserts them into RAG."""
        if not self.rag_instance:
            logger.error("RAG instance not available for insertion.")
            return False

        logger.info(f"Processing context file: {file_path}")
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                unique_contexts = json.load(f)
            if not isinstance(unique_contexts, list):
                 logger.error(f"Expected a list of contexts in {file_path}, found {type(unique_contexts)}. Skipping.")
                 return False
            if not unique_contexts:
                 logger.info(f"Context file {file_path} is empty. Nothing to insert.")
                 return True # Not an error, just nothing to do

            logger.info(f"Attempting to insert {len(unique_contexts)} contexts from {os.path.basename(file_path)}.")

            # --- Insertion Logic with Retries (from original script) ---
            retries = 0
            max_retries = self.config.get("INSERT_MAX_RETRIES", 3)
            retry_delay = self.config.get("INSERT_RETRY_DELAY", 10)

            while retries < max_retries:
                try:
                    # Assuming rag.insert is synchronous in the version used,
                    # but underlying embedding might be async. Check lightrag docs.
                    # If rag.insert itself needs await: await self.rag_instance.insert(unique_contexts)
                    self.rag_instance.insert(unique_contexts)
                    logger.info(f"Successfully inserted contexts from {os.path.basename(file_path)}.")
                    return True # Success
                except Exception as e:
                    retries += 1
                    logger.warning(f"Insertion failed for {os.path.basename(file_path)} (attempt {retries}/{max_retries}), error: {e}")
                    if retries < max_retries:
                        logger.info(f"Retrying insertion in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay) # Use asyncio.sleep for async context
                    else:
                        logger.error(f"Insertion failed for {os.path.basename(file_path)} after exceeding max retries.")
                        return False # Failure after retries

        except FileNotFoundError:
            logger.error(f"Context file not found: {file_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {file_path}: {e}", exc_info=True)
            return False
        return False # Should have returned earlier

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        """Handles DORA input events to trigger RAG indexing."""
        if dora_event["type"] == "INPUT":
            event_id = dora_event["id"]
            logger.debug(f"Received input event on ID: {event_id}")

            # Expecting input from the data prep operator
            if event_id == "unique_contexts_dir":
                if self.processed:
                    logger.info("RAG indexing already processed. Ignoring further inputs.")
                    return DoraStatus.CONTINUE

                logger.info("Received unique contexts directory path. Starting RAG indexing...")

                # Load configuration if not already done
                if not self.config:
                    if not self.load_config():
                        logger.error("Failed to load configuration. Stopping.")
                        return DoraStatus.STOP

                # Setup async functions for LLM/Embedding
                if not self.llm_model_func or not self.embedding_func_wrapper:
                    if not self._setup_async_funcs():
                         logger.error("Failed to setup async functions. Stopping.")
                         return DoraStatus.STOP

                # --- Main Async Logic ---
                async def main_indexing_task():
                    try:
                        # Initialize RAG instance
                        if not await self._initialize_rag_instance():
                            return False # Initialization failed

                        # Load the input directory path
                        input_data = dora_event["value"][0].as_py()
                        unique_contexts_dir = load_node_result(input_data) # Use helper to extract path

                        if not unique_contexts_dir or not os.path.isdir(unique_contexts_dir):
                            logger.error(f"Invalid input: Expected a valid directory path for 'unique_contexts_dir', got '{unique_contexts_dir}'. Stopping.")
                            return False

                        logger.info(f"Using unique contexts directory: {unique_contexts_dir}")

                        # Find all context files matching the pattern
                        file_pattern = self.config["INPUT_CONTEXTS_FILENAME_PATTERN"]
                        context_files = glob.glob(os.path.join(unique_contexts_dir, file_pattern))

                        if not context_files:
                            logger.warning(f"No context files found matching pattern '{file_pattern}' in {unique_contexts_dir}. Indexing will be empty.")
                            # Decide if this is an error or just proceed with an empty index
                            # Let's proceed but log clearly.

                        all_insertions_successful = True
                        for file_path in context_files:
                            success = await self._insert_contexts_from_file(file_path)
                            if not success:
                                all_insertions_successful = False
                                # Decide whether to stop on first failure or try all files
                                logger.error(f"Stopping indexing process due to failure in file: {os.path.basename(file_path)}")
                                return False # Stop on first failure

                        if not all_insertions_successful:
                             logger.error("One or more context file insertions failed.")
                             return False

                        logger.info("All context files processed successfully.")
                        return True # Indicate overall success

                    except Exception as e:
                        logger.error(f"An unhandled error occurred during indexing task: {e}", exc_info=True)
                        return False # Indicate failure

                # Run the main async task
                # Using asyncio.get_event_loop().run_until_complete() if on_event is sync
                # If DORA runs operators in an async loop, direct await might work. Check DORA docs.
                # Assuming sync context for on_event:
                loop = asyncio.get_event_loop()
                try:
                    success = loop.run_until_complete(main_indexing_task())
                except RuntimeError as e:
                     # Handle potential loop already running issues if DORA manages the loop
                     logger.warning(f"Asyncio loop issue: {e}. Trying direct execution (might fail if not in async context).")
                     # This fallback is less likely to work correctly but shows intent
                     # success = await main_indexing_task() # This line would need on_event to be async def

                if success:
                    logger.info("RAG indexing completed successfully.")
                    # Send the working directory path as output
                    output_data = create_agent_output(
                        agent_name="rag_indexer",
                        agent_result=self.config['WORKING_DIR'], # Send the path to the index
                        dataflow_status=False
                    )
                    send_output(
                        "rag_index_dir", # Output ID for the next node
                        pa.array([output_data]),
                        dora_event['metadata']
                    )
                    logger.info(f"Sent RAG index directory path: {self.config['WORKING_DIR']}")
                else:
                    logger.error("RAG indexing failed. No output sent.")
                    # Optionally send an error status output
                    self.processed = True # Mark as processed (failed)
                    return DoraStatus.STOP # Stop due to failure

                self.processed = True # Mark processing as complete
                logger.info("RagIndexOperator processing finished.")
                return DoraStatus.STOP # Stop after successful run

            else:
                logger.debug(f"Ignoring input from ID: {event_id}")


        elif dora_event["type"] == "STOP":
            logger.info("Received STOP event. Shutting down RagIndexOperator.")
        elif dora_event["type"] == "ERROR":
             logger.error(f"Received ERROR event: {dora_event['error']}")
        else:
            logger.debug(f"Received unknown event type: {dora_event['type']}")

        return DoraStatus.CONTINUE