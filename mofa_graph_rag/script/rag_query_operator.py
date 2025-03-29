import logging
import os
import json
import asyncio
import re
from pathlib import Path
import time # For potential delays/retries

import pyarrow as pa
from dora import DoraStatus
import numpy as np # Required by embedding func signature

# --- Dependency Imports ---
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils.types import EmbeddingFunc
    # Assuming openai client is needed for the funcs passed to LightRAG
    from lightrag.llm.openai import openai_complete, openai_embed
except ImportError as e:
    logging.error(f"Failed to import LightRAG/OpenAI components: {e}. Ensure lightrag and openai are installed.")
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
        # Ensure agent_result is serializable, might need json.dumps if complex
        return {"agent_name": agent_name, "agent_result": agent_result, "dataflow_status": dataflow_status}

    def load_node_result(input_data):
         if isinstance(input_data, dict) and "agent_result" in input_data:
             return input_data["agent_result"]
         # Handle cases where input might be raw string/path directly
         return input_data


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RagQueryOperator")


# --- Helper Function (from original Step 3 script) ---
def extract_queries(file_path):
    """Extracts queries from a text file using regex."""
    logger.info(f"Extracting queries from: {file_path}")
    queries = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        # Simple cleaning, might need more robust handling
        data = data.replace("**", "")

        # Regex to find lines starting with "- Question \d: "
        # Make sure regex handles potential variations or multiline questions if any
        found_queries = re.findall(r"-\s*Question\s*\d+:\s*(.+)", data, re.IGNORECASE | re.MULTILINE)
        # Strip leading/trailing whitespace from each found query
        queries = [q.strip() for q in found_queries if q.strip()]

        logger.info(f"Extracted {len(queries)} queries.")
        if not queries:
            logger.warning(f"No queries extracted from {file_path}. Check file content and regex pattern.")
    except FileNotFoundError:
        logger.error(f"Questions file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error extracting queries from {file_path}: {e}", exc_info=True)
    return queries


# --- DORA Operator Class ---

class Operator:
    """
    DORA Operator for querying a RAG system with generated questions.
    """
    def __init__(self):
        """Initialize the operator."""
        self.config = None
        self.rag_instance = None
        self.processed = False
        self.questions_file_path = None
        self.rag_index_dir = None
        self.llm_model_func = None
        self.embedding_func_wrapper = None
        logger.info("RagQueryOperator initialized.")

    def load_config(self):
        """Loads configuration from YAML."""
        try:
            yaml_file_path = get_relative_path(__file__, 'configs', 'rag_query_config.yml')
            logger.info(f"Loading configuration from: {yaml_file_path}")
            config = read_yaml(yaml_file_path)
            if config and "RAG_QUERY" in config:
                self.config = config["RAG_QUERY"]
                logger.info("Configuration loaded successfully.")
                # Validate required keys
                required_keys = [
                    "QUERY_MODE", "OUTPUT_RESULTS_FILE", "OUTPUT_ERRORS_FILE",
                    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME",
                    "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL_NAME",
                    "EMBEDDING_DIM", "EMBEDDING_MAX_TOKEN_SIZE" # Needed for EmbeddingFunc wrapper
                ]
                if not all(key in self.config for key in required_keys):
                    missing = [key for key in required_keys if key not in self.config]
                    logger.error(f"Configuration missing required keys: {missing}")
                    self.config = None
                    return False

                # Resolve output file paths
                script_dir_parent = Path(__file__).parent.parent
                self.config["OUTPUT_RESULTS_FILE"] = str(script_dir_parent / self.config["OUTPUT_RESULTS_FILE"])
                self.config["OUTPUT_ERRORS_FILE"] = str(script_dir_parent / self.config["OUTPUT_ERRORS_FILE"])
                # Ensure output directories exist
                os.makedirs(Path(self.config["OUTPUT_RESULTS_FILE"]).parent, exist_ok=True)
                os.makedirs(Path(self.config["OUTPUT_ERRORS_FILE"]).parent, exist_ok=True)
                logger.info(f"Resolved OUTPUT_RESULTS_FILE: {self.config['OUTPUT_RESULTS_FILE']}")
                logger.info(f"Resolved OUTPUT_ERRORS_FILE: {self.config['OUTPUT_ERRORS_FILE']}")

                return True
            else:
                logger.error("'RAG_QUERY' section not found in YAML or config is empty.")
                return False
        except Exception as e:
            logger.error(f"Failed to load or process configuration: {e}", exc_info=True)
            return False

    def _setup_async_funcs(self):
        """Sets up the async LLM and embedding functions based on config."""
        # This code is very similar to Step 1's setup. Refactor to utils recommended.
        if not self.config:
            logger.error("Cannot setup async functions without configuration.")
            return False

        # Define LLM function
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
            logger.debug(f"Calling LLM: {self.config['LLM_MODEL_NAME']}")
            return await openai_complete( # Use appropriate lightrag function
                model=self.config['LLM_MODEL_NAME'],
                prompt=prompt,
                system_prompt=system_prompt,
                api_key=self.config['LLM_API_KEY'],
                base_url=self.config['LLM_BASE_URL'],
                **kwargs,
            )
        self.llm_model_func = llm_model_func

        # Define Embedding function
        async def embedding_func(texts: list[str]) -> np.ndarray:
            logger.debug(f"Calling Embedding: {self.config['EMBEDDING_MODEL_NAME']} for {len(texts)} texts")
            return await openai_embed(
                texts=texts,
                model=self.config['EMBEDDING_MODEL_NAME'],
                api_key=self.config['EMBEDDING_API_KEY'],
                base_url=self.config['EMBEDDING_BASE_URL'],
            )

        # Wrap embedding func
        self.embedding_func_wrapper = EmbeddingFunc(
            embedding_dim=self.config['EMBEDDING_DIM'],
            max_token_size=self.config['EMBEDDING_MAX_TOKEN_SIZE'],
            func=embedding_func
        )
        logger.info("Async LLM and Embedding functions configured.")
        return True

    async def _initialize_rag_instance(self, rag_working_dir: str):
        """Initializes LightRAG using the *provided* working directory."""
        if not self.config or not self.llm_model_func or not self.embedding_func_wrapper:
            logger.error("Cannot initialize RAG: Missing config or async functions.")
            return False
        try:
            logger.info(f"Initializing LightRAG instance using existing directory: {rag_working_dir}...")
            # CRITICAL: Pass the rag_index_dir received from input as the working_dir
            self.rag_instance = LightRAG(
                working_dir=rag_working_dir,
                llm_model_func=self.llm_model_func,
                embedding_func=self.embedding_func_wrapper,
                # Other params like batch_num might be needed if RAG re-reads config internally
                # embedding_batch_num=self.config.get('EMBEDDING_BATCH_NUM', 10) # Example if needed
            )
            # We might not need to call initialize_storages() when loading an existing index,
            # but it's often safe/idempotent. Check LightRAG docs for loading behavior.
            logger.info("Calling initialize_storages() for existing directory (check if needed)...")
            await self.rag_instance.initialize_storages()
            logger.info("LightRAG instance pointing to existing index is ready.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG with existing directory {rag_working_dir}: {e}", exc_info=True)
            self.rag_instance = None
            return False

    async def _process_single_query(self, query_text: str, query_param: QueryParam):
        """Processes a single query using rag.aquery and handles exceptions."""
        if not self.rag_instance:
            return None, {"query": query_text, "error": "RAG instance not initialized"}

        max_retries = self.config.get("QUERY_MAX_RETRIES", 1) # Default to 1 attempt
        retry_delay = self.config.get("QUERY_RETRY_DELAY", 5)
        retries = 0

        while retries < max_retries:
            try:
                logger.debug(f"Querying RAG for: '{query_text[:100]}...' (Attempt {retries + 1})")
                # Ensure aquery is awaited
                start_time = time.time()
                result = await self.rag_instance.aquery(query_text, param=query_param)
                end_time = time.time()
                logger.debug(f"Query successful in {end_time - start_time:.2f} seconds.")
                # Assume result is serializable, might need conversion if it's a custom object
                # Example: Convert result object to dict if necessary
                # result_data = result.to_dict() if hasattr(result, 'to_dict') else result
                return {"query": query_text, "result": result}, None # Success
            except Exception as e:
                retries += 1
                logger.warning(f"Error querying RAG for '{query_text[:50]}...' (Attempt {retries}/{max_retries}): {e}")
                if retries < max_retries:
                    logger.info(f"Retrying query in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Query failed after {max_retries} attempts for: '{query_text[:50]}...'")
                    return None, {"query": query_text, "error": str(e)} # Failure after retries
        # Should not be reached if max_retries >= 1
        return None, {"query": query_text, "error": "Query failed after retries"}


    async def _run_queries_and_save_task(self, questions_file: str, rag_dir: str):
        """Main async task: initialize RAG, load questions, query, save results."""
        if not await self._initialize_rag_instance(rag_dir):
            return None, None # RAG init failed

        queries = extract_queries(questions_file)
        if not queries:
            logger.warning("No queries extracted. Nothing to process.")
            # Return empty file paths? Or signal no work done? Let's return None.
            return None, None

        query_mode = self.config.get("QUERY_MODE", "hybrid")
        query_param = QueryParam(mode=query_mode)
        logger.info(f"Configured query mode: {query_mode}")

        results_list = []
        errors_list = []

        # Process queries concurrently or sequentially? Original script implies sequential.
        # Let's run sequentially for simplicity, use asyncio.gather for concurrency if needed.
        total_queries = len(queries)
        for i, query_text in enumerate(queries):
            logger.info(f"Processing query {i+1}/{total_queries}...")
            result_data, error_data = await self._process_single_query(query_text, query_param)
            if result_data:
                results_list.append(result_data)
            if error_data:
                errors_list.append(error_data)

        # Save results
        results_file_path = self.config["OUTPUT_RESULTS_FILE"]
        errors_file_path = self.config["OUTPUT_ERRORS_FILE"]

        logger.info(f"Saving {len(results_list)} results to {results_file_path}")
        try:
            with open(results_file_path, "w", encoding="utf-8") as f:
                # Save as JSON array, matching original output structure
                json.dump(results_list, f, ensure_ascii=False, indent=4)
        except TypeError as e:
             logger.error(f"Serialization error writing results: {e}. Result objects might not be JSON serializable.")
             # Try saving with default=str as a fallback, but investigate the object structure
             try:
                 with open(results_file_path, "w", encoding="utf-8") as f:
                     json.dump(results_list, f, ensure_ascii=False, indent=4, default=str)
                 logger.warning("Results saved with string conversion fallback due to serialization error.")
             except Exception as ioe:
                 logger.error(f"Failed to write results JSON even with fallback: {ioe}")
                 results_file_path = None # Indicate write failure
        except Exception as e:
            logger.error(f"Failed to write results JSON: {e}")
            results_file_path = None # Indicate write failure


        logger.info(f"Saving {len(errors_list)} errors to {errors_file_path}")
        try:
            with open(errors_file_path, "w", encoding="utf-8") as f:
                # Save errors as a JSON array, one object per line might be better for logs
                 json.dump(errors_list, f, ensure_ascii=False, indent=4) # Or one error object per line
                # for error_entry in errors_list:
                #     json.dump(error_entry, f, ensure_ascii=False)
                #     f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write errors JSON: {e}")
            errors_file_path = None # Indicate write failure

        return results_file_path, errors_file_path


    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        """Handles DORA input events to trigger RAG querying."""
        if dora_event["type"] == "INPUT":
            event_id = dora_event["id"]
            logger.debug(f"Received input event on ID: {event_id}")

            # Store inputs until both are received
            input_data = dora_event["value"][0].as_py()
            payload = load_node_result(input_data)

            if event_id == "generated_questions_file":
                if os.path.isfile(payload):
                    self.questions_file_path = payload
                    logger.info(f"Received questions file path: {self.questions_file_path}")
                else:
                    logger.error(f"Received invalid path for questions file: {payload}")
                    # Decide how to handle: wait or error out? Let's wait for now.

            elif event_id == "rag_index_dir":
                if os.path.isdir(payload):
                    self.rag_index_dir = payload
                    logger.info(f"Received RAG index directory path: {self.rag_index_dir}")
                else:
                    logger.error(f"Received invalid path for RAG index directory: {payload}")
                    # Decide how to handle. Let's wait for now.
            else:
                 logger.debug(f"Ignoring input from unexpected ID: {event_id}")
                 return DoraStatus.CONTINUE


            # Check if both inputs are ready and not processed yet
            if self.questions_file_path and self.rag_index_dir and not self.processed:
                logger.info("Both inputs received. Starting RAG query process...")

                # Load config
                if not self.config:
                    if not self.load_config():
                        logger.error("Failed to load configuration. Stopping.")
                        return DoraStatus.STOP

                # Setup async funcs
                if not self.llm_model_func or not self.embedding_func_wrapper:
                    if not self._setup_async_funcs():
                         logger.error("Failed to setup async functions. Stopping.")
                         return DoraStatus.STOP

                # --- Run the Main Async Task ---
                async def run_main_query_task():
                    return await self._run_queries_and_save_task(
                        self.questions_file_path,
                        self.rag_index_dir
                    )

                # Execute async task
                loop = asyncio.get_event_loop()
                results_path, errors_path = None, None
                try:
                    results_path, errors_path = loop.run_until_complete(run_main_query_task())
                except RuntimeError as e:
                     logger.warning(f"Asyncio loop issue: {e}. Check DORA context.")
                     # Fallback might not work

                # --- Handle Result ---
                # Check if paths were returned (indicating success)
                if results_path is not None and errors_path is not None:
                    logger.info("RAG querying and saving completed.")
                    # Send output paths
                    output_payload = {
                        "results_file": results_path,
                        "errors_file": errors_path
                    }
                    output_data = create_agent_output(
                        agent_name="rag_query_evaluator",
                        # Send both paths, maybe as a dict
                        agent_result=output_payload,
                        # This is likely the end of this specific pipeline
                        dataflow_status=True
                    )
                    send_output(
                        "query_results", # Single output ID carrying both paths
                        pa.array([output_data]),
                        dora_event['metadata']
                    )
                    logger.info(f"Sent output file paths: Results='{results_path}', Errors='{errors_path}'")
                else:
                    logger.error("RAG querying task failed or did not return valid output paths.")
                    self.processed = True # Mark as processed (failed)
                    return DoraStatus.STOP # Stop due to failure

                self.processed = True # Mark processing complete
                logger.info("RagQueryOperator processing finished.")
                return DoraStatus.STOP # Stop after completion

            elif self.processed:
                 logger.info("RAG querying already processed. Ignoring further inputs.")
            else:
                 logger.info("Waiting for all required inputs (questions file path and RAG index dir)...")


        elif dora_event["type"] == "STOP":
            logger.info("Received STOP event. Shutting down RagQueryOperator.")
        elif dora_event["type"] == "ERROR":
             logger.error(f"Received ERROR event: {dora_event['error']}")
        else:
            logger.debug(f"Received unknown event type: {dora_event['type']}")

        return DoraStatus.CONTINUE