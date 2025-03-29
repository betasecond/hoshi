import os
import json
import glob
import logging
import time
from pathlib import Path

import pyarrow as pa
from dora import DoraStatus
from huggingface_hub import snapshot_download

try:
    from mofa.utils.files.read import read_yaml
    from mofa.utils.files.dir import get_relative_path
    from mofa.kernel.utils.util import create_agent_output # 用于标准化输出
except ImportError:
    # 提供一个简单的备用方案，如果 mofa utils 不可用
    import yaml
    def get_relative_path(current_file, sibling_directory_name, target_file_name):
        base_dir = Path(current_file).parent
        return base_dir / sibling_directory_name / target_file_name

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
        return {
            "agent_name": agent_name,
            "agent_result": agent_result,
            "dataflow_status": dataflow_status
        }


def download_dataset(repo_id, local_dir, max_retries=3, delay=5):
    logging.info(f"Attempting to download dataset '{repo_id}' to '{local_dir}'...")
    retries = 0
    while retries < max_retries:
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False 
            )
            logging.info(f"Dataset successfully downloaded to: {local_dir}")
            return True
        except Exception as e:
            retries += 1
            logging.warning(f"Error downloading dataset (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max download retries reached.")
                if os.path.exists(local_dir) and os.listdir(local_dir):
                     logging.warning(f"Directory '{local_dir}' already exists and is not empty. Assuming dataset is present despite download errors.")
                     return True
                return False
    return False

def extract_unique_contexts(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_directory}")

    jsonl_files = glob.glob(os.path.join(input_directory, "*.jsonl"))
    logging.info(f"Found {len(jsonl_files)} JSONL files in {input_directory}.")

    if not jsonl_files:
        logging.warning(f"No JSONL files found in {input_directory}. Cannot extract contexts.")

    all_contexts_extracted = True
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        # 清理文件名，移除可能导致问题的字符，或创建一个更简单的映射
        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
        output_filename = f"{safe_name}_unique_contexts.json"
        output_path = os.path.join(output_directory, output_filename)

        unique_contexts_dict = {}
        logging.info(f"Processing file: {filename}")

        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        context = json_obj.get("context")
                        if isinstance(context, str) and context and context not in unique_contexts_dict:
                            unique_contexts_dict[context] = None
                        elif context is None:
                            logging.debug(f"Line {line_number} in {filename} has null context.")
                        elif not isinstance(context, str):
                             logging.warning(f"Line {line_number} in {filename} has non-string context: {type(context)}. Skipping.")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decoding error in file {filename} at line {line_number}: {e}")
                    except Exception as e
                        logging.error(f"Error processing line {line_number} in file {filename}: {e}")

        except FileNotFoundError:
            logging.error(f"File not found during processing: {filename}")
            all_contexts_extracted = False
            continue
        except Exception as e:
            logging.error(f"An error occurred while processing file {filename}: {e}")
            all_contexts_extracted = False
            continue

        unique_contexts_list = list(unique_contexts_dict.keys())
        logging.info(f"Found {len(unique_contexts_list)} unique `context` entries in {filename}.")

        if not unique_contexts_list:
            logging.warning(f"No unique contexts extracted from {filename}. Skipping output file creation.")
            continue

        try:
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(unique_contexts_list, outfile, ensure_ascii=False, indent=4)
            logging.info(f"Unique contexts saved to: {output_path}")
        except Exception as e:
            logging.error(f"An error occurred while saving to {output_path}: {e}")
            all_contexts_extracted = False

    logging.info("Finished processing all files.")
    return all_contexts_extracted

# --- DORA Operator Class ---

class Operator:
    """
    DORA Operator for downloading the UltraDomain dataset and extracting unique contexts.
    """
    def __init__(self):
        """Initialize the operator, potentially loading static config."""
        self.config = None
        self.processed = False # Flag to ensure processing happens only once per trigger

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("DataPrepOperator initialized.")

    def load_config(self):
        """Loads configuration from a YAML file."""
        try:
            # Assumes config file is in a 'configs' sibling directory
            yaml_file_path = get_relative_path(__file__, 'configs', 'data_prep_config.yml')
            logging.info(f"Loading configuration from: {yaml_file_path}")
            config = read_yaml(yaml_file_path)
            if config and "DATA_PREP" in config:
                self.config = config["DATA_PREP"]
                logging.info("Configuration loaded successfully.")
                # Basic validation
                required_keys = ["DATASET_REPO_ID", "DOWNLOAD_DIR", "OUTPUT_DIR"]
                if not all(key in self.config for key in required_keys):
                    logging.error(f"Configuration missing required keys: {required_keys}")
                    self.config = None
                    return False
                # Resolve relative paths if needed (make them absolute)
                script_dir = Path(__file__).parent
                self.config["DOWNLOAD_DIR"] = str(script_dir / self.config["DOWNLOAD_DIR"])
                self.config["OUTPUT_DIR"] = str(script_dir / self.config["OUTPUT_DIR"])
                logging.info(f"Resolved DOWNLOAD_DIR: {self.config['DOWNLOAD_DIR']}")
                logging.info(f"Resolved OUTPUT_DIR: {self.config['OUTPUT_DIR']}")

                return True
            else:
                logging.error("'DATA_PREP' section not found in YAML or config is empty.")
                return False
        except Exception as e:
            logging.error(f"Failed to load or process configuration: {e}")
            return False

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        """
        Handles incoming DORA events. Expects a trigger input to start processing.
        """
        if dora_event["type"] == "INPUT":
            # Check if it's the trigger input (e.g., from a start node or previous step)
            # We use a simple trigger mechanism here: process on the first valid input event.
            # A more robust way might be to check dora_event["id"] == "trigger".
            if self.processed:
                logging.info("Data preparation already processed. Ignoring further inputs.")
                return DoraStatus.CONTINUE # Already done, just wait

            logging.info(f"Received input event on ID: {dora_event['id']}. Starting data preparation...")

            # Load configuration if not already loaded
            if not self.config:
                if not self.load_config():
                    logging.error("Failed to load configuration. Stopping.")
                    # Send an error status if possible, or just stop
                    # send_output("status", pa.array([create_agent_output("data_prep", "Config load failed", True)]), dora_event['metadata'])
                    return DoraStatus.STOP # Cannot proceed without config

            # --- Main Logic ---
            try:
                # Step 1: Download dataset (if not skipped)
                download_dir = self.config["DOWNLOAD_DIR"]
                output_dir = self.config["OUTPUT_DIR"]
                repo_id = self.config["DATASET_REPO_ID"]
                skip_download = self.config.get("SKIP_DOWNLOAD", False) # Default to False if not specified

                download_successful = False
                if not skip_download:
                    download_successful = download_dataset(repo_id, download_dir)
                else:
                    logging.info(f"Skipping download. Assuming dataset exists in '{download_dir}'.")
                    if os.path.isdir(download_dir) and os.listdir(download_dir):
                         logging.info(f"Directory '{download_dir}' exists and is not empty.")
                         download_successful = True
                    else:
                        logging.error(f"Error: Skipped download, but directory '{download_dir}' does not exist or is empty.")
                        download_successful = False

                # Step 2: Extract contexts if download/check was successful
                extraction_successful = False
                if download_successful:
                    logging.info("Proceeding with context extraction.")
                    extraction_successful = extract_unique_contexts(download_dir, output_dir)
                    if extraction_successful:
                        logging.info("Context extraction completed successfully.")
                    else:
                        logging.warning("Context extraction finished, but there might have been issues (e.g., no files found, save errors).")
                        # Decide if this is a fatal error. Let's proceed but log the warning.
                else:
                    logging.error("Dataset download/verification failed. Cannot proceed with context extraction.")
                    # Stop the node as it cannot produce the required output
                    # Optionally send an error message
                    # send_output("status", pa.array([create_agent_output("data_prep", "Download failed", True)]), dora_event['metadata'])
                    self.processed = True # Mark as processed (failed)
                    return DoraStatus.STOP

                # If extraction happened (even with warnings), send the output directory path
                if download_successful: # Check if we got to the extraction step at least
                     # Send the directory path where unique context files are stored
                     logging.info(f"Sending output directory path: {output_dir}")
                     # Use create_agent_output for standardized format if available
                     output_data = create_agent_output(
                         agent_name="data_prep",
                         agent_result=output_dir,
                         # Decide if this node marks the end of the *entire* dataflow (likely not)
                         dataflow_status=False
                     )
                     send_output(
                         "unique_contexts_dir", # Define a meaningful output ID
                         pa.array([output_data]),
                         dora_event['metadata']
                     )
                     logging.info("Output sent successfully.")
                else:
                     # Should have been caught earlier, but as a safeguard
                     logging.error("Processing failed before output could be generated.")


                self.processed = True # Mark processing as complete for this instance
                logging.info("Data preparation process finished.")
                # Decide if the node should stop after one run. Usually yes for batch jobs.
                # If it needs to wait for other potential triggers, use CONTINUE.
                return DoraStatus.STOP # Stop after successful completion


            except Exception as e:
                logging.exception(f"An unhandled error occurred during data preparation: {e}")
                # Optionally send an error message output
                # send_output("status", pa.array([create_agent_output("data_prep", f"Unhandled Error: {e}", True)]), dora_event['metadata'])
                self.processed = True # Mark as processed (failed)
                return DoraStatus.STOP # Stop on unhandled exception

        elif dora_event["type"] == "STOP":
            logging.info("Received STOP event. Shutting down DataPrepOperator.")

        elif dora_event["type"] == "ERROR":
             logging.error(f"Received ERROR event: {dora_event['error']}")

        else:
            logging.debug(f"Received unknown event type: {dora_event['type']}")

        return DoraStatus.CONTINUE # Continue unless explicitly stopped