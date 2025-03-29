# scripts/result_logger_operator.py

import pyarrow as pa
from dora import DoraStatus
import json
import logging
import os
from pathlib import Path # Used in fallback load_node_result if needed

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ResultLoggerOperator")

# --- Helper Function (Copied fallback version) ---
# Ideally, place this in a shared utils file (e.g., scripts/utils/payload_utils.py)
# and import it here and in other operators.
def load_node_result(input_data):
     """
     Extracts the core agent result from a potential wrapper structure.
     Handles both the standard dict structure and potentially raw data.
     """
     if isinstance(input_data, dict) and "agent_result" in input_data:
         logger.debug("Extracting result from agent output structure.")
         return input_data["agent_result"]
     logger.debug("Input data does not seem wrapped; returning as is.")
     return input_data # Return as is if format is unexpected or raw

# --- DORA Operator Class ---

class Operator:
    """
    DORA Operator that acts as a sink, logging the final results received
    from the RAG querying pipeline.
    """
    def __init__(self):
        """Initialize the result logger operator."""
        logger.info("ResultLoggerOperator initialized.")
        self.processed = False # Ensure it only processes the final result once

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        """Handles incoming DORA events, expecting the final pipeline output."""

        if dora_event["type"] == "INPUT":
            if self.processed:
                logger.info("Result already logged. Ignoring further inputs.")
                return DoraStatus.CONTINUE # Already done its job

            event_id = dora_event["id"]
            logger.info(f"Received final pipeline output on input ID: {event_id}")

            try:
                # Decode the PyArrow data
                # Assuming the input is always a pyarrow array of size 1
                data = dora_event["value"][0].as_py()

                # Extract the core payload using the helper
                final_payload = load_node_result(data)

                logger.info("---- Final Pipeline Payload Received ----")
                try:
                    # Pretty print the JSON payload if possible
                    logger.info(json.dumps(final_payload, indent=2, ensure_ascii=False))
                except TypeError:
                    # Fallback if payload is not JSON serializable
                    logger.info(f"Payload (non-JSON): {final_payload}")


                # Optional: Check if expected files exist based on payload structure
                if isinstance(final_payload, dict):
                    results_file = final_payload.get("results_file")
                    errors_file = final_payload.get("errors_file")

                    if results_file:
                        if os.path.exists(results_file):
                            logger.info(f"  ✅ Results file exists: {results_file}")
                        else:
                            logger.warning(f"  ❌ Results file NOT FOUND: {results_file}")
                    else:
                         logger.debug("  ℹ️ No 'results_file' key found in payload.")

                    if errors_file:
                        if os.path.exists(errors_file):
                            logger.info(f"  ✅ Errors file exists: {errors_file}")
                        else:
                            logger.warning(f"  ❌ Errors file NOT FOUND: {errors_file}")
                    else:
                         logger.debug("  ℹ️ No 'errors_file' key found in payload.")

                logger.info("---- End of LightRAG Pipeline Log ----")

                # Mark as processed and stop this node
                self.processed = True
                logger.info("Result logger finished processing. Stopping node.")
                return DoraStatus.STOP

            except Exception as e:
                logger.error(f"Error processing final output in logger node: {e}", exc_info=True)
                # Decide if the node should stop on error or keep running
                # Stopping might be safer if the input is crucial and processing failed.
                self.processed = True # Mark as processed (failed)
                return DoraStatus.STOP # Stop on error

        elif dora_event["type"] == "STOP":
            logger.info("ResultLoggerOperator received STOP event.")
            # Perform any cleanup if needed before stopping

        else:
            # Ignore other event types like 'START', 'CONTINUE' etc. if not needed
            logger.debug(f"Ignoring event type: {dora_event['type']}")

        # Keep the node running until it processes its input or receives STOP
        return DoraStatus.CONTINUE