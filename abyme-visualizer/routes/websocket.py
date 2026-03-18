"""
WebSocket event handlers for real-time tree visualization

Handles generation requests, stop signals, and tree update broadcasting.
"""

import sys
sys.path.append('/home/lilixing/Abyme/abyme-rllm')

import threading
from flask_socketio import emit, disconnect
from abyme.core import Abyme_API_Models
from visualizer.model_wrapper import StoppableRecursiveModel
from visualizer.event_emitter import ThrottledEventEmitter


# Global state for current generation (single-user mode)
current_model = None
current_emitter = None
generation_thread = None
generation_lock = threading.Lock()


def init_websocket_handlers(socketio):
    """
    Initialize WebSocket event handlers.

    Args:
        socketio: Flask-SocketIO instance
    """

    @socketio.on('connect', namespace='/visualizer')
    def handle_connect():
        """Handle client connection."""
        print(f"[INFO] Client connected")
        emit('connected', {'status': 'ready'})

    @socketio.on('disconnect', namespace='/visualizer')
    def handle_disconnect():
        """Handle client disconnection."""
        print(f"[INFO] Client disconnected")
        # Optional: Stop generation on disconnect
        # with generation_lock:
        #     if current_model:
        #         current_model.stop()

    @socketio.on('generate_request', namespace='/visualizer')
    def handle_generate_request(data):
        """
        Handle generation request from client.

        Args:
            data: {
                'prompt': str,
                'model': str ('deepseek', 'gpt', 'deepseek-r'),
                'max_depth': int,
                'max_call': int,
                'max_chain_length': int,
                'max_parallel_workers': int
            }
        """
        global current_model, current_emitter, generation_thread

        print(f"[INFO] Received generate_request: {data}")

        with generation_lock:
            # Check if generation is already running
            if generation_thread and generation_thread.is_alive():
                emit('generation_error', {
                    'error_message': 'Generation already in progress. Please stop it first.'
                })
                return

            try:
                # Extract parameters
                prompt = data.get('prompt', '')
                model_type = data.get('model', 'deepseek')
                max_depth = data.get('max_depth', 5)
                max_call = data.get('max_call', 3000)
                max_chain_length = data.get('max_chain_length', 5)
                max_parallel_workers = data.get('max_parallel_workers', 10)

                if not prompt:
                    emit('generation_error', {'error_message': 'Prompt cannot be empty'})
                    return

                # Create the base model using Abyme_API_Models factory
                base_model = Abyme_API_Models(
                    model=model_type,
                    max_depth=max_depth,
                    max_call=max_call,
                    max_parallel_workers=max_parallel_workers,
                    print_progress=True
                )

                # Emitter will be created in the callback once we have the manager
                current_emitter = None

                # Create event callback that creates emitter on first call
                def event_callback(event_type, node, root, call_count, manager):
                    """Callback invoked by VisualizerTreeManager on state changes."""
                    global current_emitter

                    # Create emitter on first call (when we have access to manager)
                    if current_emitter is None:
                        current_emitter = ThrottledEventEmitter(
                            socketio=socketio,
                            manager_lock=manager.lock,
                            throttle_interval=0.1
                        )
                        print("[INFO] Created ThrottledEventEmitter with manager lock")

                    # Emit the update
                    current_emitter.emit_tree_update(event_type, node, root, call_count)

                # Wrap the base model with StoppableRecursiveModel
                current_model = StoppableRecursiveModel(
                    base_model=base_model.base_model,
                    guard_model=base_model.guard_model,
                    formatter=base_model.formatter,
                    max_depth=max_depth,
                    max_call=max_call,
                    max_parallel_workers=max_parallel_workers,
                    max_subproblem_retry=2,
                    max_chain_length=max_chain_length,
                    proceed_when_fail=True,
                    print_progress=True,
                    event_callback=event_callback
                )

                # Start generation in background thread
                generation_thread = threading.Thread(
                    target=_run_generation,
                    args=(current_model, prompt, socketio),
                    daemon=True
                )
                generation_thread.start()

                emit('generation_started', {'status': 'started'})

            except Exception as e:
                print(f"[ERROR] Failed to start generation: {e}")
                import traceback
                traceback.print_exc()
                emit('generation_error', {'error_message': str(e)})

    @socketio.on('stop_generation', namespace='/visualizer')
    def handle_stop_generation(data):
        """Handle stop request from client."""
        global current_model

        print(f"[INFO] Received stop_generation request")

        with generation_lock:
            if current_model:
                current_model.stop()
                emit('generation_stopped', {'status': 'stopped'})
            else:
                emit('generation_error', {'error_message': 'No generation in progress'})


def _run_generation(model: StoppableRecursiveModel, prompt: str, socketio):
    """
    Run generation in background thread.

    Args:
        model: StoppableRecursiveModel instance
        prompt: User's input prompt
        socketio: Flask-SocketIO instance for emitting events
    """
    try:
        # Run the generation (blocking call)
        # The event_callback will be invoked with the manager reference
        result = model.generate(prompt, max_attempt=1)

        # Emit completion event
        socketio.emit('generation_complete', {
            'status': 'FINAL',
            'final_output': result
        }, namespace='/visualizer')

        print(f"[INFO] Generation completed successfully")

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Generation failed: {error_msg}")

        if "stopped by user" in error_msg:
            socketio.emit('generation_stopped', {
                'status': 'stopped',
                'message': 'Generation was stopped by user'
            }, namespace='/visualizer')
        else:
            socketio.emit('generation_error', {
                'error_message': error_msg
            }, namespace='/visualizer')

    finally:
        # Clean up
        global current_model
        with generation_lock:
            current_model = None
