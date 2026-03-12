import sys
import os

# Add abyme-rllm to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'abyme-rllm'))

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import threading
import uuid
from abyme.core import Abyme_DeepSeek
from abyme.tree_trace import total_calls, max_depth as get_max_depth, parallel_latency

app = Flask(__name__)
app.config['SECRET_KEY'] = 'abyme-visualizer-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Thread-safe global node ID tracking
node_id_map = {}  # Maps id(node) → uuid
parent_map = {}   # Maps id(node) → id(parent_node)
map_lock = threading.Lock()  # Thread safety for multi-threading

# Session-based generation tracking
active_sessions = {}  # Maps session_id → {'stopped': bool, 'thread': Thread}
sessions_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    print(f'Client connected: {session_id}')
    with sessions_lock:
        active_sessions[session_id] = {'stopped': False, 'thread': None}
    emit('connected', {'status': 'ready'})


@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    print(f'Client disconnected: {session_id}')

    # Mark this session's generation as stopped
    with sessions_lock:
        if session_id in active_sessions:
            active_sessions[session_id]['stopped'] = True
            print(f'  → Stopping generation for session {session_id}')
            # Clean up after a delay to allow thread to stop gracefully
            def cleanup():
                threading.Timer(5.0, lambda: active_sessions.pop(session_id, None)).start()
            cleanup()


@socketio.on('stop_generation')
def handle_stop():
    session_id = request.sid
    print(f'Stop generation requested by {session_id}')

    with sessions_lock:
        if session_id in active_sessions:
            active_sessions[session_id]['stopped'] = True

    emit('generation_stopped', {'message': 'Stop requested'})


@socketio.on('generate')
def handle_generate(data):
    session_id = request.sid
    prompt = data.get('prompt', '')
    config = data.get('config', {})

    print(f'\n{"="*60}')
    print(f'🚀 Starting generation: {prompt[:50]}... (session: {session_id})')
    print(f'{"="*60}\n')

    # Clear global tracking (thread-safe)
    with map_lock:
        node_id_map.clear()
        parent_map.clear()

    # Mark session as active and not stopped
    with sessions_lock:
        if session_id in active_sessions:
            active_sessions[session_id]['stopped'] = False

    # Run generation in background thread
    def run_generation():
        try:
            # Create model with event wrapping (pass session_id)
            model = create_model_with_events(config, session_id)

            # Generate
            result = model.generate(prompt, max_attempt=1)

            # Check if stopped before emitting completion
            with sessions_lock:
                if session_id in active_sessions and active_sessions[session_id]['stopped']:
                    print(f'\n⏹️  Generation stopped by user (session: {session_id})\n')
                    return

            print(f'\n{"="*60}')
            print(f'✅ Generation complete! (session: {session_id})')
            print(f'{"="*60}\n')

            # Send completion event
            socketio.emit('generation_done', {
                'final_output': result,
                'stats': {
                    'total_calls': total_calls(model.trace),
                    'max_depth': get_max_depth(model.trace),
                    'latency': parallel_latency(model.trace)
                }
            })
        except Exception as e:
            # Check if this was a user-initiated stop
            with sessions_lock:
                if session_id in active_sessions and active_sessions[session_id]['stopped']:
                    print(f'\n⏹️  Generation stopped (session: {session_id})\n')
                    return

            print(f'\n❌ Error during generation: {e}\n')
            socketio.emit('error', {'message': str(e)})

    thread = threading.Thread(target=run_generation)
    thread.daemon = True
    thread.start()

    # Store thread reference
    with sessions_lock:
        if session_id in active_sessions:
            active_sessions[session_id]['thread'] = thread


def create_model_with_events(config, session_id):
    """Create Abyme_DeepSeek model with event emission wrapping."""
    max_workers = config.get('max_parallel_workers', 10)  # Default to 10 workers

    model = Abyme_DeepSeek(
        reasoning=config.get('reasoning', False),
        max_depth=config.get('max_depth', 5),
        max_call=config.get('max_call', 100),
        max_parallel_workers=max_workers,
        print_progress=False  # We'll handle progress logging ourselves
    )

    print(f'📊 Model config: max_depth={model.max_depth}, max_call={model.max_call}, workers={max_workers}')

    # Wrap _recursive_generate to emit events
    original_recursive_generate = model._recursive_generate

    def wrapped_recursive_generate(node):
        # Check if this session has been stopped
        with sessions_lock:
            if session_id in active_sessions and active_sessions[session_id]['stopped']:
                raise Exception("Generation stopped by user")
        # Generate UUID for this node (thread-safe)
        node_id = str(uuid.uuid4())
        with map_lock:
            node_id_map[id(node)] = node_id
            parent_id = parent_map.get(id(node))

            # Check if this node is a continuation (someone's next)
            # Look through all tracked nodes to find if any has this node as next
            continuation_parent_id = None
            for tracked_node_id, tracked_node in [(nid, n) for nid, n in node_id_map.items()]:
                # We need to get the actual node object to check its next pointer
                # This is tricky because we only have node IDs...
                # Better approach: track continuation relationships explicitly
                pass

        # Progress logging
        prompt_preview = node.prompt[:60] + '...' if len(node.prompt) > 60 else node.prompt
        print(f'🔄 [Depth {node.depth}] Generating: {prompt_preview}')

        # Emit node_start event
        socketio.emit('node_start', {
            'node_id': node_id,
            'parent_id': parent_id,
            'prompt': node.prompt,
            'context': node.context,
            'depth': node.depth
        })

        # If this node has a parent, emit the edge immediately (subproblem edge)
        if parent_id:
            socketio.emit('subproblem_created', {
                'parent_id': parent_id,
                'child_id': node_id
            })

        # Call original _recursive_generate
        result = original_recursive_generate(node)

        # After generation, emit continuation edge immediately if exists (thread-safe)
        with map_lock:
            if node.next:
                next_id = node_id_map.get(id(node.next))
                if next_id:
                    socketio.emit('continuation_created', {
                        'node_id': node_id,
                        'next_id': next_id
                    })

        # Progress logging
        output_preview = node.output[:60] + '...' if len(node.output) > 60 else node.output
        print(f'✓ [Depth {node.depth}] Complete ({node.latency:.2f}s): {output_preview}')
        print(f'  Call #{model.call_count}/{model.max_call}')

        # Emit node_complete event
        socketio.emit('node_complete', {
            'node_id': node_id,
            'output': node.output,
            'latency': node.latency
        })

        # Mark as final if no continuation (thread-safe)
        with map_lock:
            if not node.next:
                # This is a final node (leaf in the chain)
                print(f'🏁 [Depth {node.depth}] Final node reached')
                socketio.emit('node_final', {
                    'node_id': node_id
                })

        return result

    # Implement wrapped subproblem generation methods with event emission

    def wrapped_dfs_generate(subproblems, parent_node):
        # Emit waiting event - parent has output and is waiting for children
        with map_lock:
            parent_node_id = node_id_map.get(id(parent_node))

        if parent_node_id and hasattr(parent_node, 'output') and parent_node.output:
            socketio.emit('node_waiting', {
                'node_id': parent_node_id,
                'output': parent_node.output,
                'num_subproblems': len(subproblems)
            })
            print(f'⏳ [Depth {parent_node.depth}] Waiting for {len(subproblems)} subproblems')

        responses = []
        for sub in subproblems:
            from abyme.tree_trace import TreeTraceNode
            newnode = TreeTraceNode(prompt=sub, context="", depth=parent_node.depth + 1)

            # Track parent relationship (thread-safe)
            with map_lock:
                parent_map[id(newnode)] = node_id_map.get(id(parent_node))

            # Add child to parent BEFORE recursive generation (matches original model.py behavior)
            parent_node.add_subproblem(newnode)

            for i in range(model.max_subproblem_retry):
                try:
                    model._recursive_generate(newnode)
                    break
                except Exception as e:
                    if i == model.max_subproblem_retry - 1:
                        raise Exception(f"Sub-problem generation failed after {model.max_subproblem_retry} attempts: {e}")
                    continue

            responses.append(newnode.get_final_output())
        return responses

    # Wrap parallel version
    def wrapped_parallel_generate(subproblems, parent_node):
        # Emit waiting event - parent has output and is waiting for children
        with map_lock:
            parent_node_id = node_id_map.get(id(parent_node))

        if parent_node_id and hasattr(parent_node, 'output') and parent_node.output:
            socketio.emit('node_waiting', {
                'node_id': parent_node_id,
                'output': parent_node.output,
                'num_subproblems': len(subproblems)
            })
            print(f'⏳ [Depth {parent_node.depth}] Waiting for {len(subproblems)} subproblems (parallel)')

        import asyncio
        import concurrent.futures
        from abyme.tree_trace import TreeTraceNode

        async def process_subproblem(sub: str, executor_ref):
            loop = asyncio.get_running_loop()
            newnode = TreeTraceNode(prompt=sub, context="", depth=parent_node.depth + 1)

            # Track parent relationship (thread-safe)
            with map_lock:
                parent_map[id(newnode)] = node_id_map.get(id(parent_node))

            last_error = None
            for attempt in range(model.max_subproblem_retry):
                try:
                    await loop.run_in_executor(
                        executor_ref,
                        model._recursive_generate,
                        newnode
                    )
                    break
                except Exception as e:
                    last_error = e
                    if attempt == model.max_subproblem_retry - 1:
                        raise Exception(f"Sub-problem generation failed after {model.max_subproblem_retry} attempts: {e}")
                    continue

            response = newnode.get_final_output()
            return newnode, response

        async def process_all_subproblems(executor_ref):
            tasks = [process_subproblem(sub, executor_ref) for sub in subproblems]
            results = await asyncio.gather(*tasks)

            responses = []
            for newnode, response in results:
                parent_node.add_subproblem(newnode)
                responses.append(response)

            return responses

        with concurrent.futures.ThreadPoolExecutor(max_workers=model.max_parallel_workers) as executor:
            try:
                responses = asyncio.run(process_all_subproblems(executor))
                return responses
            except Exception as e:
                raise Exception(f"Error in parallel sub-problem generation: {e}")

    model._recursive_generate = wrapped_recursive_generate
    model._dfs_sequential_subproblem_generate = wrapped_dfs_generate
    model._parallel_subproblem_generate = wrapped_parallel_generate

    return model


if __name__ == '__main__':
    print('Starting Abyme Visualizer on http://localhost:5000')
    print('Access from Windows browser at: http://localhost:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
