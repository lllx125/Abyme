# Abyme Tree Visualizer

Real-time visualization of Abyme recursive model generation trees with interactive D3.js interface.

## Features

- **Real-time Tree Visualization**: Live updates as the model generates (throttled to 10 FPS)
- **Interactive Controls**: Adjust max_depth, max_call, max_chain_length, and max_parallel_workers
- **Node States**: Visual representation of all node statuses with color coding
- **Edge Types**: Distinct styles for AND (pink), OR (lime), and PAST (white arrow) edges
- **Hover Tooltips**: Preview prompt and output on node hover
- **Detail Sidebar**: Click nodes to view full prompt and output in markdown format
- **Stop Capability**: Gracefully cancel generation mid-flight
- **Console Logging**: All events logged to browser console for debugging

## Quick Start (One Command!)

From the Abyme root directory:

```bash
./visualize.sh
```

This script will:
- Create a virtual environment (first run only)
- Install all dependencies automatically
- Install abyme-rllm in development mode
- Start the Flask server on `http://localhost:5000`

Then just open your browser to `http://localhost:5000`!

## Manual Installation (Optional)

If you prefer to set up manually:

### 1. Install Python Dependencies

```bash
cd /home/lilixing/Abyme/abyme-visualizer
pip install -r requirements.txt
```

### 2. Install Abyme

```bash
cd /home/lilixing/Abyme/abyme-rllm
pip install -e .
```

### 3. Start the Server

```bash
cd /home/lilixing/Abyme/abyme-visualizer
python app.py
```

The server will start on `http://localhost:5000`

### 4. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

### 3. Using the Interface

#### Left Control Panel:
- **Prompt**: Enter your problem/question
- **Model**: Select model (deepseek, gpt, or deepseek-r)
- **Max Depth**: Maximum recursion depth (1-20)
- **Max Calls**: Maximum total API calls (100-5000)
- **Max Chain Length**: Maximum continuation chain (1-10)
- **Max Workers**: Parallel worker threads (1-50)

#### Tree Visualization:
- **Pan**: Click and drag to move around
- **Zoom**: Scroll to zoom in/out
- **Hover**: See node preview tooltip
- **Click**: Open detail sidebar with full content

#### Node Status Colors:
- **Grey**: WAIT_GEN (waiting to generate)
- **Yellow (pulsing)**: GENERATING (actively generating)
- **Orange-red**: WAIT_SUB (waiting for subproblems)
- **Blue**: COMPLETED (generation completed)
- **Green**: FINAL (final state, no more work)
- **Red**: FAILED (generation failed)
- **Dark grey**: CANCELLED (cancelled by OR node)

#### Edge Types:
- **Pink (glowing)**: AND edges (all children required)
- **Lime (glowing)**: OR edges (first successful child)
- **White arrow (glowing)**: PAST edges (temporal continuation)

## Architecture

### Backend (Python)
- **app.py**: Flask application entry point
- **config.py**: Configuration settings
- **visualizer/tree_manager_events.py**: VisualizerTreeManager (emits events)
- **visualizer/event_emitter.py**: ThrottledEventEmitter (100ms throttle)
- **visualizer/tree_serializer.py**: Thread-safe tree serialization
- **visualizer/model_wrapper.py**: StoppableRecursiveModel (graceful stop)
- **routes/main.py**: Main page route
- **routes/websocket.py**: WebSocket event handlers

### Frontend (JavaScript ES6 Modules)
- **main.js**: Application coordinator
- **socket_client.js**: Socket.IO wrapper
- **tree_renderer.js**: D3.js visualization
- **controls.js**: Input/slider/button handlers
- **sidebar.js**: Detail sidebar with markdown
- **metrics.js**: Call count display
- **logger.js**: Console logging utility

### Styles (CSS)
- **style.css**: Global styles and layout
- **controls.css**: Control panel styles
- **tree.css**: Tree canvas and node/edge styles
- **sidebar.css**: Detail sidebar and markdown styles

## WebSocket Events

### Client → Server
- `generate_request`: Start generation with parameters
- `stop_generation`: Stop current generation

### Server → Client
- `tree_update`: Tree data + metrics (throttled to 100ms)
- `generation_started`: Generation has begun
- `generation_complete`: Generation finished successfully
- `generation_stopped`: Generation was stopped by user
- `generation_error`: Error occurred during generation

## Thread Safety

The visualizer implements several thread-safety mechanisms:

1. **Lock-Protected Serialization**: `draw_tree()` calls hold `manager.lock` to prevent concurrent modifications
2. **Event Throttling**: Updates coalesced to max 10/second to avoid overwhelming WebSocket
3. **Cooperative Cancellation**: Workers check `stop_requested` Event at every iteration
4. **Full Snapshots**: Send complete tree state (not deltas) to avoid sync issues

## Development

### Running in Debug Mode

```bash
export FLASK_ENV=development
python app.py
```

### Viewing Console Logs

Open browser DevTools (F12) and check the Console tab. All events are logged with timestamps and color-coded severity levels.

### Modifying Throttle Interval

Edit `config.py`:

```python
TREE_UPDATE_THROTTLE_INTERVAL = 0.1  # 100ms = ~10 FPS
```

## Troubleshooting

### Port Already in Use

If port 5000 is already in use, modify [app.py:56](app.py:56):

```python
socketio.run(app, host='0.0.0.0', port=5001)
```

### WebSocket Connection Failed

Check that:
1. Flask server is running
2. No firewall blocking port 5000
3. Browser console shows connection attempt

### Tree Not Updating

Check browser console for:
1. WebSocket connection status
2. `tree_update` events being received
3. Any JavaScript errors

### Stop Button Not Working

Verify in console that:
1. `stop_generation` event is emitted
2. Backend logs show stop signal received
3. Workers are exiting gracefully

## License

MIT License - See parent Abyme project for details.

## Credits

Built with:
- Flask + Flask-SocketIO (backend)
- D3.js v7 (visualization)
- Socket.IO (WebSocket)
- Marked.js + DOMPurify (markdown rendering)
