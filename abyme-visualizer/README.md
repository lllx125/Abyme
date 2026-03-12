# Abyme Visualizer

Real-time chat interface with dynamic tree visualization for the Abyme recursive model.

## Features

- **ChatGPT-style interface** - Enter prompts and see responses in a familiar chat UI
- **Real-time tree growth** - Watch the recursive generation tree build dynamically
- **Color-coded nodes**:
  - 🟡 **Yellow** - Currently generating
  - 🔵 **Blue** - Generation complete (not final)
  - 🟢 **Green** - Final node (leaf)
- **Interactive features**:
  - Click any node to view full details in sidebar
  - Hover over nodes for quick preview
  - Zoom & pan to navigate large trees
  - Stop button to halt generation
- **Smart tree layout**:
  - First child directly below parent
  - Siblings positioned to avoid overlap
  - Continuation arrows show sequential flow
- **Markdown rendering** - Formatted text in node details

## Quick Start (Easiest Way)

From anywhere in the Abyme project:

```bash
./visualize.sh
```

Or from the visualizer directory:

```bash
./run
```

That's it! The script will automatically:
- Set up the virtual environment (first time only)
- Install dependencies (first time only)
- Start the server

Then open **http://localhost:5000** in your browser.

## Manual Installation (If Needed)

### 1. Install Python dependencies

```bash
cd /home/lilixing/Abyme/abyme-visualizer
pip install -r requirements.txt
```

### 2. Ensure Abyme is installed

The visualizer uses the `abyme-rllm` package from the parent directory. Make sure you have:
- DeepSeek API key in your environment (`.env` file or `DEEPSEEK_API_KEY` variable)

### 3. Run the server

```bash
source venv/bin/activate
python app.py
```

The server will start on `http://localhost:5000`

## Usage

### On WSL (Windows)

1. Run the launcher:
   ```bash
   ./run
   ```

2. Open your Windows browser and navigate to:
   ```
   http://localhost:5000
   ```

### Sending Prompts

1. Type your problem or question in the text area at the bottom left
2. Click **Send** or press **Ctrl+Enter** (Windows) / **Cmd+Enter** (Mac)
3. Watch the tree grow in real-time on the right panel
4. The final answer will appear in the chat

### Interacting with the Tree

- **Hover** over any node to see its details (prompt, context, output, latency)
- **Scroll** to zoom in/out
- **Click and drag** to pan around the tree
- Nodes change color as they progress:
  - Yellow → Red → Green (for leaf nodes)

## Configuration

You can adjust the model parameters by editing [main.js](static/js/main.js#L128):

```javascript
config: {
    max_depth: 5,           // Maximum recursion depth
    max_call: 100,          // Maximum total API calls
    max_parallel_workers: 4 // Parallel subproblem processing
}
```

## Project Structure

```
abyme-visualizer/
├── app.py                  # Flask server with SocketIO
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Main UI
└── static/
    ├── css/
    │   └── style.css      # Custom styles & animations
    └── js/
        ├── main.js        # WebSocket coordinator
        ├── chat.js        # Chat UI component
        └── tree.js        # D3.js tree visualization
```

## How It Works

1. **User sends prompt** → WebSocket message to Flask server
2. **Server creates Abyme_DeepSeek model** with event wrapping
3. **As generation happens**, server emits events:
   - `node_start` - New node created (yellow)
   - `node_complete` - Node generated (red)
   - `subproblem_created` - Edge added (parent → child)
   - `continuation_created` - Edge added (node → next)
   - `node_final` - Leaf node marked (green)
   - `generation_done` - Final answer ready
4. **Frontend receives events** → Updates tree visualization in real-time
5. **Final answer displayed** in chat

## Troubleshooting

### Port already in use
If port 5000 is already in use, edit [app.py](app.py#L174):
```python
socketio.run(app, host='0.0.0.0', port=5001, debug=True)
```

### Connection issues on WSL
- Ensure Windows firewall allows the connection
- Try accessing via `http://127.0.0.1:5000` instead of `localhost`

### DeepSeek API errors
- Check that `DEEPSEEK_API_KEY` is set in your environment
- Verify your API key is valid and has credits

### Tree not updating
- Check browser console for JavaScript errors
- Verify WebSocket connection status (green dot in bottom left)
- Ensure Flask server is running without errors

## Example Prompts

Try these to see the tree visualization:

**Math problem:**
```
Find the derivative of f(x) = (x^2 + 1)^(x^3) using logarithmic differentiation
```

**Logic puzzle:**
```
If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?
```

**Complex reasoning:**
```
Can a square be partitioned into an odd number of triangles of equal area?
```

## Development

### Adding new features

The code is modular:
- **Backend events**: Modify [app.py](app.py) `create_model_with_events()`
- **Tree layout**: Edit [tree.js](static/js/tree.js) `calculatePositions()`
- **UI styling**: Update [style.css](static/css/style.css)

### Debugging

Enable debug mode in the browser console:
```javascript
socket.on('node_start', (data) => {
    console.log('Node started:', data);
});
```

Server logs show all events being emitted.

## License

Part of the Abyme project.
