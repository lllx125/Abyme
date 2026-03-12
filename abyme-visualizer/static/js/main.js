/**
 * Main coordinator for Abyme Visualizer
 * Handles WebSocket communication and coordinates between chat and tree visualization
 */

// Initialize components
let socket;
let treeViz;
let chat;
let isGenerating = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Socket.IO
    socket = io();

    // Initialize UI components
    treeViz = new TreeVisualizer('tree-svg');
    chat = new ChatUI('messages');

    // Setup WebSocket event handlers
    setupSocketHandlers();

    // Setup UI event handlers
    setupUIHandlers();
});

/**
 * Setup WebSocket event handlers
 */
function setupSocketHandlers() {
    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });

    socket.on('connected', (data) => {
        console.log('Server ready:', data);
        updateStatus('Ready');
    });

    socket.on('node_start', (data) => {
        const promptPreview = data.prompt.substring(0, 60) + (data.prompt.length > 60 ? '...' : '');
        console.log(`🔄 [Depth ${data.depth}] Node start: ${promptPreview}`);
        treeViz.addNode(data);
    });

    socket.on('node_waiting', (data) => {
        const outputPreview = data.output.substring(0, 60) + (data.output.length > 60 ? '...' : '');
        console.log(`⏳ Waiting for ${data.num_subproblems} subproblems: ${outputPreview}`);
        treeViz.setNodeWaiting(data.node_id, data.output, data.num_subproblems);
    });

    socket.on('node_complete', (data) => {
        const outputPreview = data.output.substring(0, 60) + (data.output.length > 60 ? '...' : '');
        console.log(`✓ [${data.latency.toFixed(2)}s] Node complete: ${outputPreview}`);
        treeViz.completeNode(data.node_id, data.output, data.latency);
    });

    socket.on('subproblem_created', (data) => {
        console.log(`├─ Subproblem delegation created`);
        treeViz.addEdge(data.parent_id, data.child_id, 'subproblem');
    });

    socket.on('continuation_created', (data) => {
        console.log(`➡️  Continuation created`);
        treeViz.addEdge(data.node_id, data.next_id, 'continuation');
    });

    socket.on('node_final', (data) => {
        console.log(`🏁 Final node reached`);
        treeViz.finalizeNode(data.node_id);
    });

    socket.on('generation_done', (data) => {
        console.log('═'.repeat(60));
        console.log('✅ Generation complete!');
        console.log(`📊 Stats: ${data.stats.total_calls} calls | Depth ${data.stats.max_depth} | ${data.stats.latency.toFixed(2)}s`);
        console.log('═'.repeat(60));

        chat.removeStatusMessage();
        chat.addAssistantMessage(data.final_output);

        updateStatus('Ready');
        updateFinalStats(data.stats);

        isGenerating = false;
        updateSendButton();
    });

    socket.on('error', (data) => {
        console.error('❌ Error:', data.message);
        chat.addErrorMessage(data.message);

        updateStatus('Error');

        isGenerating = false;
        updateSendButton();
    });

    socket.on('generation_stopped', (data) => {
        console.log('⏸️  Generation stopped:', data.message);
    });
}

/**
 * Setup UI event handlers
 */
function setupUIHandlers() {
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const promptInput = document.getElementById('prompt-input');
    const maxDepthSlider = document.getElementById('max-depth-slider');
    const maxCallSlider = document.getElementById('max-call-slider');
    const maxDepthValue = document.getElementById('max-depth-value');
    const maxCallValue = document.getElementById('max-call-value');

    // Send button click
    sendBtn.addEventListener('click', () => {
        sendPrompt();
    });

    // Stop button click
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            stopGeneration();
        });
    }

    // Enter key in textarea (Ctrl+Enter or Cmd+Enter to send)
    promptInput.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            sendPrompt();
        }
    });

    // Max depth slider
    if (maxDepthSlider && maxDepthValue) {
        maxDepthSlider.addEventListener('input', (event) => {
            maxDepthValue.textContent = event.target.value;
        });
    }

    // Max call slider
    if (maxCallSlider && maxCallValue) {
        maxCallSlider.addEventListener('input', (event) => {
            maxCallValue.textContent = event.target.value;
        });
    }
}

/**
 * Send prompt to server
 */
function sendPrompt() {
    const promptInput = document.getElementById('prompt-input');
    const prompt = promptInput.value.trim();

    if (!prompt || isGenerating) {
        return;
    }

    // Get slider values
    const maxDepth = parseInt(document.getElementById('max-depth-slider').value);
    const maxCall = parseInt(document.getElementById('max-call-slider').value);

    // Add user message to chat
    chat.addUserMessage(prompt);
    chat.addStatusMessage('Generating...');

    // Reset tree visualization
    treeViz.reset();

    // Update status
    updateStatus('Generating...');
    isGenerating = true;
    updateSendButton();

    // Send to server
    console.log('🚀 Starting generation...');
    console.log('Config:', { max_depth: maxDepth, max_call: maxCall, max_parallel_workers: 10 });

    socket.emit('generate', {
        prompt: prompt,
        config: {
            max_depth: maxDepth,
            max_call: maxCall,
            max_parallel_workers: 10  // Updated to 10 workers
        }
    });

    // Clear input
    promptInput.value = '';
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(connected) {
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');

    if (statusDot && statusText) {
        if (connected) {
            statusDot.classList.remove('bg-gray-500');
            statusDot.classList.add('bg-green-500');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('bg-green-500');
            statusDot.classList.add('bg-gray-500');
            statusText.textContent = 'Disconnected';
        }
    }
}

/**
 * Update status text
 */
function updateStatus(text) {
    const statusElem = document.getElementById('status');
    if (statusElem) {
        statusElem.textContent = text;

        // Update color based on status
        statusElem.className = 'font-medium';
        if (text === 'Ready' || text === 'Connected') {
            statusElem.classList.add('text-green-400');
        } else if (text === 'Generating...') {
            statusElem.classList.add('text-yellow-400');
        } else if (text === 'Error') {
            statusElem.classList.add('text-red-400');
        } else {
            statusElem.classList.add('text-gray-400');
        }
    }
}

/**
 * Update final stats after generation complete
 */
function updateFinalStats(stats) {
    const callCountElem = document.getElementById('call-count');
    const currentDepthElem = document.getElementById('current-depth');

    if (callCountElem && stats.total_calls !== undefined) {
        callCountElem.textContent = stats.total_calls;
    }

    if (currentDepthElem && stats.max_depth !== undefined) {
        currentDepthElem.textContent = stats.max_depth;
    }

    console.log('Final stats:', stats);
}

/**
 * Update send button state
 */
function updateSendButton() {
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const maxDepthSlider = document.getElementById('max-depth-slider');
    const maxCallSlider = document.getElementById('max-call-slider');

    if (sendBtn) {
        if (isGenerating) {
            sendBtn.disabled = true;
            sendBtn.textContent = 'Generating...';
        } else {
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
        }
    }

    if (stopBtn) {
        if (isGenerating) {
            stopBtn.classList.remove('hidden');
        } else {
            stopBtn.classList.add('hidden');
        }
    }

    // Lock/unlock sliders
    if (maxDepthSlider) {
        maxDepthSlider.disabled = isGenerating;
    }
    if (maxCallSlider) {
        maxCallSlider.disabled = isGenerating;
    }
}

/**
 * Stop the current generation
 */
function stopGeneration() {
    if (!isGenerating) return;

    console.log('⏸️  Stop requested - disconnecting from updates...');
    socket.emit('stop_generation');

    chat.removeStatusMessage();
    chat.addErrorMessage('Generation stopped by user (server may still be running background tasks)');
    updateStatus('Stopped');

    isGenerating = false;
    updateSendButton();
}
