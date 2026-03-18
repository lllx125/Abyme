/**
 * Main Application Coordinator
 * Initializes and coordinates all modules
 */

import { logger } from './logger.js';
import { SocketClient } from './socket_client.js';
import { TreeRenderer } from './tree_renderer.js';
import { Controls } from './controls.js';
import { Sidebar } from './sidebar.js';
import { Metrics } from './metrics.js';

const MODULE = 'Main';

class AbymeVisualizer {
    constructor() {
        this.socket = null;
        this.treeRenderer = null;
        this.controls = null;
        this.sidebar = null;
        this.metrics = null;
    }

    /**
     * Initialize the application
     */
    init() {
        logger.info(MODULE, '='.repeat(60));
        logger.info(MODULE, 'Abyme Tree Visualizer Starting...');
        logger.info(MODULE, '='.repeat(60));

        // Initialize modules
        this.metrics = new Metrics();
        this.sidebar = new Sidebar();
        this.treeRenderer = new TreeRenderer('tree-canvas', (node) => this.handleNodeClick(node));
        this.controls = new Controls(
            (params) => this.handleStart(params),
            () => this.handleStop()
        );

        // Initialize WebSocket
        this.socket = new SocketClient();
        this.socket.connect();

        // Register WebSocket event handlers
        this.registerSocketEvents();

        logger.info(MODULE, 'Application initialized successfully');
    }

    /**
     * Register WebSocket event handlers
     */
    registerSocketEvents() {
        // Tree update events
        this.socket.on('tree_update', (data) => {
            logger.info(MODULE, 'Received tree_update', data);

            // Update tree visualization
            if (data.tree && data.tree.nodes && data.tree.edges) {
                this.treeRenderer.updateTree(data.tree.nodes, data.tree.edges);
            }

            // Update metrics
            if (data.metrics) {
                this.metrics.update(data.metrics);

                if (data.metrics.is_finished) {
                    this.handleGenerationComplete();
                }
            }
        });

        // Generation started
        this.socket.on('generation_started', (data) => {
            logger.info(MODULE, 'Generation started', data);
            this.metrics.setStatus('Generating...', 'status-generating');
        });

        // Generation complete
        this.socket.on('generation_complete', (data) => {
            logger.info(MODULE, 'Generation complete', data);
            this.handleGenerationComplete();
            this.metrics.setStatus('Complete', 'status-complete');

            if (data.final_output) {
                logger.info(MODULE, 'Final output:', data.final_output);
            }
        });

        // Generation stopped
        this.socket.on('generation_stopped', (data) => {
            logger.info(MODULE, 'Generation stopped', data);
            this.handleGenerationComplete();
            this.metrics.setStatus('Stopped', 'status-stopped');
        });

        // Generation error
        this.socket.on('generation_error', (data) => {
            logger.error(MODULE, 'Generation error', data);
            this.handleGenerationComplete();
            this.metrics.setStatus('Error', 'status-error');

            if (data.error_message) {
                alert(`Error: ${data.error_message}`);
            }
        });
    }

    /**
     * Handle start button click
     * @param {object} params - Generation parameters
     */
    handleStart(params) {
        logger.info(MODULE, 'Starting generation', params);

        // Clear previous tree
        this.treeRenderer.clear();
        this.metrics.reset();
        this.metrics.setStatus('Starting...', 'status-starting');

        // Send generation request
        this.socket.sendGenerateRequest(params);
    }

    /**
     * Handle stop button click
     */
    handleStop() {
        logger.info(MODULE, 'Stopping generation');
        this.socket.sendStopRequest();
        this.metrics.setStatus('Stopping...', 'status-stopping');
    }

    /**
     * Handle generation complete (re-enable controls)
     */
    handleGenerationComplete() {
        logger.info(MODULE, 'Generation completed - re-enabling controls');
        this.controls.setGenerating(false);
    }

    /**
     * Handle node click (show sidebar)
     * @param {object} node - Node data
     */
    handleNodeClick(node) {
        logger.info(MODULE, 'Node clicked, showing sidebar', node);
        this.sidebar.showNodeDetail(node);
    }
}

/**
 * Application entry point
 */
export function initApp() {
    const app = new AbymeVisualizer();
    app.init();

    // Expose to window for debugging
    window.AbymeVisualizer = app;

    logger.info(MODULE, 'Application ready!');
}
