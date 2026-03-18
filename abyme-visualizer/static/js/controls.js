/**
 * Controls Module
 * Handles user inputs, sliders, and buttons
 */

import { logger } from './logger.js';

const MODULE = 'Controls';

export class Controls {
    constructor(onStart, onStop) {
        this.onStart = onStart;
        this.onStop = onStop;
        this.isGenerating = false;

        this.elements = {
            promptInput: document.getElementById('prompt-input'),
            modelSelect: document.getElementById('model-select'),
            maxDepthSlider: document.getElementById('max-depth-slider'),
            maxDepthValue: document.getElementById('max-depth-value'),
            maxCallSlider: document.getElementById('max-call-slider'),
            maxCallValue: document.getElementById('max-call-value'),
            maxChainSlider: document.getElementById('max-chain-slider'),
            maxChainValue: document.getElementById('max-chain-value'),
            maxWorkersSlider: document.getElementById('max-workers-slider'),
            maxWorkersValue: document.getElementById('max-workers-value'),
            startBtn: document.getElementById('start-btn'),
            stopBtn: document.getElementById('stop-btn')
        };

        this.init();
    }

    /**
     * Initialize event listeners
     */
    init() {
        logger.info(MODULE, 'Initializing controls');

        // Slider value updates
        this.elements.maxDepthSlider.addEventListener('input', (e) => {
            this.elements.maxDepthValue.textContent = e.target.value;
            logger.debug(MODULE, 'Max depth changed', e.target.value);
        });

        this.elements.maxCallSlider.addEventListener('input', (e) => {
            this.elements.maxCallValue.textContent = e.target.value;
            logger.debug(MODULE, 'Max call changed', e.target.value);
        });

        this.elements.maxChainSlider.addEventListener('input', (e) => {
            this.elements.maxChainValue.textContent = e.target.value;
            logger.debug(MODULE, 'Max chain length changed', e.target.value);
        });

        this.elements.maxWorkersSlider.addEventListener('input', (e) => {
            this.elements.maxWorkersValue.textContent = e.target.value;
            logger.debug(MODULE, 'Max workers changed', e.target.value);
        });

        // Button click handlers
        this.elements.startBtn.addEventListener('click', () => {
            logger.info(MODULE, 'Start button clicked');
            this.handleStart();
        });

        this.elements.stopBtn.addEventListener('click', () => {
            logger.info(MODULE, 'Stop button clicked');
            this.handleStop();
        });
    }

    /**
     * Handle start button click
     */
    handleStart() {
        const prompt = this.elements.promptInput.value.trim();

        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        const params = {
            prompt: prompt,
            model: this.elements.modelSelect.value,
            max_depth: parseInt(this.elements.maxDepthSlider.value),
            max_call: parseInt(this.elements.maxCallSlider.value),
            max_chain_length: parseInt(this.elements.maxChainSlider.value),
            max_parallel_workers: parseInt(this.elements.maxWorkersSlider.value)
        };

        logger.info(MODULE, 'Starting generation with params', params);

        if (this.onStart) {
            this.onStart(params);
        }

        this.setGenerating(true);
    }

    /**
     * Handle stop button click
     */
    handleStop() {
        if (this.onStop) {
            this.onStop();
        }
    }

    /**
     * Set generating state
     * @param {boolean} generating - Whether generation is in progress
     */
    setGenerating(generating) {
        this.isGenerating = generating;

        if (generating) {
            // Disable controls
            this.elements.promptInput.disabled = true;
            this.elements.modelSelect.disabled = true;
            this.elements.maxDepthSlider.disabled = true;
            this.elements.maxCallSlider.disabled = true;
            this.elements.maxChainSlider.disabled = true;
            this.elements.maxWorkersSlider.disabled = true;

            // Hide start, show stop
            this.elements.startBtn.style.display = 'none';
            this.elements.stopBtn.style.display = 'block';
        } else {
            // Enable controls
            this.elements.promptInput.disabled = false;
            this.elements.modelSelect.disabled = false;
            this.elements.maxDepthSlider.disabled = false;
            this.elements.maxCallSlider.disabled = false;
            this.elements.maxChainSlider.disabled = false;
            this.elements.maxWorkersSlider.disabled = false;

            // Show start, hide stop
            this.elements.startBtn.style.display = 'block';
            this.elements.stopBtn.style.display = 'none';
        }

        logger.debug(MODULE, `Generating state set to: ${generating}`);
    }
}
