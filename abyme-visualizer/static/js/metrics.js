/**
 * Metrics Module
 * Displays real-time metrics in the top bar
 */

import { logger } from './logger.js';

const MODULE = 'Metrics';

export class Metrics {
    constructor() {
        this.elements = {
            callCount: document.getElementById('call-count'),
            nodeCount: document.getElementById('node-count'),
            status: document.getElementById('status')
        };

        this.init();
    }

    /**
     * Initialize
     */
    init() {
        logger.info(MODULE, 'Initializing metrics display');
        this.reset();
    }

    /**
     * Update metrics
     * @param {object} metrics - Metrics data
     */
    update(metrics) {
        if (metrics.call_count !== undefined) {
            this.elements.callCount.textContent = metrics.call_count;
        }

        if (metrics.node_count !== undefined) {
            this.elements.nodeCount.textContent = metrics.node_count;
        }

        logger.debug(MODULE, 'Metrics updated', metrics);
    }

    /**
     * Set status
     * @param {string} status - Status text
     * @param {string} className - CSS class for styling
     */
    setStatus(status, className = '') {
        this.elements.status.textContent = status;
        this.elements.status.className = 'metrics-value ' + className;
        logger.debug(MODULE, `Status set to: ${status}`);
    }

    /**
     * Reset metrics
     */
    reset() {
        this.elements.callCount.textContent = '0';
        this.elements.nodeCount.textContent = '0';
        this.setStatus('Ready');
        logger.debug(MODULE, 'Metrics reset');
    }
}
