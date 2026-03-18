/**
 * Sidebar Module
 * Handles the node detail sidebar with markdown rendering
 */

import { logger } from './logger.js';

const MODULE = 'Sidebar';

export class Sidebar {
    constructor() {
        this.sidebar = document.getElementById('detail-sidebar');
        this.closeBtn = document.getElementById('close-sidebar-btn');
        this.promptEl = document.getElementById('detail-prompt');
        this.outputEl = document.getElementById('detail-output');
        this.isOpen = false;

        this.init();
    }

    /**
     * Initialize event listeners
     */
    init() {
        logger.info(MODULE, 'Initializing sidebar');

        // Close button
        this.closeBtn.addEventListener('click', () => {
            this.hide();
        });

        // Click outside to close
        this.sidebar.addEventListener('click', (e) => {
            if (e.target === this.sidebar) {
                this.hide();
            }
        });
    }

    /**
     * Show node details
     * @param {object} node - Node data
     */
    showNodeDetail(node) {
        logger.info(MODULE, 'Showing node detail', node);

        // Render prompt with markdown
        const promptHtml = this.renderMarkdown(node.prompt);
        this.promptEl.innerHTML = promptHtml;

        // Render fragment + output with markdown
        const combinedText = node.fragment + '\n' + node.output;
        const outputHtml = this.renderMarkdown(combinedText);
        this.outputEl.innerHTML = outputHtml;

        // Show sidebar
        this.show();
    }

    /**
     * Render markdown with DOMPurify sanitization
     * @param {string} text - Markdown text
     * @returns {string} - Sanitized HTML
     */
    renderMarkdown(text) {
        if (!text) return '<em>No content</em>';

        // Parse markdown
        const rawHtml = marked.parse(text);

        // Sanitize to prevent XSS
        const cleanHtml = DOMPurify.sanitize(rawHtml);

        return cleanHtml;
    }

    /**
     * Show sidebar
     */
    show() {
        this.sidebar.classList.add('open');
        this.isOpen = true;
        logger.debug(MODULE, 'Sidebar shown');
    }

    /**
     * Hide sidebar
     */
    hide() {
        this.sidebar.classList.remove('open');
        this.isOpen = false;
        logger.debug(MODULE, 'Sidebar hidden');
    }
}
