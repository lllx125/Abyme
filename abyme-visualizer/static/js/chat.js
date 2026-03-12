/**
 * ChatUI - Manages the chat interface for user prompts and assistant responses
 */

class ChatUI {
    constructor(messagesContainerId) {
        this.messagesContainer = document.getElementById(messagesContainerId);
        this.clearInitialMessage();
    }

    /**
     * Clear the initial welcome message
     */
    clearInitialMessage() {
        if (this.messagesContainer && this.messagesContainer.children.length === 1) {
            this.messagesContainer.innerHTML = '';
        }
    }

    /**
     * Add a user message to the chat
     */
    addUserMessage(text) {
        this.clearInitialMessage();

        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex justify-end';

        messageDiv.innerHTML = `
            <div class="bg-blue-600 rounded-lg px-4 py-3 max-w-[85%]">
                <div class="text-sm text-white whitespace-pre-wrap">${this.escapeHtml(text)}</div>
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    /**
     * Add an assistant message to the chat
     */
    addAssistantMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex justify-start';

        messageDiv.innerHTML = `
            <div class="bg-gray-700 rounded-lg px-4 py-3 max-w-[85%]">
                <div class="text-sm font-semibold text-blue-400 mb-1">Abyme</div>
                <div class="text-sm text-gray-100 whitespace-pre-wrap">${this.escapeHtml(text)}</div>
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    /**
     * Add a status message (e.g., "Generating...")
     */
    addStatusMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex justify-center';
        messageDiv.id = 'status-message';

        messageDiv.innerHTML = `
            <div class="bg-gray-750 border border-gray-600 rounded-lg px-4 py-2">
                <div class="text-xs text-gray-400 italic flex items-center space-x-2">
                    <svg class="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>${this.escapeHtml(text)}</span>
                </div>
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    /**
     * Remove status message
     */
    removeStatusMessage() {
        const statusMsg = document.getElementById('status-message');
        if (statusMsg) {
            statusMsg.remove();
        }
    }

    /**
     * Add an error message to the chat
     */
    addErrorMessage(text) {
        this.removeStatusMessage();

        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex justify-center';

        messageDiv.innerHTML = `
            <div class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg px-4 py-3 max-w-[85%]">
                <div class="text-sm font-semibold text-red-400 mb-1 flex items-center space-x-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span>Error</span>
                </div>
                <div class="text-sm text-red-200 whitespace-pre-wrap">${this.escapeHtml(text)}</div>
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    /**
     * Scroll chat to bottom
     */
    scrollToBottom() {
        if (this.messagesContainer) {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
}
