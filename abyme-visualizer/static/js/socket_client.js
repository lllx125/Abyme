/**
 * Socket.IO Client Wrapper
 * Manages WebSocket connection and event handling
 */

import { logger } from './logger.js';

const MODULE = 'SocketClient';

export class SocketClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.eventHandlers = new Map();
    }

    /**
     * Connect to the WebSocket server
     */
    connect() {
        logger.info(MODULE, 'Connecting to WebSocket server...');

        this.socket = io('http://localhost:5000/visualizer', {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 5
        });

        // Built-in Socket.IO events
        this.socket.on('connect', () => {
            this.connected = true;
            logger.info(MODULE, 'Connected to server', { socketId: this.socket.id });
        });

        this.socket.on('disconnect', (reason) => {
            this.connected = false;
            logger.warn(MODULE, 'Disconnected from server', { reason });
        });

        this.socket.on('connect_error', (error) => {
            logger.error(MODULE, 'Connection error', error);
        });

        this.socket.on('connected', (data) => {
            logger.info(MODULE, 'Received connected event', data);
        });

        return this;
    }

    /**
     * Register an event handler
     * @param {string} eventName - Name of the event
     * @param {function} handler - Handler function
     */
    on(eventName, handler) {
        if (!this.eventHandlers.has(eventName)) {
            this.eventHandlers.set(eventName, []);
        }
        this.eventHandlers.get(eventName).push(handler);

        // Register with Socket.IO
        this.socket.on(eventName, (data) => {
            logger.debug(MODULE, `Received event: ${eventName}`, data);
            handler(data);
        });

        return this;
    }

    /**
     * Emit an event to the server
     * @param {string} eventName - Name of the event
     * @param {object} data - Data to send
     */
    emit(eventName, data) {
        if (!this.connected) {
            logger.error(MODULE, 'Cannot emit - not connected');
            return;
        }

        logger.info(MODULE, `Emitting event: ${eventName}`, data);
        this.socket.emit(eventName, data);
    }

    /**
     * Send generation request
     * @param {object} params - Generation parameters
     */
    sendGenerateRequest(params) {
        this.emit('generate_request', params);
    }

    /**
     * Send stop request
     */
    sendStopRequest() {
        this.emit('stop_generation', {});
    }

    /**
     * Disconnect from server
     */
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            logger.info(MODULE, 'Disconnected');
        }
    }
}
