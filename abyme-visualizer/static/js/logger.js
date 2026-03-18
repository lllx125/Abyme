/**
 * Logger Module
 * Provides structured console logging with timestamps and color-coded levels
 */

function timestamp() {
    const now = new Date();
    return now.toTimeString().split(' ')[0];
}

export const logger = {
    info: (module, message, data = null) => {
        const prefix = `[${timestamp()}] [INFO] [${module}]`;
        if (data !== null) {
            console.log(`%c${prefix} ${message}`, 'color: #4CAF50', data);
        } else {
            console.log(`%c${prefix} ${message}`, 'color: #4CAF50');
        }
    },

    warn: (module, message, data = null) => {
        const prefix = `[${timestamp()}] [WARN] [${module}]`;
        if (data !== null) {
            console.warn(`%c${prefix} ${message}`, 'color: #FF9800', data);
        } else {
            console.warn(`%c${prefix} ${message}`, 'color: #FF9800');
        }
    },

    error: (module, message, data = null) => {
        const prefix = `[${timestamp()}] [ERROR] [${module}]`;
        if (data !== null) {
            console.error(`%c${prefix} ${message}`, 'color: #F44336', data);
        } else {
            console.error(`%c${prefix} ${message}`, 'color: #F44336');
        }
    },

    debug: (module, message, data = null) => {
        const prefix = `[${timestamp()}] [DEBUG] [${module}]`;
        if (data !== null) {
            console.debug(`%c${prefix} ${message}`, 'color: #9E9E9E', data);
        } else {
            console.debug(`%c${prefix} ${message}`, 'color: #9E9E9E');
        }
    }
};
