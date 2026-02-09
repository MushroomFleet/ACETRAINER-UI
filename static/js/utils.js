/**
 * utils.js — Toast notifications, confirm dialog, loading overlay, helpers.
 */

const Utils = {
    // ===== Toast Notifications =====
    toast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const el = document.createElement('div');
        el.className = `toast toast-${type}`;
        el.textContent = message;
        container.appendChild(el);
        setTimeout(() => {
            el.style.opacity = '0';
            el.style.transition = 'opacity 300ms';
            setTimeout(() => el.remove(), 300);
        }, duration);
    },

    success(msg) { this.toast(msg, 'success'); },
    error(msg) { this.toast(msg, 'error', 6000); },
    info(msg) { this.toast(msg, 'info'); },

    // ===== Confirm Dialog =====
    confirm(message) {
        return new Promise((resolve) => {
            const modal = document.getElementById('confirm-modal');
            const msgEl = document.getElementById('confirm-message');
            const cancelBtn = document.getElementById('confirm-cancel');
            const okBtn = document.getElementById('confirm-ok');

            msgEl.textContent = message;
            modal.classList.remove('hidden');

            const cleanup = () => {
                modal.classList.add('hidden');
                cancelBtn.removeEventListener('click', onCancel);
                okBtn.removeEventListener('click', onOk);
            };

            const onCancel = () => { cleanup(); resolve(false); };
            const onOk = () => { cleanup(); resolve(true); };

            cancelBtn.addEventListener('click', onCancel);
            okBtn.addEventListener('click', onOk);
        });
    },

    // ===== Loading Overlay =====
    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        const msg = document.getElementById('loading-message');
        msg.textContent = message;
        overlay.classList.remove('hidden');
    },

    hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
    },

    // ===== API Helpers =====
    async api(url, options = {}) {
        const defaults = {
            headers: { 'Content-Type': 'application/json' },
        };
        if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
            options.body = JSON.stringify(options.body);
        }
        if (options.body instanceof FormData) {
            delete defaults.headers['Content-Type'];
        }
        const resp = await fetch(url, { ...defaults, ...options });
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || `HTTP ${resp.status}`);
        }
        return data;
    },

    async apiGet(url) {
        return this.api(url);
    },

    async apiPost(url, body) {
        return this.api(url, { method: 'POST', body });
    },

    async apiDelete(url) {
        return this.api(url, { method: 'DELETE' });
    },

    async apiUpload(url, formData) {
        const resp = await fetch(url, { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        return data;
    },

    // ===== Formatting =====
    formatDuration(seconds) {
        if (!seconds || seconds <= 0) return '0:00';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    },

    formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },

    sanitizeStem(name) {
        // Remove extension, replace invalid chars
        let stem = name.replace(/\.[^.]+$/, '');
        stem = stem.replace(/[^a-zA-Z0-9_\-]/g, '_');
        stem = stem.replace(/_+/g, '_').replace(/^_|_$/g, '');
        return stem || 'untitled';
    },

    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2, 8);
    },

    // ===== Audio Duration =====
    getAudioDuration(blob) {
        return new Promise((resolve) => {
            const url = URL.createObjectURL(blob);
            const audio = new Audio();
            audio.addEventListener('loadedmetadata', () => {
                const duration = audio.duration;
                URL.revokeObjectURL(url);
                resolve(isFinite(duration) ? duration : 0);
            });
            audio.addEventListener('error', () => {
                URL.revokeObjectURL(url);
                resolve(0);
            });
            audio.src = url;
        });
    },

    // ===== Debounce =====
    debounce(fn, delay = 300) {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn(...args), delay);
        };
    },
};

window.Utils = Utils;
