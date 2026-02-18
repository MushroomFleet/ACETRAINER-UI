/**
 * visualization_panel.js -- PaCMAP visualization tab controller.
 * Connects to /visualization Socket.IO namespace, renders Plotly charts.
 */

const VizPanel = {
    socket: null,
    autoUpdate: true,
    updateInterval: 15000,
    updateTimer: null,
    isTabActive: false,

    init() {
        this.connectSocket();
        this.bindEvents();
    },

    // ===== Socket.IO (separate namespace) =====
    connectSocket() {
        this.socket = io('/visualization', {
            transports: ['websocket', 'polling'],
        });

        this.socket.on('connect', () => {
            console.log('[VizPanel] Connected to /visualization');
        });

        this.socket.on('visualization_update', (data) => {
            if (data.success) {
                this.renderPlot(data.figure);
                this.updateStats(data.stats);
                this.setBadge('Live', 'bg-green-900 text-green-300');
            } else {
                this.showEmpty(data.error || 'Insufficient data');
            }
        });

        this.socket.on('viz_status_update', (data) => {
            if (data.stats) this.updateStats(data.stats);
        });

        this.socket.on('animation_ready', (data) => {
            if (data.created) {
                const formats = Object.keys(data.created).join(' + ');
                Utils.success('Training animation ready! (' + formats + ')');
                this.setBadge('Animation Ready', 'bg-purple-900 text-purple-300');
            }
        });

        this.socket.on('disconnect', () => {
            console.log('[VizPanel] Disconnected');
        });
    },

    // ===== Events =====
    bindEvents() {
        const refreshBtn = document.getElementById('viz-refresh-btn');
        const toggleBtn = document.getElementById('viz-toggle-auto');
        const clearBtn = document.getElementById('viz-clear-all');
        const intervalSel = document.getElementById('viz-interval-select');
        const guideToggle = document.getElementById('viz-guide-toggle');

        if (refreshBtn) refreshBtn.addEventListener('click', () => this.requestUpdate());

        if (toggleBtn) toggleBtn.addEventListener('click', () => {
            this.autoUpdate = !this.autoUpdate;
            toggleBtn.textContent = this.autoUpdate ? 'Pause Auto-Update' : 'Resume Auto-Update';
            if (this.autoUpdate) {
                this.startAutoUpdate();
            } else {
                this.stopAutoUpdate();
            }
        });

        if (clearBtn) clearBtn.addEventListener('click', async () => {
            const ok = await Utils.confirm('Clear all visualization data? This cannot be undone.');
            if (!ok) return;
            try {
                await Utils.apiPost('/api/viz/clear/all', {});
                this.showEmpty('Data cleared. Will update when new embeddings arrive.');
                this.updateStats({
                    latent: { total_samples: 0 },
                    lora: { total_samples: 0 },
                    prompt: { total_samples: 0 },
                });
                this.setBadge('Cleared', 'bg-gray-800 text-gray-500');
            } catch (e) {
                Utils.error('Failed to clear: ' + e.message);
            }
        });

        if (intervalSel) intervalSel.addEventListener('change', (e) => {
            this.updateInterval = parseInt(e.target.value);
            if (this.autoUpdate) {
                this.stopAutoUpdate();
                this.startAutoUpdate();
            }
        });

        if (guideToggle) guideToggle.addEventListener('click', () => {
            const content = document.getElementById('viz-guide-content');
            const chevron = document.getElementById('viz-guide-chevron');
            content.classList.toggle('hidden');
            chevron.style.transform = content.classList.contains('hidden') ? '' : 'rotate(180deg)';
        });

        // Save Snapshot button
        const saveBtn = document.getElementById('viz-save-snapshot');
        if (saveBtn) saveBtn.addEventListener('click', async () => {
            try {
                const result = await Utils.apiPost('/api/viz/save', {});
                if (result.success) {
                    Utils.success('Snapshot saved: ' + Object.keys(result.saved).join(', '));
                } else {
                    Utils.error(result.error || 'Save failed');
                }
            } catch (e) {
                Utils.error('Failed to save snapshot: ' + e.message);
            }
        });

        // Create Animation button
        const animBtn = document.getElementById('viz-create-animation');
        if (animBtn) animBtn.addEventListener('click', async () => {
            animBtn.disabled = true;
            animBtn.textContent = 'Creating...';
            try {
                const result = await Utils.apiPost('/api/viz/create_animation', {});
                if (result.success) {
                    const formats = Object.keys(result.created).join(', ');
                    Utils.success('Animation created: ' + formats);
                } else {
                    Utils.error(result.error || 'Animation failed');
                }
            } catch (e) {
                Utils.error('Failed to create animation: ' + e.message);
            } finally {
                animBtn.disabled = false;
                animBtn.textContent = 'Create Animation';
            }
        });

        // Auto-save toggle
        const autoSaveCheck = document.getElementById('viz-auto-save');
        if (autoSaveCheck) autoSaveCheck.addEventListener('change', async () => {
            try {
                await Utils.apiPost('/api/viz/toggle_auto_save', { enabled: autoSaveCheck.checked });
            } catch (e) { /* non-fatal */ }
        });
    },

    // ===== Tab Activation =====
    onTabActivated() {
        this.isTabActive = true;
        this.requestUpdate();
        this.startAutoUpdate();
    },

    onTabDeactivated() {
        this.isTabActive = false;
        this.stopAutoUpdate();
    },

    // ===== Auto-Update =====
    startAutoUpdate() {
        this.stopAutoUpdate();
        if (!this.autoUpdate) return;
        this.updateTimer = setInterval(() => {
            if (this.isTabActive && this.autoUpdate) {
                this.requestUpdate();
            }
        }, this.updateInterval);
    },

    stopAutoUpdate() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    },

    // ===== Request Visualization =====
    requestUpdate() {
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_visualization');
        }
    },

    // ===== Render Plotly =====
    renderPlot(figureJson) {
        const container = document.getElementById('viz-plot-container');
        const emptyState = document.getElementById('viz-empty-state');

        try {
            const figure = JSON.parse(figureJson);
            if (emptyState) emptyState.style.display = 'none';

            Plotly.react(
                container,
                figure.data,
                figure.layout,
                {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                    displaylogo: false,
                }
            );
        } catch (e) {
            console.error('[VizPanel] Render error:', e);
            this.showEmpty('Failed to render visualization');
        }
    },

    // ===== Update Stats =====
    updateStats(stats) {
        if (!stats) return;
        const fmt = (s) => {
            const n = s.total_samples || 0;
            const proj = s.has_projection ? ' (projected)' : '';
            return `${n} samples${proj}`;
        };
        const el = (id, val) => {
            const e = document.getElementById(id);
            if (e) e.textContent = val;
        };
        el('viz-stat-latent', fmt(stats.latent || {}));
        el('viz-stat-lora', fmt(stats.lora || {}));
        el('viz-stat-prompt', fmt(stats.prompt || {}));
    },

    // ===== Empty State =====
    showEmpty(message) {
        const container = document.getElementById('viz-plot-container');
        const emptyState = document.getElementById('viz-empty-state');
        if (emptyState) {
            emptyState.style.display = '';
            const p = emptyState.querySelector('p');
            if (p) p.textContent = message || 'Waiting for training data...';
        }
        // Clear any existing Plotly chart
        try { Plotly.purge(container); } catch (e) {}
    },

    // ===== Status Badge =====
    setBadge(text, classes) {
        const badge = document.getElementById('viz-status-badge');
        if (badge) {
            badge.textContent = text;
            badge.className = 'text-xs px-2 py-1 rounded-full ' + classes;
        }
    },
};

window.VizPanel = VizPanel;
