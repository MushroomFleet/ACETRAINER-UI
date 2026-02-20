/**
 * app.js — Tab navigation, global state, initialization.
 */

const App = {
    currentTab: 'dataset',
    socket: null,

    init() {
        this.initTabs();
        this.initSocket();
        this.pollGPU();
        DatasetEditor.init();
        TrainerUI.init();
        VizPanel.init();
    },

    // ===== Tab Navigation =====
    initTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.switchTab(btn.dataset.tab);
            });
        });
    },

    switchTab(tabName) {
        this.currentTab = tabName;

        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.style.display = content.id === `tab-${tabName}` ? '' : 'none';
            content.classList.toggle('active', content.id === `tab-${tabName}`);
        });

        // Refresh trainer dataset info when switching to trainer tab
        if (tabName === 'trainer') {
            TrainerUI.refreshDatasetInfo();
        }

        // Visualization tab activation/deactivation
        if (tabName === 'visualization') {
            VizPanel.onTabActivated();
        } else {
            VizPanel.onTabDeactivated();
        }
    },

    // ===== Socket.IO =====
    initSocket() {
        this.socket = io('/training', {
            transports: ['websocket', 'polling'],
        });

        this.socket.on('connect', () => {
            console.log('Socket.IO connected to /training');
        });

        this.socket.on('training_log', (data) => {
            TrainerUI.onTrainingLog(data);
        });

        this.socket.on('training_complete', (data) => {
            TrainerUI.onTrainingComplete(data);
        });

        this.socket.on('checkpoint_saved', (data) => {
            TrainerUI.onCheckpointSaved(data);
        });

        this.socket.on('gpu_stats', (data) => {
            this.updateGPUBadge(data);
            TrainerUI.onGPUStats(data);
        });

        this.socket.on('training_status', (data) => {
            TrainerUI.onStatusUpdate(data);
        });

        this.socket.on('disconnect', () => {
            console.log('Socket.IO disconnected');
        });
    },

    // ===== GPU Badge (header) =====
    updateGPUBadge(stats) {
        if (!stats) return;
        const nameEl = document.getElementById('gpu-name');
        const memEl = document.getElementById('gpu-mem');
        const tempEl = document.getElementById('gpu-temp');

        if (stats.name) nameEl.textContent = stats.name;
        if (stats.mem_used_mb !== undefined) {
            const used = (stats.mem_used_mb / 1024).toFixed(1);
            const total = (stats.mem_total_mb / 1024).toFixed(1);
            memEl.textContent = `VRAM: ${used} / ${total} GB`;

            // Color code based on usage
            const pct = stats.mem_used_mb / stats.mem_total_mb;
            memEl.className = 'px-2 py-0.5 rounded text-xs';
            if (pct > 0.9) memEl.classList.add('bg-red-900', 'text-red-300');
            else if (pct > 0.7) memEl.classList.add('bg-yellow-900', 'text-yellow-300');
            else memEl.classList.add('bg-gray-800', 'text-gray-400');
        }
        if (stats.temp_c !== undefined) {
            tempEl.textContent = `${stats.temp_c}C`;
        }
    },

    pollGPU() {
        // Poll GPU stats every 10s for the header badge
        const poll = async () => {
            try {
                const stats = await Utils.apiGet('/api/trainer/gpu');
                this.updateGPUBadge(stats);
            } catch (e) { /* nvidia-smi not available */ }
        };
        poll();
        setInterval(poll, 10000);
    },
};

// ===== Boot =====
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
