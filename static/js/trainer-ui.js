/**
 * trainer-ui.js — Training configuration, execution control, real-time monitoring,
 * loss chart, log terminal, checkpoint listing, GPU stats.
 */

const TrainerUI = {
    lossChart: null,
    lossData: [],
    isRunning: false,
    lastStepTime: null,

    // ===== Presets =====
    PRESETS: {
        conservative: {
            lora_r: 16, lora_alpha: 32, precision: 'bf16-mixed',
            accumulate_grad_batches: 2, label: 'Conservative (~14GB)'
        },
        balanced: {
            lora_r: 64, lora_alpha: 32, precision: 'bf16-mixed',
            accumulate_grad_batches: 1, label: 'Balanced (~18GB)'
        },
        aggressive: {
            lora_r: 256, lora_alpha: 32, precision: 'bf16-mixed',
            accumulate_grad_batches: 1, label: 'Aggressive (~22GB)'
        },
    },

    init() {
        this.bindEvents();
        this.initLossChart();
        this.checkExistingTraining();
    },

    bindEvents() {
        // Dataset source toggle
        document.querySelectorAll('input[name="dataset-source"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const zipUpload = document.getElementById('trainer-zip-upload');
                zipUpload.classList.toggle('hidden', e.target.value !== 'zip');
            });
        });

        // ZIP upload for trainer
        document.getElementById('trainer-zip-input').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            Utils.showLoading('Uploading and extracting ZIP...');
            try {
                const formData = new FormData();
                formData.append('file', file);
                const result = await Utils.apiUpload('/api/dataset/upload-zip', formData);
                Utils.hideLoading();
                Utils.success(`Extracted ${result.count} samples from ZIP`);
            } catch (err) {
                Utils.hideLoading();
                Utils.error('ZIP upload failed: ' + err.message);
            }
        });

        // Convert dataset
        document.getElementById('btn-convert-dataset').addEventListener('click', () => this.convertDataset());

        // Presets
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => this.applyPreset(btn.dataset.preset));
        });

        // Training controls
        document.getElementById('btn-start-training').addEventListener('click', () => this.startTraining());
        document.getElementById('btn-stop-training').addEventListener('click', () => this.stopTraining());
        document.getElementById('btn-tensorboard').addEventListener('click', () => this.launchTensorBoard());
    },

    // ===== Presets =====
    applyPreset(name) {
        const p = this.PRESETS[name];
        if (!p) return;

        document.getElementById('cfg-lora-r').value = p.lora_r;
        document.getElementById('cfg-lora-alpha').value = p.lora_alpha;
        document.getElementById('cfg-precision').value = p.precision;
        document.getElementById('cfg-grad-accum').value = p.accumulate_grad_batches;

        // Update active state
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.preset === name);
        });
    },

    // ===== Dataset Conversion =====
    async convertDataset() {
        const name = document.getElementById('trainer-dataset-name').value.trim() || 'lora_dataset';
        const repeatCount = parseInt(document.getElementById('trainer-repeat-count').value) || 2000;
        const statusEl = document.getElementById('convert-status');

        statusEl.textContent = 'Converting...';
        statusEl.className = 'text-xs text-yellow-400';

        try {
            const result = await Utils.apiPost('/api/dataset/convert', {
                repeat_count: repeatCount,
                output_name: name,
            });

            if (result.success) {
                statusEl.textContent = `Ready: ${result.num_samples} samples x ${result.repeat_count} = ${result.total_rows} rows`;
                statusEl.className = 'text-xs text-green-400';
                this.showDatasetInfo(result);
                Utils.success('Dataset converted successfully');
            } else {
                statusEl.textContent = result.error || 'Conversion failed';
                statusEl.className = 'text-xs text-red-400';
            }
        } catch (e) {
            statusEl.textContent = 'Error: ' + e.message;
            statusEl.className = 'text-xs text-red-400';
            Utils.error('Dataset conversion failed: ' + e.message);
        }
    },

    showDatasetInfo(info) {
        const box = document.getElementById('dataset-info-box');
        box.classList.remove('hidden');
        box.innerHTML = `
            <div class="grid grid-cols-2 gap-2">
                <div>Unique samples: <span class="text-gray-200 font-mono">${info.num_samples}</span></div>
                <div>Repeat count: <span class="text-gray-200 font-mono">${info.repeat_count}</span></div>
                <div>Total rows: <span class="text-gray-200 font-mono">${info.total_rows.toLocaleString()}</span></div>
                <div>Path: <span class="text-gray-300 font-mono text-xs">${info.path}</span></div>
            </div>
        `;
    },

    async refreshDatasetInfo() {
        const name = document.getElementById('trainer-dataset-name').value.trim() || 'lora_dataset';
        try {
            const info = await Utils.apiGet(`/api/dataset/info?name=${encodeURIComponent(name)}`);
            if (info && !info.error) {
                const statusEl = document.getElementById('editor-dataset-status');
                statusEl.textContent = `(${info.unique_samples} samples, ${info.total_rows.toLocaleString()} rows)`;
                statusEl.className = 'text-xs text-green-400';
                document.getElementById('convert-status').textContent = `Ready: ${info.unique_samples} samples x ${info.repeat_count} = ${info.total_rows} rows`;
                document.getElementById('convert-status').className = 'text-xs text-green-400';
            }
        } catch (e) {
            // Dataset doesn't exist yet, that's fine
        }

        // Also validate server-side data dir
        try {
            const val = await Utils.apiGet('/api/dataset/validate');
            if (val.total > 0) {
                const statusEl = document.getElementById('editor-dataset-status');
                if (val.valid) {
                    statusEl.textContent = `(${val.total} samples on server, ready to convert)`;
                    statusEl.className = 'text-xs text-green-400';
                } else {
                    statusEl.textContent = `(${val.total} samples on server, ${val.invalid_count} invalid)`;
                    statusEl.className = 'text-xs text-yellow-400';
                }
            }
        } catch (e) { /* ignore */ }
    },

    // ===== Gather Config =====
    gatherConfig() {
        const datasetName = document.getElementById('trainer-dataset-name').value.trim() || 'lora_dataset';

        // Gather target modules
        const modules = [];
        document.querySelectorAll('.lora-module:checked').forEach(cb => {
            modules.push(cb.value);
        });

        return {
            dataset_path: datasetName,
            exp_name: document.getElementById('cfg-exp-name').value.trim() || 'my_lora',
            checkpoint_dir: document.getElementById('cfg-checkpoint-dir').value.trim() || '',
            ckpt_path: document.getElementById('cfg-ckpt-path').value.trim() || '',
            lora_r: parseInt(document.getElementById('cfg-lora-r').value) || 64,
            lora_alpha: parseInt(document.getElementById('cfg-lora-alpha').value) || 32,
            use_rslora: document.getElementById('cfg-rslora').checked,
            target_modules: modules,
            learning_rate: parseFloat(document.getElementById('cfg-lr').value) || 1e-4,
            max_steps: parseInt(document.getElementById('cfg-max-steps').value) || 5000,
            precision: document.getElementById('cfg-precision').value,
            accumulate_grad_batches: parseInt(document.getElementById('cfg-grad-accum').value) || 1,
            gradient_clip_val: parseFloat(document.getElementById('cfg-grad-clip').value) || 0.5,
            shift: parseFloat(document.getElementById('cfg-shift').value) || 3.0,
            num_workers: parseInt(document.getElementById('cfg-workers').value) || 4,
            save_every: parseInt(document.getElementById('cfg-save-every').value) || 500,
            plot_every: parseInt(document.getElementById('cfg-plot-every').value) || 1000,
        };
    },

    // ===== Start Training =====
    async startTraining() {
        const config = this.gatherConfig();

        // Validation
        if (!config.dataset_path) {
            Utils.error('Please specify a dataset name');
            return;
        }

        try {
            const result = await Utils.apiPost('/api/trainer/start', config);
            if (result.success) {
                this.setRunningState(true, config.max_steps);
                Utils.success('Training started (PID: ' + result.pid + ')');
            } else {
                Utils.error('Failed to start: ' + result.error);
            }
        } catch (e) {
            Utils.error('Failed to start training: ' + e.message);
        }
    },

    // ===== Stop Training =====
    async stopTraining() {
        const ok = await Utils.confirm('Stop the current training run? Progress will be lost since the last checkpoint.');
        if (!ok) return;

        try {
            await Utils.apiPost('/api/trainer/stop', {});
            Utils.info('Training stopped');
        } catch (e) {
            Utils.error('Failed to stop: ' + e.message);
        }
    },

    // ===== TensorBoard =====
    async launchTensorBoard() {
        try {
            const result = await Utils.apiPost('/api/trainer/tensorboard', {});
            if (result.success) {
                Utils.success('TensorBoard launched');
                window.open(result.url, '_blank');
            }
        } catch (e) {
            Utils.error('Failed to launch TensorBoard: ' + e.message);
        }
    },

    // ===== State Management =====
    setRunningState(running, maxSteps = 0) {
        this.isRunning = running;

        document.getElementById('btn-start-training').disabled = running;
        document.getElementById('btn-stop-training').disabled = !running;

        const badge = document.getElementById('training-status-badge');
        if (running) {
            badge.textContent = 'Training...';
            badge.className = 'text-sm px-3 py-1 rounded-full bg-purple-900 text-purple-300 animate-pulse';
        } else {
            badge.textContent = 'Idle';
            badge.className = 'text-sm px-3 py-1 rounded-full bg-gray-800 text-gray-400';
        }

        // Show progress section
        document.getElementById('progress-section').classList.toggle('hidden', !running && this.lossData.length === 0);

        if (running) {
            this.lossData = [];
            this.updateChart();
            document.getElementById('log-terminal').textContent = '';
            document.getElementById('checkpoint-list').innerHTML = 'No checkpoints yet.';
        }
    },

    // ===== Socket.IO Event Handlers =====

    onTrainingLog(data) {
        // Append to log terminal
        const terminal = document.getElementById('log-terminal');
        terminal.textContent += data.line + '\n';
        if (document.getElementById('log-autoscroll').checked) {
            terminal.scrollTop = terminal.scrollHeight;
        }

        // Update metrics if available
        if (data.metrics) {
            this.updateMetrics(data.metrics);
        }

        // Update progress
        if (data.step !== undefined && data.max_steps) {
            this.updateProgress(data.step, data.max_steps);
        }
    },

    onTrainingComplete(data) {
        this.setRunningState(false);

        if (data.stopped) {
            Utils.info('Training was stopped by user');
        } else if (data.return_code === 0) {
            Utils.success('Training completed successfully!');
        } else {
            Utils.error(`Training ended with exit code ${data.return_code}`);
        }

        // Keep progress section visible to see final state
        document.getElementById('progress-section').classList.remove('hidden');

        const badge = document.getElementById('training-status-badge');
        badge.textContent = data.return_code === 0 ? 'Completed' : (data.stopped ? 'Stopped' : 'Failed');
        badge.className = `text-sm px-3 py-1 rounded-full ${data.return_code === 0 ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`;
    },

    onCheckpointSaved(data) {
        Utils.info(`Checkpoint saved: ${data.name}`);
        this.addCheckpoint(data.name, data.step);
    },

    onGPUStats(stats) {
        if (!stats) return;
        const used = (stats.mem_used_mb / 1024).toFixed(1);
        const total = (stats.mem_total_mb / 1024).toFixed(1);
        document.getElementById('metric-gpu-mem').textContent = `${used} / ${total} GB`;
        document.getElementById('metric-gpu-util').textContent = `${stats.gpu_util_pct}%`;
    },

    onStatusUpdate(data) {
        if (data.is_running) {
            this.setRunningState(true, data.max_steps);
            if (data.current_step > 0) {
                this.updateProgress(data.current_step, data.max_steps);
            }
        }
    },

    // ===== Progress =====
    updateProgress(step, maxSteps) {
        const pct = maxSteps > 0 ? Math.min(100, (step / maxSteps) * 100) : 0;
        document.getElementById('progress-step').textContent = `Step ${step.toLocaleString()} / ${maxSteps.toLocaleString()}`;
        document.getElementById('progress-pct').textContent = `${pct.toFixed(1)}%`;
        document.getElementById('progress-bar').style.width = `${pct}%`;

        // Step time
        const now = Date.now();
        if (this.lastStepTime) {
            const elapsed = ((now - this.lastStepTime) / 1000).toFixed(1);
            document.getElementById('metric-step-time').textContent = `~${elapsed}s`;
        }
        this.lastStepTime = now;
    },

    // ===== Metrics =====
    updateMetrics(metrics) {
        if (metrics.loss !== undefined) {
            document.getElementById('metric-loss').textContent = metrics.loss.toFixed(4);
            this.lossData.push({ step: metrics.step || this.lossData.length, loss: metrics.loss });
            this.updateChart();
        }
        if (metrics.denoising_loss !== undefined) {
            document.getElementById('metric-denoise').textContent = metrics.denoising_loss.toFixed(4);
        }
        if (metrics.learning_rate !== undefined) {
            document.getElementById('metric-lr').textContent = metrics.learning_rate.toExponential(2);
        }
    },

    // ===== Loss Chart =====
    initLossChart() {
        const ctx = document.getElementById('loss-chart');
        if (!ctx) return;

        this.lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: 'Step', color: '#6b7280', font: { size: 10 } },
                        ticks: { color: '#6b7280', font: { size: 9 }, maxTicksLimit: 10 },
                        grid: { color: '#1f2937' },
                    },
                    y: {
                        display: true,
                        title: { display: true, text: 'Loss', color: '#6b7280', font: { size: 10 } },
                        ticks: { color: '#6b7280', font: { size: 9 } },
                        grid: { color: '#1f2937' },
                    }
                }
            }
        });
    },

    updateChart() {
        if (!this.lossChart) return;

        // Downsample if too many points
        let data = this.lossData;
        if (data.length > 500) {
            const step = Math.ceil(data.length / 500);
            data = data.filter((_, i) => i % step === 0);
        }

        this.lossChart.data.labels = data.map(d => d.step);
        this.lossChart.data.datasets[0].data = data.map(d => d.loss);
        this.lossChart.update();
    },

    // ===== Checkpoints =====
    addCheckpoint(name, step) {
        const container = document.getElementById('checkpoint-list');
        if (container.textContent.includes('No checkpoints')) {
            container.innerHTML = '';
        }
        const el = document.createElement('div');
        el.className = 'flex items-center gap-2 p-2 bg-gray-800 rounded text-sm';
        el.innerHTML = `
            <span class="text-purple-400 font-mono">${name}</span>
            <span class="text-gray-500">(step ${step})</span>
        `;
        container.appendChild(el);
    },

    // ===== Check Existing Training =====
    async checkExistingTraining() {
        try {
            const status = await Utils.apiGet('/api/trainer/status');
            if (status.is_running) {
                this.setRunningState(true, status.max_steps);
                this.updateProgress(status.current_step, status.max_steps);

                // Fetch existing logs
                const logs = await Utils.apiGet('/api/trainer/logs');
                if (logs.lines) {
                    document.getElementById('log-terminal').textContent = logs.lines.join('\n');
                }

                // Fetch metrics history for chart
                const metrics = await Utils.apiGet('/api/trainer/metrics');
                if (metrics.metrics) {
                    for (const m of metrics.metrics) {
                        if (m.loss !== undefined) {
                            this.lossData.push({ step: m.step || this.lossData.length, loss: m.loss });
                        }
                    }
                    this.updateChart();
                }

                // Show progress section
                document.getElementById('progress-section').classList.remove('hidden');
                Utils.info('Reconnected to running training');
            }
        } catch (e) { /* server not running yet */ }
    },
};

window.TrainerUI = TrainerUI;
