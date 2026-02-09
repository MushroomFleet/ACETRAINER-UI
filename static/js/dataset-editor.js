/**
 * dataset-editor.js — Full dataset editor: file upload, carousel, caption editing,
 * bulk operations, validation, ZIP import/export, and upload to server.
 */

const DatasetEditor = {
    samples: [],
    currentIndex: -1,
    audioUrl: null,

    init() {
        this.bindEvents();
        this.loadSamples();
    },

    bindEvents() {
        // File inputs
        document.getElementById('btn-add-files').addEventListener('click', () => {
            document.getElementById('file-input-mp3').click();
        });
        document.getElementById('file-input-mp3').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
            e.target.value = '';
        });

        document.getElementById('btn-import-zip').addEventListener('click', () => {
            document.getElementById('file-input-zip').click();
        });
        document.getElementById('file-input-zip').addEventListener('change', (e) => {
            if (e.target.files[0]) this.handleZipImport(e.target.files[0]);
            e.target.value = '';
        });

        // Toolbar
        document.getElementById('btn-export-zip').addEventListener('click', () => this.exportZip());
        document.getElementById('btn-clear-all').addEventListener('click', () => this.clearAll());

        // Navigation
        document.getElementById('btn-prev').addEventListener('click', () => this.navigate(-1));
        document.getElementById('btn-next').addEventListener('click', () => this.navigate(1));

        // Editor actions
        document.getElementById('btn-save-sample').addEventListener('click', () => this.saveCurrent());
        document.getElementById('btn-delete-sample').addEventListener('click', () => this.deleteCurrent());

        // Instrumental toggle
        document.getElementById('editor-instrumental').addEventListener('change', (e) => {
            document.getElementById('editor-lyrics').disabled = e.target.checked;
            if (e.target.checked) {
                document.getElementById('editor-lyrics').style.opacity = '0.4';
            } else {
                document.getElementById('editor-lyrics').style.opacity = '1';
            }
        });

        // Bulk operations
        document.getElementById('btn-bulk-prepend').addEventListener('click', () => this.bulkPrepend());
        document.getElementById('btn-bulk-append').addEventListener('click', () => this.bulkAppend());
        document.getElementById('btn-bulk-replace').addEventListener('click', () => this.bulkReplace());

        // Bulk panel toggle
        document.getElementById('toggle-bulk-panel').addEventListener('click', () => {
            const panel = document.getElementById('bulk-panel');
            const chevron = document.getElementById('bulk-chevron');
            panel.classList.toggle('hidden');
            chevron.style.transform = panel.classList.contains('hidden') ? '' : 'rotate(180deg)';
        });

        // Validation & proceed
        document.getElementById('btn-validate').addEventListener('click', () => this.runValidation());
        document.getElementById('btn-proceed-trainer').addEventListener('click', () => this.proceedToTrainer());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (App.currentTab !== 'dataset') return;
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
            if (e.key === 'ArrowLeft') this.navigate(-1);
            if (e.key === 'ArrowRight') this.navigate(1);
        });
    },

    // ===== Load from IndexedDB =====
    async loadSamples() {
        this.samples = await DB.getAllSamples();
        // Sort by createdAt
        this.samples.sort((a, b) => a.createdAt - b.createdAt);
        this.renderSampleList();
        this.updateCounts();

        if (this.samples.length > 0 && this.currentIndex < 0) {
            this.selectSample(0);
        } else if (this.samples.length === 0) {
            this.currentIndex = -1;
            this.showEmptyState();
        }
    },

    // ===== File Upload =====
    async handleFileUpload(files) {
        if (!files || files.length === 0) return;
        Utils.showLoading(`Adding ${files.length} file(s)...`);

        let added = 0;
        for (const file of files) {
            if (!file.name.toLowerCase().endsWith('.mp3')) continue;
            try {
                await DB.addFromFile(file);
                added++;
            } catch (e) {
                console.error('Error adding file:', file.name, e);
            }
        }

        Utils.hideLoading();
        if (added > 0) {
            Utils.success(`Added ${added} audio file(s)`);
            await this.loadSamples();
            this.selectSample(this.samples.length - 1);
        }
    },

    // ===== ZIP Import =====
    async handleZipImport(file) {
        Utils.showLoading('Extracting ZIP...');
        try {
            const zip = await JSZip.loadAsync(file);
            const entries = {};

            // First pass: categorize files
            for (const [path, zipEntry] of Object.entries(zip.files)) {
                if (zipEntry.dir) continue;
                const name = path.split('/').pop();
                if (!name) continue;

                if (name.endsWith('.mp3')) {
                    const stem = name.replace('.mp3', '');
                    if (!entries[stem]) entries[stem] = {};
                    entries[stem].audio = zipEntry;
                } else if (name.endsWith('_prompt.txt')) {
                    const stem = name.replace('_prompt.txt', '');
                    if (!entries[stem]) entries[stem] = {};
                    entries[stem].prompt = zipEntry;
                } else if (name.endsWith('_lyrics.txt')) {
                    const stem = name.replace('_lyrics.txt', '');
                    if (!entries[stem]) entries[stem] = {};
                    entries[stem].lyrics = zipEntry;
                }
            }

            let added = 0;
            for (const [stem, files] of Object.entries(entries)) {
                if (!files.audio) continue;
                const audioBlob = await files.audio.async('blob');
                const prompt = files.prompt ? await files.prompt.async('string') : '';
                const lyrics = files.lyrics ? await files.lyrics.async('string') : '';
                await DB.addFromZipEntry(stem, audioBlob, prompt.trim(), lyrics.trim());
                added++;
            }

            Utils.hideLoading();
            Utils.success(`Imported ${added} sample(s) from ZIP`);
            await this.loadSamples();
            if (this.samples.length > 0) this.selectSample(0);

        } catch (e) {
            Utils.hideLoading();
            Utils.error('Failed to import ZIP: ' + e.message);
        }
    },

    // ===== ZIP Export =====
    async exportZip() {
        const samples = await DB.getAllSamples();
        if (samples.length === 0) {
            Utils.error('No samples to export');
            return;
        }

        // Validate first
        const stats = await DB.getStats();
        if (stats.duplicateNames.length > 0) {
            Utils.error('Cannot export: duplicate filenames found: ' + stats.duplicateNames.join(', '));
            return;
        }

        Utils.showLoading('Building ZIP...');
        try {
            const zip = new JSZip();
            const folder = zip.folder('data');

            for (const s of samples) {
                // Audio
                if (s.audioBlob) {
                    folder.file(`${s.filename}.mp3`, s.audioBlob);
                }
                // Prompt
                folder.file(`${s.filename}_prompt.txt`, s.prompt || '');
                // Lyrics
                folder.file(`${s.filename}_lyrics.txt`, s.lyrics || '');
            }

            const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 1 } });
            const datasetName = document.getElementById('dataset-name').value || 'dataset';
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${datasetName}.zip`;
            a.click();
            URL.revokeObjectURL(url);

            Utils.hideLoading();
            Utils.success(`Exported ${samples.length} samples as ZIP`);
        } catch (e) {
            Utils.hideLoading();
            Utils.error('Export failed: ' + e.message);
        }
    },

    // ===== Clear All =====
    async clearAll() {
        if (this.samples.length === 0) return;
        const ok = await Utils.confirm(`Delete all ${this.samples.length} samples from the editor? This cannot be undone.`);
        if (!ok) return;

        await DB.clearAllSamples();
        this.samples = [];
        this.currentIndex = -1;
        this.renderSampleList();
        this.showEmptyState();
        this.updateCounts();
        Utils.success('All samples cleared');
    },

    // ===== Sample List =====
    renderSampleList() {
        const container = document.getElementById('sample-list');
        if (this.samples.length === 0) {
            container.innerHTML = '<div class="p-4 text-center text-gray-600 text-sm">No samples loaded.<br>Click "+ Add MP3s" to begin.</div>';
            return;
        }

        container.innerHTML = this.samples.map((s, i) => {
            const valid = DB.checkValidity(s);
            const active = i === this.currentIndex ? 'active' : '';
            const dur = Utils.formatDuration(s.audioDuration);
            return `<div class="sample-item ${active}" data-index="${i}">
                <span class="dot ${valid ? 'valid' : 'invalid'}"></span>
                <span class="name">${this.escHtml(s.filename)}</span>
                <span class="dur">${dur}</span>
            </div>`;
        }).join('');

        // Click handlers
        container.querySelectorAll('.sample-item').forEach(el => {
            el.addEventListener('click', () => {
                this.selectSample(parseInt(el.dataset.index));
            });
        });
    },

    // ===== Select / Navigate =====
    selectSample(index) {
        if (index < 0 || index >= this.samples.length) return;

        // Auto-save current before switching
        if (this.currentIndex >= 0 && this.currentIndex < this.samples.length) {
            this.saveCurrentSilent();
        }

        this.currentIndex = index;
        this.renderSampleList();
        this.renderEditor();
    },

    navigate(direction) {
        const newIndex = this.currentIndex + direction;
        if (newIndex >= 0 && newIndex < this.samples.length) {
            this.selectSample(newIndex);
        }
    },

    // ===== Editor =====
    renderEditor() {
        if (this.currentIndex < 0 || this.currentIndex >= this.samples.length) {
            this.showEmptyState();
            return;
        }

        const s = this.samples[this.currentIndex];

        document.getElementById('editor-empty').classList.add('hidden');
        document.getElementById('editor-form').classList.remove('hidden');

        // Navigation
        document.getElementById('nav-position').textContent = `${this.currentIndex + 1} / ${this.samples.length}`;
        document.getElementById('btn-prev').disabled = this.currentIndex === 0;
        document.getElementById('btn-next').disabled = this.currentIndex === this.samples.length - 1;

        // Validity badge
        const valid = DB.checkValidity(s);
        const badge = document.getElementById('editor-validity');
        badge.textContent = valid ? 'Valid' : 'Incomplete';
        badge.className = `text-sm px-2 py-0.5 rounded ${valid ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`;

        // Audio
        const player = document.getElementById('audio-player');
        if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
        if (s.audioBlob) {
            this.audioUrl = URL.createObjectURL(s.audioBlob);
            player.src = this.audioUrl;
        } else {
            player.src = '';
        }

        document.getElementById('editor-filename-display').textContent = `${s.filename}.mp3`;
        document.getElementById('editor-duration').textContent = `Duration: ${Utils.formatDuration(s.audioDuration)}`;

        // Fields
        document.getElementById('editor-stem').value = s.filename || '';
        document.getElementById('editor-prompt').value = s.prompt || '';
        document.getElementById('editor-instrumental').checked = s.isInstrumental || false;
        document.getElementById('editor-lyrics').value = s.lyrics || '';
        document.getElementById('editor-lyrics').disabled = s.isInstrumental || false;
        document.getElementById('editor-lyrics').style.opacity = s.isInstrumental ? '0.4' : '1';
    },

    showEmptyState() {
        document.getElementById('editor-empty').classList.remove('hidden');
        document.getElementById('editor-form').classList.add('hidden');
    },

    // ===== Save =====
    async saveCurrent() {
        if (this.currentIndex < 0) return;
        await this.saveCurrentSilent();
        await this.loadSamples();
        Utils.success('Sample saved');
    },

    async saveCurrentSilent() {
        if (this.currentIndex < 0 || this.currentIndex >= this.samples.length) return;
        const s = this.samples[this.currentIndex];

        s.filename = document.getElementById('editor-stem').value.trim();
        s.prompt = document.getElementById('editor-prompt').value.trim();
        s.isInstrumental = document.getElementById('editor-instrumental').checked;
        s.lyrics = document.getElementById('editor-lyrics').value;
        s.isValid = DB.checkValidity(s);

        await DB.putSample(s);
    },

    // ===== Delete =====
    async deleteCurrent() {
        if (this.currentIndex < 0) return;
        const s = this.samples[this.currentIndex];
        const ok = await Utils.confirm(`Delete sample "${s.filename}"?`);
        if (!ok) return;

        await DB.deleteSample(s.id);
        this.samples.splice(this.currentIndex, 1);

        if (this.currentIndex >= this.samples.length) {
            this.currentIndex = this.samples.length - 1;
        }

        await this.loadSamples();
        if (this.currentIndex >= 0) {
            this.renderEditor();
        }
        Utils.success('Sample deleted');
    },

    // ===== Counts =====
    updateCounts() {
        const total = this.samples.length;
        const valid = this.samples.filter(s => DB.checkValidity(s)).length;
        document.getElementById('sample-count').textContent = total;
        document.getElementById('valid-count').textContent = valid;
        document.getElementById('invalid-count').textContent = total - valid;
    },

    // ===== Bulk Operations =====
    async bulkPrepend() {
        const tags = document.getElementById('bulk-prepend').value.trim();
        if (!tags) return;
        const samples = await DB.getAllSamples();
        for (const s of samples) {
            if (s.prompt) {
                s.prompt = tags + ', ' + s.prompt;
            } else {
                s.prompt = tags;
            }
            s.isValid = DB.checkValidity(s);
            await DB.putSample(s);
        }
        await this.loadSamples();
        this.renderEditor();
        Utils.success(`Prepended tags to ${samples.length} samples`);
    },

    async bulkAppend() {
        const tags = document.getElementById('bulk-append').value.trim();
        if (!tags) return;
        const samples = await DB.getAllSamples();
        for (const s of samples) {
            if (s.prompt) {
                s.prompt = s.prompt + ', ' + tags;
            } else {
                s.prompt = tags;
            }
            s.isValid = DB.checkValidity(s);
            await DB.putSample(s);
        }
        await this.loadSamples();
        this.renderEditor();
        Utils.success(`Appended tags to ${samples.length} samples`);
    },

    async bulkReplace() {
        const find = document.getElementById('bulk-find').value;
        const replace = document.getElementById('bulk-replace').value;
        if (!find) return;
        const samples = await DB.getAllSamples();
        let changed = 0;
        for (const s of samples) {
            if (s.prompt && s.prompt.includes(find)) {
                s.prompt = s.prompt.split(find).join(replace);
                s.isValid = DB.checkValidity(s);
                await DB.putSample(s);
                changed++;
            }
        }
        await this.loadSamples();
        this.renderEditor();
        Utils.success(`Replaced in ${changed} sample(s)`);
    },

    // ===== Validation =====
    async runValidation() {
        await DB.revalidateAll();
        await this.loadSamples();
        const stats = await DB.getStats();
        this.renderValidationSummary(stats);
    },

    renderValidationSummary(stats) {
        const el = document.getElementById('validation-summary');
        const issues = [];

        if (stats.missingPrompt.length > 0) {
            issues.push(`<span class="text-red-400">Missing prompt (${stats.missingPrompt.length}): ${stats.missingPrompt.slice(0, 5).join(', ')}${stats.missingPrompt.length > 5 ? '...' : ''}</span>`);
        }
        if (stats.missingLyrics.length > 0) {
            issues.push(`<span class="text-red-400">Missing lyrics (${stats.missingLyrics.length}): ${stats.missingLyrics.slice(0, 5).join(', ')}${stats.missingLyrics.length > 5 ? '...' : ''}</span>`);
        }
        if (stats.duplicateNames.length > 0) {
            issues.push(`<span class="text-red-400">Duplicate filenames: ${stats.duplicateNames.join(', ')}</span>`);
        }

        el.innerHTML = `
            <div class="space-y-1">
                <div>Total samples: <span class="text-gray-200 font-mono">${stats.total}</span></div>
                <div>Valid: <span class="text-green-400 font-mono">${stats.validCount}</span> &nbsp; Invalid: <span class="text-red-400 font-mono">${stats.invalidCount}</span></div>
                <div>Total audio: <span class="font-mono">${Utils.formatDuration(stats.totalDurationSec)}</span> &nbsp; Avg: <span class="font-mono">${Utils.formatDuration(stats.avgDurationSec)}</span></div>
                ${issues.length > 0 ? '<div class="mt-2 space-y-1">' + issues.join('<br>') + '</div>' : '<div class="text-green-400 mt-2">All samples valid!</div>'}
            </div>
        `;

        const proceedBtn = document.getElementById('btn-proceed-trainer');
        proceedBtn.disabled = !stats.isReady;
    },

    // ===== Proceed to Trainer =====
    async proceedToTrainer() {
        const samples = await DB.getAllSamples();
        if (samples.length === 0) {
            Utils.error('No samples to upload');
            return;
        }

        // Validate
        const stats = await DB.getStats();
        if (!stats.isReady) {
            Utils.error('Dataset is not valid. Please fix all issues first.');
            return;
        }

        Utils.showLoading('Uploading dataset to server...');

        try {
            // Clear server data dir first
            await Utils.apiDelete('/api/dataset/clear');

            // Upload each sample's files
            let uploaded = 0;
            for (const s of samples) {
                const formData = new FormData();
                formData.append('stem', s.filename);
                formData.append('prompt', s.prompt || '');
                formData.append('lyrics', s.lyrics || '');
                if (s.audioBlob) {
                    formData.append('audio', s.audioBlob, `${s.filename}.mp3`);
                }
                await Utils.apiUpload('/api/dataset/save-sample', formData);
                uploaded++;
            }

            Utils.hideLoading();
            Utils.success(`Uploaded ${uploaded} samples to server`);

            // Switch to trainer tab
            App.switchTab('trainer');

        } catch (e) {
            Utils.hideLoading();
            Utils.error('Upload failed: ' + e.message);
        }
    },

    // ===== Utility =====
    escHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },
};

window.DatasetEditor = DatasetEditor;
