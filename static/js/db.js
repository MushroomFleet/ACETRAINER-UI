/**
 * db.js — IndexedDB wrapper for dataset samples stored client-side.
 */

const DB = {
    DB_NAME: 'ACEStepDatasets',
    DB_VERSION: 1,
    STORE_SAMPLES: 'samples',
    STORE_META: 'metadata',
    _db: null,

    async open() {
        if (this._db) return this._db;
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(this.DB_NAME, this.DB_VERSION);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(this.STORE_SAMPLES)) {
                    const store = db.createObjectStore(this.STORE_SAMPLES, { keyPath: 'id' });
                    store.createIndex('filename', 'filename', { unique: false });
                    store.createIndex('createdAt', 'createdAt', { unique: false });
                }
                if (!db.objectStoreNames.contains(this.STORE_META)) {
                    db.createObjectStore(this.STORE_META, { keyPath: 'key' });
                }
            };
            req.onsuccess = (e) => {
                this._db = e.target.result;
                resolve(this._db);
            };
            req.onerror = (e) => reject(e.target.error);
        });
    },

    async _tx(storeName, mode = 'readonly') {
        const db = await this.open();
        return db.transaction(storeName, mode).objectStore(storeName);
    },

    async _promisify(request) {
        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    },

    // ===== Sample CRUD =====

    async getAllSamples() {
        const store = await this._tx(this.STORE_SAMPLES);
        return this._promisify(store.getAll());
    },

    async getSample(id) {
        const store = await this._tx(this.STORE_SAMPLES);
        return this._promisify(store.get(id));
    },

    async putSample(sample) {
        sample.updatedAt = Date.now();
        const store = await this._tx(this.STORE_SAMPLES, 'readwrite');
        return this._promisify(store.put(sample));
    },

    async deleteSample(id) {
        const store = await this._tx(this.STORE_SAMPLES, 'readwrite');
        return this._promisify(store.delete(id));
    },

    async clearAllSamples() {
        const store = await this._tx(this.STORE_SAMPLES, 'readwrite');
        return this._promisify(store.clear());
    },

    async getSampleCount() {
        const store = await this._tx(this.STORE_SAMPLES);
        return this._promisify(store.count());
    },

    /**
     * Add a new sample from an uploaded MP3 file.
     */
    async addFromFile(file) {
        const blob = file.slice ? file : new Blob([file]);
        const stem = Utils.sanitizeStem(file.name);
        const duration = await Utils.getAudioDuration(blob);

        const sample = {
            id: Utils.generateId(),
            filename: stem,
            audioBlob: blob,
            audioDuration: duration,
            prompt: '',
            lyrics: '',
            isInstrumental: false,
            isValid: false,
            createdAt: Date.now(),
            updatedAt: Date.now(),
        };

        await this.putSample(sample);
        return sample;
    },

    /**
     * Add a sample from ZIP extraction (with optional pre-filled prompt/lyrics).
     */
    async addFromZipEntry(stem, audioBlob, prompt, lyrics) {
        const duration = await Utils.getAudioDuration(audioBlob);

        const sample = {
            id: Utils.generateId(),
            filename: stem,
            audioBlob: audioBlob,
            audioDuration: duration,
            prompt: prompt || '',
            lyrics: lyrics || '',
            isInstrumental: false,
            isValid: false,
            createdAt: Date.now(),
            updatedAt: Date.now(),
        };

        // Compute validity
        sample.isValid = this.checkValidity(sample);

        await this.putSample(sample);
        return sample;
    },

    checkValidity(sample) {
        if (!sample.audioBlob) return false;
        if (!sample.filename || sample.filename.trim() === '') return false;
        if (!sample.prompt || sample.prompt.trim() === '') return false;
        if (!sample.isInstrumental && (!sample.lyrics || sample.lyrics.trim() === '')) return false;
        return true;
    },

    async revalidateAll() {
        const samples = await this.getAllSamples();
        for (const s of samples) {
            const valid = this.checkValidity(s);
            if (s.isValid !== valid) {
                s.isValid = valid;
                await this.putSample(s);
            }
        }
        return samples;
    },

    // ===== Metadata =====

    async getMeta(key) {
        const store = await this._tx(this.STORE_META);
        const result = await this._promisify(store.get(key));
        return result ? result.value : null;
    },

    async setMeta(key, value) {
        const store = await this._tx(this.STORE_META, 'readwrite');
        return this._promisify(store.put({ key, value }));
    },

    // ===== Stats =====

    async getStats() {
        const samples = await this.getAllSamples();
        const valid = samples.filter(s => s.isValid);
        const totalDuration = samples.reduce((acc, s) => acc + (s.audioDuration || 0), 0);
        const missingPrompt = samples.filter(s => !s.prompt || s.prompt.trim() === '');
        const missingLyrics = samples.filter(s => !s.isInstrumental && (!s.lyrics || s.lyrics.trim() === ''));
        const duplicateNames = this._findDuplicates(samples.map(s => s.filename));

        return {
            total: samples.length,
            validCount: valid.length,
            invalidCount: samples.length - valid.length,
            totalDurationSec: totalDuration,
            avgDurationSec: samples.length > 0 ? totalDuration / samples.length : 0,
            missingPrompt: missingPrompt.map(s => s.filename),
            missingLyrics: missingLyrics.map(s => s.filename),
            duplicateNames,
            isReady: valid.length === samples.length && samples.length > 0 && duplicateNames.length === 0,
        };
    },

    _findDuplicates(arr) {
        const seen = {};
        const dupes = [];
        for (const item of arr) {
            if (seen[item]) dupes.push(item);
            else seen[item] = true;
        }
        return [...new Set(dupes)];
    },
};

window.DB = DB;
