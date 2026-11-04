/**
 * Guppy Bowl · TARS Voice Engine
 * 100% serverless, in-browser neural text-to-speech powered by Piper WASM
 * with Interstellar spacecraft intercom DSP (bandpass + analog saturation).
 */

export class GuppyVoice {
  constructor() {
    this.enabled = true;
    this.voiceId = 'en_US-hfc_male-medium'; // TARS deadpan US male
    this.session = null;
    this.audioCtx = null;
    this.currentSource = null;
    this.isSpeaking = false;
    this.loading = false;
    this.subcomms = true;  // Interstellar spacecraft intercom filter
    this.drive = 8;
    this.pitch = 1.0;
    this.speed = 1.0;
    this.currentRequestId = 0;
  }

  async init() {
    if (this.session || this.loading) return;
    this.loading = true;
    try {
      const { TtsSession } = await import('https://cdn.jsdelivr.net/npm/@realtimex/piper-tts-web/+esm');
      this.session = await TtsSession.create({
        voiceId: this.voiceId,
        allowLocalModels: true,
        fallbackStrategy: 'auto'
      });
    } catch (err) {
      console.warn('Guppy voice preload failed:', err);
    } finally {
      this.loading = false;
    }
  }

  toggle() {
    this.enabled = !this.enabled;
    if (!this.enabled) this.stop();
    return this.enabled;
  }

  toggleSubcomms() {
    this.subcomms = !this.subcomms;
    return this.subcomms;
  }

  stop() {
    this.currentRequestId++; // Invalidate any pending synthesis
    if (this.currentSource) {
      try { this.currentSource.stop(); } catch (e) {}
      this.currentSource = null;
    }
    this.isSpeaking = false;
  }

  _makeDistortionCurve(amount) {
    if (amount <= 0) return null;
    const k = amount;
    const n = 44100;
    const curve = new Float32Array(n);
    const deg = Math.PI / 180;
    for (let i = 0; i < n; ++i) {
      const x = (i * 2) / n - 1;
      curve[i] = ((3 + k) * x * 20 * deg) / (Math.PI + k * Math.abs(x));
    }
    return curve;
  }

  _formatForSpeech(raw) {
    if (!raw) return '';
    let text = raw.replace(/<[^>]*>/g, '').trim();
    // Ensure standard space after punctuation so eSpeak-ng phonemizer doesn't merge words across stops
    text = text.replace(/([.,!?;:])(?=[^\s])/g, '$1 ');
    // Capitalize sentence starts so the sentence boundary detector pauses at full stops and commas
    text = text.replace(/(^\s*|[.!?]\s+)([a-z])/g, (_, p, c) => p + c.toUpperCase());
    return text;
  }

  async speak(rawText) {
    if (!this.enabled || !rawText) return;
    const text = this._formatForSpeech(rawText);
    if (!text) return;

    // Immediately stop any currently playing voice and claim a new request ID
    this.stop();
    const reqId = this.currentRequestId;

    if (!this.session) {
      await this.init();
      if (!this.session || reqId !== this.currentRequestId) return;
    }

    try {
      const wavBlob = await this.session.predict(text);

      // Discard if a newer message was triggered while synthesizing
      if (reqId !== this.currentRequestId) return;

      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this.audioCtx.state === 'suspended') {
        try { await this.audioCtx.resume(); } catch (e) {}
      }

      // If browser autoplay policy blocked audio because user hasn't touched the page yet,
      // silently drop this initial message so it does NOT queue up and overlap later!
      if (this.audioCtx.state === 'suspended') {
        return;
      }

      if (reqId !== this.currentRequestId) return;

      const arrayBuffer = await wavBlob.arrayBuffer();
      const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);

      if (reqId !== this.currentRequestId) return;

      // Stop again right before starting to prevent race conditions
      if (this.currentSource) {
        try { this.currentSource.stop(); } catch (e) {}
      }

      this.currentSource = this.audioCtx.createBufferSource();
      this.currentSource.buffer = audioBuffer;
      this.currentSource.playbackRate.value = this.speed;

      let lastNode = this.currentSource;

      if (this.subcomms) {
        // TARS Spacecraft Intercom DSP Chain (tuned for clarity & warmth):
        const hp = this.audioCtx.createBiquadFilter();
        hp.type = 'highpass';
        hp.frequency.value = 220; // Preserve vocal body

        const lp = this.audioCtx.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency.value = 4800; // Preserve consonant clarity

        const mid = this.audioCtx.createBiquadFilter();
        mid.type = 'peaking';
        mid.frequency.value = 1600;
        mid.Q.value = 1.2;
        mid.gain.value = 3.5;

        const shaper = this.audioCtx.createWaveShaper();
        shaper.curve = this._makeDistortionCurve(this.drive);
        shaper.oversample = '4x';

        lastNode.connect(hp);
        hp.connect(lp);
        lp.connect(mid);
        mid.connect(shaper);
        lastNode = shaper;
      }

      lastNode.connect(this.audioCtx.destination);

      this.isSpeaking = true;
      this.currentSource.onended = () => {
        if (reqId === this.currentRequestId) {
          this.isSpeaking = false;
          this.currentSource = null;
        }
      };

      this.currentSource.start(0);
    } catch (e) {
      console.warn('Guppy voice synthesis error:', e);
      this.isSpeaking = false;
    }
  }
}
