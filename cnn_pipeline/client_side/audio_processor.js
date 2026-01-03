// This AudioWorkletProcessor now accumulates audio data in a buffer
// and sends chunks to the main thread at regular intervals (hop size)
// to support the sliding window approach.

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        console.log('✅ AudioProcessor initialized on audio thread.');

        this.isRecording = false;
        
        // This is a single, continuous buffer to store all incoming audio
        this.buffer = new Float32Array(0);

        // Properties for the sliding window, which will be configured by the main thread
        this.windowSize = 0;
        this.hopSize = 0;
        this.samplesPerWindow = 0;
        this.samplesPerHop = 0;
        this.currentOffset = 0;
        
        // Listen for messages from the main thread to control recording and configuration
        this.port.onmessage = (event) => {
            if (event.data.type === 'START_SLIDING_WINDOW') {
                this.isRecording = true;
                // Configure the window and hop sizes based on the message from the main thread
                this.windowSize = event.data.windowSize;
                this.hopSize = event.data.hopSize;
                this.samplesPerWindow = Math.floor(this.windowSize * sampleRate);
                this.samplesPerHop = Math.floor(this.hopSize * sampleRate);
                this.buffer = new Float32Array(0); // Reset buffer for a new session
                this.currentOffset = 0; // Reset offset
                console.log(`🎙️ Recording started with window: ${this.windowSize}s, hop: ${this.hopSize}s.`);
            } else if (event.data.type === 'STOP_RECORDING') {
                this.isRecording = false;
                console.log('🛑 Recording stopped on audio thread.');
                // At the end, send any remaining audio in the buffer that hasn't been sent as a full hop
                if (this.buffer.length - this.currentOffset >= this.samplesPerWindow) {
                    const chunk = this.buffer.slice(this.currentOffset, this.currentOffset + this.samplesPerWindow);
                    this.port.postMessage({ type: 'AUDIO_CHUNK', data: chunk.buffer }, [chunk.buffer]);
                }
            }
        };
    }

    // The process method is called by the browser to process audio
    process(inputs, outputs) {
        if (!this.isRecording) {
            return true;
        }

        const input = inputs[0];
        const inputChannelData = input[0];

        // If there's no input data, just return
        if (!inputChannelData) {
            return true;
        }

        // Append the new audio data to our continuous buffer
        const newBuffer = new Float32Array(this.buffer.length + inputChannelData.length);
        newBuffer.set(this.buffer);
        newBuffer.set(inputChannelData, this.buffer.length);
        this.buffer = newBuffer;

        // Check if we have enough data to send a new chunk (one full window)
        // This check runs on a "hop" basis
        while (this.buffer.length - this.currentOffset >= this.samplesPerWindow) {
            const chunk = this.buffer.slice(this.currentOffset, this.currentOffset + this.samplesPerWindow);
            this.port.postMessage({ type: 'AUDIO_CHUNK', data: chunk.buffer }, [chunk.buffer]);
            
            // Move the offset forward by the hop size
            this.currentOffset += this.samplesPerHop;
        }
        
        // Return true to keep the AudioWorkletNode active
        return true;
    }
}

// Register the AudioWorkletProcessor with a unique name
registerProcessor('audio_processor', AudioProcessor);
console.log('✅ audio_processor registered.');