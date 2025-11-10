import express from "express";
import { createServer } from "http";
import dotenv from "dotenv";
import { Server } from "socket.io";
import { AssemblyAI } from "assemblyai";
import { GoogleGenAI } from "@google/genai";
import wav from "wav";
import { Writable } from "stream"; // Essential for in-memory buffer handling

import { AiProcessing } from "./controllers/aiAgent.controller.js";

dotenv.config();

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
  maxHttpBufferSize: 50e6, // 50MB in case large audio/images
});

const PORT = process.env.PORT || 5000;

// AssemblyAI client
const assemblyClient = new AssemblyAI({
  apiKey: process.env.STT_API_KEY,
});

// Google AI client
const googleAI = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY,
});

// ----------------------------
// 🧠 TTS Helper Functions
// ----------------------------

// Utility to write to a Buffer in memory
class BufferWritable extends Writable {
  constructor(options) {
    super(options);
    this.chunks = [];
  }
  _write(chunk, encoding, callback) {
    this.chunks.push(chunk);
    callback();
  }
  getBuffer() {
    return Buffer.concat(this.chunks);
  }
}

/**
 * Wraps raw PCM data with a proper WAV header in memory.
 * This fixes the frontend "Unable to decode audio data" error.
 * @param {Buffer} pcmData - Raw PCM audio buffer from Gemini TTS.
 * @returns {Promise<Buffer>} - The complete WAV file buffer.
 */
async function getWaveBuffer(
  pcmData,
  channels = 1,
  rate = 24000,
  sampleWidth = 2
) {
  return new Promise((resolve, reject) => {
    const bufferStream = new BufferWritable();
    
    // CORRECTED: Using wav.Writer (stream) instead of wav.FileWriter (file system)
    const writer = new wav.Writer({
      channels,
      sampleRate: rate,
      bitDepth: sampleWidth * 8,
    });

    writer.on("finish", () => resolve(bufferStream.getBuffer()));
    writer.on("error", reject);

    writer.pipe(bufferStream); 
    writer.write(pcmData);
    writer.end();
  });
}

// 🎙️ Gemini Text-to-Speech
async function textToAudio(text) {
  try {
    const response = await googleAI.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: "Kore" },
          },
        },
      },
    });

    const data =
      response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!data) throw new Error("No audio data returned from Gemini API.");

    // This is the raw PCM data buffer (missing WAV header)
    const rawPcmBuffer = Buffer.from(data, "base64");

    // Wrap the raw PCM data with a WAV header in memory
    const waveBuffer = await getWaveBuffer(rawPcmBuffer);

    // Optional: Save for debugging (requires your original saveWaveFile utility)
    if (process.env.DEBUG_SAVE_AUDIO === "true") {
      const fileName = `tts_output_${Date.now()}.wav`;
      await saveWaveFile(fileName, rawPcmBuffer);
      console.log("TTS audio saved:", fileName);
    }

    // Return the complete WAV buffer
    return waveBuffer;
  } catch (error) {
    console.error("Error in textToAudio:", error);
    throw error;
  }
}

// Helper to save WAV file (only needed if DEBUG_SAVE_AUDIO is true)
async function saveWaveFile(
  filename,
  pcmData,
  channels = 1,
  rate = 24000,
  sampleWidth = 2
) {
  return new Promise((resolve, reject) => {
    const writer = new wav.FileWriter(filename, {
      channels,
      sampleRate: rate,
      bitDepth: sampleWidth * 8,
    });

    writer.on("finish", resolve);
    writer.on("error", reject);

    writer.write(pcmData);
    writer.end();
  });
}

// ----------------------------
// 🔊 Express Routes
// ----------------------------
app.get("/", (req, res) => {
  res.json({
    status: "online",
    service: "PISIGHT Backend",
    timestamp: new Date().toISOString(),
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    connections: io.engine.clientsCount,
  });
});

// ----------------------------
// 🔌 Socket.IO Connection
// ----------------------------
io.on("connection", (socket) => {
  console.log(`[${new Date().toISOString()}] Device connected: ${socket.id}`);

  const sessionData = {
    image: null,
    imageChunks: [],
    isProcessing: false,
  };

  // ------------------------
  // Full audio handler
  // ------------------------
  socket.on("audio_full", async (arrayBuffer) => {
    if (sessionData.isProcessing) {
      socket.emit("error", {
        type: "busy",
        message: "Still processing previous request",
      });
      return;
    }
    sessionData.isProcessing = true;

    try {
      const audioBuffer = Buffer.from(arrayBuffer);
      console.log(
        `[${socket.id}] Received full audio: ${audioBuffer.length} bytes`
      );
      
      // ... (AssemblyAI Transcription Logic) ...
      const uploadResponse = await assemblyClient.files.upload(audioBuffer);
      console.log(uploadResponse);

      const transcriptResponse = await assemblyClient.transcripts.create({
        audio_url: uploadResponse,
      });

      let completedTranscript;
      while (true) {
        completedTranscript = await assemblyClient.transcripts.get(
          transcriptResponse.id
        );
        if (completedTranscript.status === "completed") break;
        if (completedTranscript.status === "error")
          throw new Error(completedTranscript.error);
        await new Promise((r) => setTimeout(r, 3000));
      }

      const transcript = completedTranscript.text || "";
      console.log(`[${socket.id}] Transcription:`, transcript);

      // Process with AI
      const aiResponseText = await AiProcessing(sessionData.image, transcript);
      console.log(`[${socket.id}] AI Response:`, aiResponseText);

      // Convert AI response to audio (now returns WAV BUFFER)
      const aiAudioBuffer = await textToAudio(aiResponseText);

      socket.emit("ai_response", {
        text: aiResponseText,
        audio: aiAudioBuffer.toString("base64"),
        timestamp: Date.now(),
      });
    } catch (err) {
      console.error(`[${socket.id}] Error processing full audio:`, err);
      socket.emit("error", { type: "full_audio", message: err.message });
    } finally {
      sessionData.isProcessing = false;
    }
  });

  // ------------------------
  // Image upload (chunked)
  // ------------------------
  socket.on("image_chunk", (data) => {
    try {
      const bufferChunk = Buffer.from(data.chunk);
      sessionData.imageChunks.push(bufferChunk);

      if (data.isLast) {
        sessionData.image = Buffer.concat(sessionData.imageChunks);
        sessionData.imageChunks = [];
        console.log(
          `[${socket.id}] Image received: ${sessionData.image.length} bytes`
        );
        socket.emit("image_received", {
          size: sessionData.image.length,
          timestamp: Date.now(),
        });
      }
    } catch (err) {
      console.error(`[${socket.id}] Error processing image:`, err);
      socket.emit("error", { type: "image_upload", message: err.message });
    }
  });

  // ------------------------
  // Text messages
  // ------------------------
  socket.on("text_message", async (message) => {
    if (sessionData.isProcessing) {
      socket.emit("error", {
        type: "busy",
        message: "Still processing previous request",
      });
      return;
    }

    sessionData.isProcessing = true;
    try {
      const aiResponseText = await AiProcessing(sessionData.image, message);
      // Convert AI response to audio (now returns WAV BUFFER)
      const aiAudioBuffer = await textToAudio(aiResponseText);

      socket.emit("ai_response", {
        text: aiResponseText,
        audio: aiAudioBuffer.toString("base64"),
        timestamp: Date.now(),
      });
    } catch (err) {
      socket.emit("error", { type: "text_processing", message: err.message });
    } finally {
      sessionData.isProcessing = false;
    }
  });

  // ------------------------
  // Clear image
  // ------------------------
  socket.on("clear_image", () => {
    sessionData.image = null;
    socket.emit("image_cleared");
  });

  // ------------------------
  // Disconnect
  // ------------------------
  socket.on("disconnect", () => {
    console.log(`[${socket.id}] Device disconnected`);
    sessionData.image = null;
    sessionData.imageChunks = [];
  });
});

// ----------------------------
// Error middleware
// ----------------------------
app.use((err, req, res, next) => {
  console.error("Express error:", err);
  res.status(500).json({ error: "Internal server error" });
});

// ----------------------------
// Graceful shutdown
// ----------------------------
process.on("SIGTERM", () => {
  console.log("SIGTERM received, closing server...");
  server.close(() => {
    console.log("Server closed");
    process.exit(0);
  });
});

server.listen(PORT, () => {
  console.log(`🚀 PISIGHT Backend running on port ${PORT}`);
  console.log(`📡 Socket.IO ready for connections`);
});