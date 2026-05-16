# PISIGHT – Real-Time Vision & Voice IoT Assistant
deployment link:https://pi-sight-project.onrender.com

## Overview

PISIGHT is an AI-powered real-time IoT assistant designed to combine:

*  Computer Vision
*  Voice Interaction
*  Generative AI
*  Real-Time Speech Response
*  IoT Communication

The system enables users to interact with the environment using voice commands and image-based context. It can analyze surroundings, understand spoken queries, process images in real time, and generate intelligent spoken responses.

PISIGHT is designed for assistive technology, smart surveillance, accessibility, and intelligent automation applications.

---

# Key Features

## 🎙️ Real-Time Voice Interaction

* Captures user voice input
* Converts speech to text using AssemblyAI
* Supports continuous conversational interaction
* Processes audio in real time through WebSockets

---

## 👁️ AI Vision Processing

* Accepts image uploads from IoT devices or frontend
* Maintains image context during conversation
* Enables scene understanding and contextual questioning
* Supports real-time visual assistance

---

## 🤖 Generative AI Intelligence

Powered using Google Gemini AI models.

Capabilities include:

* Visual question answering
* Context-aware responses
* Conversational assistance
* Intelligent scene interpretation
* Real-time AI reasoning

---

## 🔊 AI Voice Responses

* Converts AI-generated text into speech
* Uses Gemini Text-to-Speech
* Returns WAV audio buffers for frontend playback
* Enables natural conversational experience

---

## 🌐 Real-Time Communication

Uses Socket.IO for:

* Live voice streaming
* Real-time image transfer
* Instant AI responses
* Bidirectional communication

---

## ☁️ Cloud Deployment

* Backend deployed on Render
* Frontend served through Express
* Fully cloud-accessible architecture
* Supports IoT device integration

---

# Technology Stack

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| Node.js          | Backend Runtime         |
| Express.js       | Web Server              |
| Socket.IO        | Real-Time Communication |
| AssemblyAI       | Speech-to-Text          |
| Google Gemini AI | AI Processing + TTS     |
| HTML/CSS/JS      | Frontend Interface      |
| Render           | Cloud Deployment        |

---

# System Architecture

## Workflow

1. User speaks into microphone
2. Audio is transmitted to backend
3. AssemblyAI converts speech to text
4. Image context is attached if available
5. Gemini AI processes the request
6. AI generates intelligent response
7. Gemini TTS converts response to audio
8. Audio response is streamed back to user

---

# Folder Structure

```text
project/
│
├── public/
│   └── index.html
│
├── src/
│   ├── controllers/
│   │   └── aiAgent.controller.js
│   │
│   └── server.js
│
├── package.json
├── package-lock.json
└── .env
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd pisight
```

---

## Install Dependencies

```bash
npm install
```

---

# Environment Variables

Create a `.env` file:

```env
PORT=5000
STT_API_KEY=your_assemblyai_key
GOOGLE_API_KEY=your_google_gemini_key
DEBUG_SAVE_AUDIO=false
```

---

# Running Locally

## Start Development Server

```bash
npm run dev
```

## Start Production Server

```bash
npm start
```

---

# Deployment

## Render Deployment

### Build Command

```bash
npm install
```

### Start Command

```bash
npm start
```

---

# Socket Events

## Client → Server

| Event        | Description             |
| ------------ | ----------------------- |
| audio_full   | Sends full audio buffer |
| image_chunk  | Uploads image chunks    |
| text_message | Sends text query        |
| clear_image  | Clears image context    |

---

## Server → Client

| Event          | Description              |
| -------------- | ------------------------ |
| ai_response    | AI text + audio response |
| image_received | Image upload success     |
| image_cleared  | Clears stored image      |
| error          | Error handling           |

---

# Use Cases

## 👨‍🦯 Assistive Technology

Helps visually impaired users understand surroundings using:

* Scene descriptions
* Object understanding
* Voice interaction

---

## 🏠 Smart IoT Assistant

Can integrate with:

* Smart cameras
* Raspberry Pi devices
* IoT sensors
* Embedded systems

---

## 🛡️ Intelligent Surveillance

Supports:

* Real-time monitoring
* Scene analysis
* Event-based questioning

---

## 🎓 Educational AI System

Can be used for:

* AI demonstrations
* Hackathons
* IoT showcases
* Research prototypes

---

# Future Improvements

Potential enhancements include:

* Live video streaming support
* Edge AI deployment
* Mobile application integration
* Multilingual voice support
* Object detection models
* Offline processing mode
* Raspberry Pi hardware integration
* AI memory and personalization

---

# Advantages

* Real-time interaction
* Low-latency communication
* AI-powered contextual understanding
* Cloud deployable
* Scalable architecture
* IoT-ready framework
* Voice + Vision fusion system

---

# Conclusion

PISIGHT demonstrates how modern AI technologies can be integrated with IoT systems to create intelligent real-time assistants capable of understanding both voice and visual information.

By combining computer vision, speech recognition, generative AI, and real-time communication, the system provides an advanced interactive experience suitable for assistive, automation, and smart environment applications.
