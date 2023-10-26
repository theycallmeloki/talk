
const pythonCode = `
import click
import os
import time
import webrtcvad
from transformers import pipeline
import torch
import base64


# Global variables
vad = webrtcvad.Vad(1)  # Sensitivity level: 1
model_path = 'openai/whisper-large-v2'
device = 'cuda:0'
dtype = torch.float32
asr_pipeline = pipeline("automatic-speech-recognition",
                        model=model_path,
                        device=device,
                        torch_dtype=dtype)

def init_model(model_path='openai/whisper-large-v2'):
    global asr_pipeline
    asr_pipeline = pipeline("automatic-speech-recognition",
                            model=model_path,
                            device=device,
                            torch_dtype=dtype)

def vad_function(audio_buffer):
    """Detects voice activity in the audio buffer."""
    
    # Logging information about the buffer
    print("VAD Audio Buffer Type:", type(audio_buffer))
    print(f"Truncated Audio Buffer (Python): {audio_buffer[:10]}")

    # Decoding the audio buffer from base64
    decoded_buffer = base64.b64decode(audio_buffer)
    print(f"Decoded Buffer Length (Python): {len(decoded_buffer)}")
    print(f"Truncated Decoded Buffer (Python): {decoded_buffer[:10]}")
    
    return vad.is_speech(decoded_buffer, sample_rate=16000)

def asr_inference(audio_buffer):
    """Performs ASR on the audio buffer."""
    
    # Logging information about the buffer
    print("ASR Audio Buffer Type:", type(audio_buffer))
    print(f"Truncated Audio Buffer (Python): {audio_buffer[:10]}")
    
    # Decoding the audio buffer from base64
    decoded_buffer = base64.b64decode(audio_buffer)
    print(f"Decoded Buffer Length (Python): {len(decoded_buffer)}")
    print(f"Truncated Decoded Buffer (Python): {decoded_buffer[:10]}")
    
    return asr_pipeline(decoded_buffer)


def seconds_to_srt_time_format(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

@click.command()
@click.option('--model', default='openai/whisper-base', help='ASR model to use for speech recognition. Default is "openai/whisper-base".')
@click.option('--device', default='cuda:0', help='Device to use for computation. Default is "cuda:0". If you want to use CPU, specify "cpu".')
@click.option('--dtype', default='float32', help='Data type for computation. Can be either "float32" or "float16". Default is "float32".')
@click.option('--batch-size', type=int, default=8, help='Batch size for processing. Default is 8.')
@click.option('--chunk-length', type=int, default=30, help='Length of audio chunks to process at once. Default is 30 seconds.')
@click.argument('audio_file', type=str)
def cli_asr(model, device, dtype, batch_size, chunk_length, audio_file):
    init_model(model)

    # Perform ASR
    print("Model loaded.")
    start_time = time.perf_counter()
    outputs = asr_inference(audio_file)  # NOTE: This is a placeholder. You'll need to adapt for buffers or reading files directly.

    # Output the results
    print(outputs)
    print("Transcription complete.")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"ASR took {elapsed_time:.2f} seconds.")

    # Save ASR chunks to an SRT file
    audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    srt_filename = f"{audio_file_name}.srt"
    with open(srt_filename, 'w') as srt_file:
        for index, chunk in enumerate(outputs['chunks']):
            start_time = seconds_to_srt_time_format(chunk['timestamp'][0])
            end_time = seconds_to_srt_time_format(chunk['timestamp'][1])
            srt_file.write(f"{index + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{chunk['text'].strip()}\n\n")

if __name__ == '__main__':
    cli_asr()

`;


import { spawn } from 'child_process';
import readline from 'readline';
import config from './config.json';
const { audioListenerScript } = config;
import { talk } from './src/talk';
import { Mutex } from './src/depedenciesLibrary/mutex';
import { pythonBridge } from 'python-bridge';

let python = pythonBridge({
  python: 'python3'
});
const directoryOfIndexTs = __dirname;
python.ex`import sys`;
python.ex`import numpy as np`
python.ex`sys.path.append(${directoryOfIndexTs})`;


const fs = require('fs');
const path = require('path');

// CONSTANTS
const SAMPLING_RATE = 16000;
const CHANNELS = 1;
const BIT_DEPTH = 16;
const ONE_SECOND = SAMPLING_RATE * (BIT_DEPTH / 8) * CHANNELS;
const BUFFER_LENGTH_SECONDS = 28;
const BUFFER_LENGTH_MS = BUFFER_LENGTH_SECONDS * 1000;
const VAD_ENABLED = config.voiceActivityDetectionEnabled;
const INTERRUPTION_ENABLED = config.interruptionEnabled;
const INTERRUPTION_LENGTH_CHARS = 20;
const VAD_BUFFER_SIZE = 8;

const DEFAULT_LLAMA_SERVER_URL = 'http://127.0.0.1:8080'
const MAX_DIALOGUES_IN_CONTEXT = 10;
const GRAPH_FILE = 'talk.dot';

let llamaServerUrl: string = DEFAULT_LLAMA_SERVER_URL;

if ('llamaServerUrl' in config) {
  llamaServerUrl = config.llamaServerUrl as string;
}

(async () => {
  (python`print(${"Booting up python..."})` as any);
  await initializeASR();
})();


const DEFAULT_PROMPT = "Continue the dialogue, speak for bob only. \nMake it a fun lighthearted conversation."

let conversationPrompt: string = DEFAULT_PROMPT;
let personaConfig: string = "";
if ('personaFile' in config) {
  const personaFilePath = path.resolve(config.personaFile);
  if (fs.existsSync(personaFilePath)) {
    personaConfig = fs.readFileSync(personaFilePath, 'utf8');
    conversationPrompt = "";
  }
}

// INTERFACES
type EventType = 'audioBytes' | 'responseReflex' | 'transcription' | 'cutTranscription' | 'talk' | 'interrupt' | 'responseInput';
interface Event {
  eventType: EventType;
  timestamp: number;
  data: { [key: string]: any };
}
interface AudioBytesEvent extends Event {
  eventType: 'audioBytes';
  data: {
    buffer: {
      raw: string;
    };
  }
}
interface ResponseReflexEvent extends Event {
  eventType: 'responseReflex';
  data: {
    transcription: string
  }
}
interface TranscriptionEvent extends Event {
  eventType: 'transcription';
  data: {
    buffer: Buffer;
    transcription: string;
    lastAudioByteEventTimestamp: number;
  }
}
interface CutTranscriptionEvent extends Event {
  eventType: 'cutTranscription';
  data: {
    buffer: Buffer;
    transcription: string;
    lastAudioByteEventTimestamp: number;
  }
}
interface TalkEvent extends Event {
  eventType: 'talk';
  data: {
    response: string;
  }
}
interface ResponseInputEvent extends Event {
  eventType: 'responseInput',
  data: {}
}
interface InterruptEvent extends Event {
  eventType: 'interrupt';
  data: {
    streamId: string;
  }
}
interface EventLog {
  events: Event[];
}
const eventlog: EventLog = {
  events: []
};
type AudioBufferFormat = {
  raw: string;
};


// asr hot replacement module section

async function initializeASR(modelPath = 'openai/whisper-base') {
  await python.ex`from local_whisper import init_model`;
  await python`init_model(${modelPath})`;
}

async function voiceActivityDetection(audioBuffer: any) {
  console.log("Keys in audioBuffer:", Object.keys(audioBuffer));
  if (audioBuffer.raw) {
    const truncatedBuffer = audioBuffer.raw.slice(0, 10);
    console.log("Truncated audioBuffer.raw:", truncatedBuffer);
  } else {
    console.error("Error: audioBuffer does not contain 'raw' key");
  }
  await python.ex`from local_whisper import vad_function`;

  // Check if audioBuffer has 'raw' key
  if (audioBuffer && Buffer.isBuffer(audioBuffer)) {
    const base64EncodedBuffer = audioBuffer.toString('base64');
    return await python`vad_function(${base64EncodedBuffer})`;
  } else {
    console.error("Error: audioBuffer is not a valid buffer");
    return false;
  }
}

async function asrInference(audioBuffer: any) {
  console.log("Keys in audioBuffer:", Object.keys(audioBuffer));
  if (audioBuffer.raw) {
    const truncatedBuffer = audioBuffer.raw.slice(0, 10);
    console.log("Truncated audioBuffer.raw:", truncatedBuffer);
  } else {
    console.error("Error: audioBuffer does not contain 'raw' key");
  }
  await python.ex`from local_whisper import asr_inference`;

  // Convert audioBuffer to numpy ndarray if it's a dictionary with 'raw' key
  if (audioBuffer && Buffer.isBuffer(audioBuffer)) {
    return await python`

      buffer_np = np.frombuffer(${audioBuffer}, dtype=np.int16)
      asr_inference(buffer_np)
    `;
  } else {
    console.error("Error: audioBuffer is not a valid buffer");
    return "";
  }
}



async function endASR() {
  await python.end();
}

// EVENTLOG UTILITY FUNCTIONS
// From the event log, get the transcription so far
const getLastTranscriptionEvent = (): TranscriptionEvent => {
  const transcriptionEvents = eventlog.events.filter(e => e.eventType === 'transcription');
  return transcriptionEvents[transcriptionEvents.length - 1] as TranscriptionEvent;
}

const getLastResponseReflexTimestamp = (): number => {
  const responseReflexEvents = eventlog.events.filter(e => e.eventType === 'responseReflex');
  return responseReflexEvents.length > 0 ? responseReflexEvents[responseReflexEvents.length - 1].timestamp : eventlog.events[0].timestamp;
};

const getCutTimestamp = (): number => {
  const cutTranscriptionEvents = eventlog.events.filter(e => e.eventType === 'cutTranscription');
  const lastCut = cutTranscriptionEvents.length > 0 ? cutTranscriptionEvents[cutTranscriptionEvents.length - 1].data.lastAudioByteEventTimestamp : eventlog.events[0].timestamp;
  const lastResponseReflex = getLastResponseReflexTimestamp();
  return Math.max(lastResponseReflex, lastCut);
}

const getTransciptionSoFar = (): string => {
  const lastResponseReflex = getLastResponseReflexTimestamp();
  const cutTranscriptionEvents = eventlog.events.filter(e => e.eventType === 'cutTranscription' && e.timestamp > lastResponseReflex);
  const lastTranscriptionEvent = getLastTranscriptionEvent();
  const lastCutTranscriptionEvent = cutTranscriptionEvents[cutTranscriptionEvents.length - 1];
  let transcription = cutTranscriptionEvents.map(e => e.data.transcription).join(' ');
  if (!lastCutTranscriptionEvent || lastCutTranscriptionEvent.timestamp !== lastTranscriptionEvent.timestamp) {
    transcription = transcription + (lastTranscriptionEvent?.data?.transcription || '')
  }
  return transcription
}

const getDialogue = (): string => {
  const dialogueEvents = eventlog.events
    .filter(e => e.eventType === 'responseReflex' || e.eventType === 'talk');

  let result = [];
  let lastType = null;
  let mergedText = '';

  for (let e of dialogueEvents) {
    const currentSpeaker = e.eventType === 'responseReflex' ? 'alice' : 'bob';
    const currentText = e.eventType === 'responseReflex' ? e.data.transcription : e.data.response;

    if (lastType && lastType === currentSpeaker) {
      mergedText += ' ' + currentText;
    } else {
      if (mergedText) result.push(mergedText);
      mergedText = `${currentSpeaker}: ${currentText}`;
    }

    lastType = currentSpeaker;
  }

  // push last merged text
  if (mergedText) result.push(mergedText);

  if (result.length > MAX_DIALOGUES_IN_CONTEXT) {
    result = result.slice(-MAX_DIALOGUES_IN_CONTEXT);
  }

  return result.join('\n');
}

// const updateScreenEvents: Set<EventType> = new Set([])
const updateScreenEvents: Set<EventType> = new Set(['responseReflex', 'cutTranscription', 'talk', 'interrupt']);
const updateScreen = (event: Event) => {
  if (updateScreenEvents.has(event.eventType)) {
    console.log(getDialogue());
    console.log(event);
  }
}

// Graphviz
const graph = ["digraph G {"];

const updateGraphEvents: Set<EventType> = new Set(['responseInput', 'responseReflex', 'audioBytes', 'transcription', 'cutTranscription', 'talk', 'interrupt']);
const updateGraph = (event: Event, prevEvent: void | Event) => {
  let label = event.eventType;
  if (event.data?.transcription) {
    label += `: ${event.data.transcription}`;
  } else if (event.data?.response) {
    label += `: ${event.data.response}`;
  }
  graph.push(`    ${event.eventType}${event.timestamp} [label="${label}"]`)
  if (prevEvent?.eventType && updateGraphEvents.has(event.eventType) && updateGraphEvents.has(prevEvent.eventType)) {
    graph.push(`    ${prevEvent.eventType}${prevEvent.timestamp} -> ${event.eventType}${event.timestamp}`);
  }
}
const writeGraph = () => {
  graph.push('}');
  fs.writeFileSync(GRAPH_FILE, '');
  for (let line in graph) {
    fs.appendFileSync(GRAPH_FILE, `${graph[line]}\n`, 'utf8');
  }
}

// EVENTS
const newEventHandler = (event: Event, prevEvent: void | Event): void => {
  eventlog.events.push(event);
  updateScreen(event);
  updateGraph(event, prevEvent);
  const downstreamEvents = eventDag[event.eventType];
  for (const downstreamEvent in downstreamEvents) {
    const downstreamEventFn = downstreamEvents[downstreamEvent as EventType];
    // Note: Unecessary existence check, this is typesafe
    if (downstreamEventFn) {
      downstreamEventFn(event);
    }
  }
}

const newAudioBytesEvent = (buffer: Buffer): void => {
  const truncatedBufferNode = buffer.slice(0, 10);
  console.log("Buffer Length (Node.js):", buffer.length);
  console.log("Truncated Buffer (Node.js):", truncatedBufferNode);
  const audioBytesEvent: AudioBytesEvent = {
    timestamp: Number(Date.now()),
    eventType: 'audioBytes',
    data: { buffer: { raw: buffer.toString('base64') } }
  };
  newEventHandler(audioBytesEvent);
};


let transcriptionMutex = false;
const transcriptionEventHandler = async (event: AudioBytesEvent) => {

  // TODO: Unbounded linear growth. Instead, walk backwards or something.
  const lastCut = getCutTimestamp();
  const audioBytesEvents = eventlog.events.filter(e => e.eventType === 'audioBytes' && e.timestamp >= lastCut);
  // Check if the user has stopped speaking
  if (VAD_ENABLED && (audioBytesEvents.length > (VAD_BUFFER_SIZE - 1))) {
    const activityEvents: Buffer[] = [];
    for (let i = VAD_BUFFER_SIZE; i > 0; i--) {
      let bufferItem = audioBytesEvents[audioBytesEvents.length - i].data.buffer;

      // If bufferItem is a direct buffer, proceed directly
      if (Buffer.isBuffer(bufferItem)) {
        activityEvents.push(bufferItem);
        continue;
      }

      // Check if bufferItem is an object with a 'raw' property
      if (typeof bufferItem === 'object' && bufferItem.raw && typeof bufferItem.raw === 'string') {
        try {
          bufferItem = Buffer.from(bufferItem.raw, 'base64');
          activityEvents.push(bufferItem);
        } catch (error) {
          console.error("Error converting raw string to buffer:", bufferItem.raw.substring(0, 50) + '...', error);
        }
      } else {
        console.error("Invalid buffer item in activityEvents. Type:", typeof bufferItem, "Value:", bufferItem);
      }
    }
    const activityBuffer = Buffer.concat(activityEvents);
    const lastTranscription = getLastTranscriptionEvent()
    const doneSpeaking = await voiceActivityDetection(activityBuffer);
    if (doneSpeaking && lastTranscription && lastTranscription.data.transcription.length) {
      return responseInputEventHandler();
    }
  }

  // Filtering and converting raw string data to buffers
  const buffersToConcatenate = audioBytesEvents
    .filter(event => typeof event.data.buffer.raw === 'string')
    .map(event => {
      try {
        return Buffer.from(event.data.buffer.raw, 'base64');
      } catch (error) {
        console.error("Error converting string to buffer:", event.data.buffer.raw.substring(0, 50) + '...', error);
        return null;
      }
    })
    .filter(item => item !== null)  // Remove any null values
    .filter(item => {
      if (!Buffer.isBuffer(item)) {
        console.error("Item is not a buffer:", item);
        return false;
      }
      return true;
    });

  // Ensure we only concatenate valid buffers
  const joinedBuffer = Buffer.concat(buffersToConcatenate as Buffer[]);

  // TODO: Wait for 1s, because whisper bindings currently throw out if not enough audio passed in
  // Therefore fix whisper
  if (!transcriptionMutex && joinedBuffer.length > ONE_SECOND) {
    try {
      transcriptionMutex = true;
      const rawTranscription = await asrInference(joinedBuffer);

      let transcription = rawTranscription.replace(/\s*\[[^\]]*\]\s*|\s*\([^)]*\)\s*/g, ''); // clear up text in brackets
      transcription = transcription.replace(/[^a-zA-Z0-9\.,\?!\s\:\'\-]/g, ""); // retain only alphabets chars and punctuation
      transcription = transcription.trim();

      const transcriptionEvent: TranscriptionEvent = {
        timestamp: Number(Date.now()),
        eventType: 'transcription',
        data: {
          buffer: joinedBuffer,
          transcription,
          lastAudioByteEventTimestamp: audioBytesEvents[audioBytesEvents.length - 1].timestamp
        }
      }
      newEventHandler(transcriptionEvent, event);

    } catch (error) {
      console.error(`Whisper promise error: ${error}`);
    } finally {
      transcriptionMutex = false;
    }
  }
}

const cutTranscriptionEventHandler = async (event: TranscriptionEvent) => {
  const lastCut = getCutTimestamp();
  const timeDiff = event.timestamp - lastCut;
  if (timeDiff > BUFFER_LENGTH_MS) {
    const cutTranscriptionEvent: CutTranscriptionEvent = {
      timestamp: event.timestamp,
      eventType: 'cutTranscription',
      data: {
        buffer: event.data.buffer,
        transcription: event.data.transcription,
        lastAudioByteEventTimestamp: event.data.lastAudioByteEventTimestamp
      }
    }
    newEventHandler(cutTranscriptionEvent, event);
  }
}

const responseReflexEventHandler = async (event: TranscriptionEvent): Promise<void> => {
  // Check if there was a response input between the last two transcription events
  const transcriptionEvents = eventlog.events.filter(e => e.eventType === 'transcription');
  const lastTranscriptionEventTimestamp = transcriptionEvents.length > 1 ? transcriptionEvents[transcriptionEvents.length - 2].timestamp : eventlog.events[0].timestamp;
  const responseInputEvents = eventlog.events.filter(e => (e.eventType === 'responseInput'));
  const lastResponseInputTimestamp = responseInputEvents.length > 0 ? responseInputEvents[responseInputEvents.length - 1].timestamp : eventlog.events[0].timestamp;
  if (lastResponseInputTimestamp > lastTranscriptionEventTimestamp) {
    const transcription = getTransciptionSoFar();
    if (transcription) {
      const responseReflexEvent: ResponseReflexEvent = {
        timestamp: Number(Date.now()),
        eventType: 'responseReflex',
        data: {
          transcription: transcription
        }
      }
      newEventHandler(responseReflexEvent, event);
    } else {
      console.log('No transcription yet. Please speak into the microphone.')
    }
  }
}

const mutex = new Mutex();

const talkEventHandler = async (event: ResponseReflexEvent): Promise<void> => {
  await mutex.lock();
  try {

    // Check if stream has been interrupted by the user
    const interruptCallback = (token: string, streamId: string): boolean => {
      const streamInterrupts = eventlog.events.filter(e => e.eventType === 'interrupt' && (e.data?.streamId == streamId));
      if (streamInterrupts?.length) {
        return true;
      }
      const lastTranscription = getLastTranscriptionEvent();
      const lastTranscriptionLength = lastTranscription?.data?.transcription?.length;
      const lastTranscriptionTimestamp = lastTranscription?.timestamp;
      const lastResponseReflexTimestamp = getLastResponseReflexTimestamp();
      if ((lastTranscriptionLength > 0) && (lastTranscriptionTimestamp > lastResponseReflexTimestamp) && (lastTranscriptionLength >= INTERRUPTION_LENGTH_CHARS)) {
        const interruptEvent: InterruptEvent = {
          timestamp: Number(Date.now()),
          eventType: 'interrupt',
          data: {
            streamId
          }
        }
        newEventHandler(interruptEvent, event);
        return true;
      }
      return false;
    }
    const talkCallback = (sentence: string) => {
      const talkEvent: TalkEvent = {
        timestamp: Number(Date.now()),
        eventType: 'talk',
        data: {
          response: sentence.trim()
        }
      }
      newEventHandler(talkEvent, event);
    };
    const input = getDialogue();
    const talkPromise = talk(
      conversationPrompt,
      input,
      llamaServerUrl,
      personaConfig,
      INTERRUPTION_ENABLED ? interruptCallback : null,
      talkCallback
    );

    await talkPromise;
  } finally {
    mutex.unlock();
  }
}

const responseInputEventHandler = (): void => {
  const responseInputEvent: ResponseInputEvent = {
    eventType: 'responseInput',
    timestamp: Number(Date.now()),
    data: {}
  }
  newEventHandler(responseInputEvent);
}

// Defines the DAG through which events trigger each other
// Implicitly used by newEventHandler to spawn the correct downstream event handler
// All event spawners call newEventHandler
// newEventHandler adds the new event to event log
// This is actually not great. Might just have it be implicit.
const eventDag: { [key in EventType]: { [key in EventType]?: (event: any) => void } } = {
  audioBytes: {
    transcription: transcriptionEventHandler,
  },
  responseReflex: {
    talk: talkEventHandler,
  },
  transcription: {
    cutTranscription: cutTranscriptionEventHandler,
    responseReflex: responseReflexEventHandler
  },
  cutTranscription: {},
  talk: {},
  responseInput: {},
  interrupt: {}
}

const audioProcess = spawn('bash', [audioListenerScript]);
audioProcess.stdout.on('readable', () => {
  let data;
  while (data = audioProcess.stdout.read()) {
    console.log('Data type:', typeof data);
    const truncatedData = data.slice(0, 10);
    console.log('Truncated data from audioProcess:', truncatedData);
    newAudioBytesEvent(data);
  }
});
audioProcess.stderr.on('data', () => {
  // consume data events to prevent process from hanging
});

readline.emitKeypressEvents(process.stdin);
process.stdin.setRawMode(true);
process.stdin.on('keypress', async (str, key) => {
  // Detect Ctrl+C and manually emit SIGINT to preserve default behavior
  if (key.sequence === '\u0003') {
    await endASR();
    writeGraph();
    process.exit();
  }

  // R for respond
  if (key.sequence === 'r') {
    responseInputEventHandler();
  }
});
