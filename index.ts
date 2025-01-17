import { spawn } from 'child_process';
import readline from 'readline';
import config from './config.json';
const { audioListenerScript } = config;
import { talk } from './src/talk';
import { Mutex } from './src/depedenciesLibrary/mutex';
import { pythonBridge } from 'python-bridge';
import fs from 'fs';
import path from 'path';
fs.writeFileSync('events.json', '[]');

let python = pythonBridge({
  python: 'python3'
});

function logEvent(eventType: string, eventData: any) {
  const currentLogs = JSON.parse(fs.readFileSync('events.json', 'utf8'));
  const eventLog = {
    timestamp: new Date().toISOString(),
    eventType,
    eventData
  };
  currentLogs.push(eventLog);
  fs.writeFileSync('events.json', JSON.stringify(currentLogs, null, 4));
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


(async () => {
  console.log("Booting up python...");
  await python.ex`
    import base64
    import webrtcvad
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from pydub import AudioSegment
    import io
    import torch
    
    model_id = 'distil-whisper/distil-large-v2'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
      model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=False
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    
    def raw_to_wav(raw_data, sample_rate=16000, sample_width=2):
        audio = AudioSegment(
            raw_data,
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=1
        )
        # Use an in-memory bytes buffer to store the WAV data
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return buffer.getvalue()
    
    def asr_inference(audio_buffer):
        decoded_buffer = base64.b64decode(audio_buffer)
        wav_buffer = raw_to_wav(decoded_buffer)

        # Preprocess the audio file (wav format is required for the processor)
        input_values = processor(wav_buffer, return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(device)

        # Perform the speech recognition
        with torch.no_grad():
            logits = model(input_values).logits

        # Retrieve the predicted ids and decode them to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        print(transcription[0])
        return transcription[0]

    def vad_function(audio_buffer):
        decoded_buffer = base64.b64decode(audio_buffer)
        # Frame size in bytes
        FRAME_SIZE = 960  # 10 ms frame for 16 kHz

        # Split buffer into frames of FRAME_SIZE
        frames = [decoded_buffer[i:i+FRAME_SIZE] for i in range(0, len(decoded_buffer), FRAME_SIZE)]

        # Process each frame and check for speech
        for frame in frames:
            if len(frame) == FRAME_SIZE and vad.is_speech(frame, sample_rate=16000):
                return True

        return False
  `;
})();

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

const DEFAULT_PROMPT = "Continue the dialogue, speak for bob only. \nMake it a fun lighthearted conversation."

let conversationPrompt: string = DEFAULT_PROMPT;
let personaConfig: string = "";
if ('personaFile' in config) {
  // @ts-ignore
  const personaFilePath = path.resolve(config.personaFile);
  if (fs.existsSync(personaFilePath)) {
    personaConfig = fs.readFileSync(personaFilePath, 'utf8');
    conversationPrompt = "";
  }
}

// asr hot replacement module section

async function voiceActivityDetection(audioBuffer: any) {

  // Check if audioBuffer has 'raw' key
  if (audioBuffer && Buffer.isBuffer(audioBuffer)) {
    const base64EncodedBuffer = audioBuffer.toString('base64');

    return await python.ex`vad_function(${base64EncodedBuffer})`;
  } else {
    // Suppressed console.error
    return false;
  }
}

async function asrInference(audioBuffer: any) {

  // Convert audioBuffer to wav if it's a dictionary with 'raw' key
  if (audioBuffer && Buffer.isBuffer(audioBuffer)) {

    const base64EncodedBuffer = audioBuffer.toString('base64');

    return await python.ex`asr_inference(${base64EncodedBuffer})
    `;
  } else {
    // Suppressed console.error
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

const updateScreenEvents: Set<EventType> = new Set(['responseReflex', 'cutTranscription', 'talk', 'interrupt']);
const updateScreen = (event: Event) => {
  if (updateScreenEvents.has(event.eventType)) {
    // Suppressed: console.log(getDialogue());
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
  logEvent('event', event);
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
        // Suppressed console.error
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
        return false;
      }
      return true;
    });

  // Ensure we only concatenate valid buffers
  const joinedBuffer = Buffer.concat(buffersToConcatenate as Buffer[]);


  if (!transcriptionMutex && joinedBuffer.length > ONE_SECOND) {
    try {
      transcriptionMutex = true;
      const rawTranscription = await asrInference(joinedBuffer) || "";

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
      console.error("Error during ASR process:", error);
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
