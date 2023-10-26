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
    print("VAD Audio Buffer Type:", type(audio_buffer))
    
    # If it's a dictionary, print its keys
    if isinstance(audio_buffer, dict):
        print("Keys in audio_buffer:", audio_buffer.keys())
        
        # Check for the 'raw' key
        if 'raw' not in audio_buffer:
            print("Error: audioBuffer does not contain 'raw' key")
            return None
        
        byte_values = base64.b64decode(audio_buffer['raw'])

        
        # Print a truncated version of the buffer
        print("Truncated byte_values:", byte_values[:10])
    else:
        print("audio_buffer is not a dictionary")
        return None

    """Detects voice activity in the audio buffer."""
    return vad.is_speech(byte_values, sample_rate=16000)

def asr_inference(audio_buffer):
    print("ASR Audio Buffer Type:", type(audio_buffer))
    
    # If it's a dictionary, print its keys
    if isinstance(audio_buffer, dict):
        print("Keys in audio_buffer:", audio_buffer.keys())
        
        # Check for the 'raw' key
        if 'raw' not in audio_buffer:
            print("Error: audioBuffer does not contain 'raw' key")
            return None
        
        # Print a truncated version of the buffer
        print("Truncated audio_buffer['raw']:", audio_buffer['raw'][:10])
    else:
        print("audio_buffer is not a dictionary")
        return None

    """Performs ASR on the audio buffer."""
    return asr_pipeline(audio_buffer)


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