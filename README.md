# Assignment-2-Fine-tuning-Text-to-Speech-TTS-Models
Assignment 2: Fine-tuning Text-to-Speech (TTS) Models for English Technical Speech and Regional Languages

This repository contains two fine-tuned Text-to-Speech (TTS) models: one for English technical jargon and another for Hindi speech synthesis. Each model is designed to convert text input into high-quality audio output.

## Table of Contents
- [Models Overview](#models-overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Task 1: English Technical TTS Model](#task-1-english-technical-tts-model)
  - [Task 2: Hindi TTS Model](#task-2-hindi-tts-model)
- [Usage Instructions](#usage-instructions)
- [Conclusion](#conclusion)

## Models Overview

### English Technical TTS Model
- **Model** (https://huggingface.co/Tejasva-Maurya/English_Technical_finetuned)
- This model is fine-tuned to handle technical jargon in English, producing clear and natural-sounding audio from text input.

### Hindi TTS Model
- **Model** (https://huggingface.co/Tejasva-Maurya/Hindi_SpeechT5_finetuned)
- This model is specifically designed for Hindi speech synthesis, enabling smooth conversion of Hindi text into spoken words.

## Prerequisites

Before running the models, ensure you have the following libraries installed:

```bash
pip install transformers torch speechbrain datasets
```

## Getting Started

### Task 1: English Technical TTS Model

1. **Load the Model**
   Use the following code to load the English TTS model:

   ```python
    from transformers import AutoProcessor, AutoModelForTextToSpectrogram, SpeechT5HifiGan
    
    # Load English model
    processor = AutoProcessor.from_pretrained("Tejasva-Maurya/English_Technical_finetuned")
    model = AutoModelForTextToSpectrogram.from_pretrained("Tejasva-Maurya/English_Technical_finetuned")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
   ```

2. **Load the Dataset and Create Embeddings**
   Load the englist techical dataset embeddings for processing:

   ```python
    import os
    import torch
    from speechbrain.pretrained import EncoderClassifier
    from datasets import load_dataset, Audio
    
    dataset = load_dataset("csv", data_files = "/content/drive/MyDrive/Colab Notebooks/TTS/ALL_DATASET/metadata_COLAB.csv",split="train", trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", spk_model_name),
    )
    def create_speaker_embedding(waveform):
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings
    def prepare_dataset(example):
        audio = example["audio"]
        example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
        return example
    
    
    # Calculate the number of rows for a part of the dataset
    part = len(dataset) //1000
    
    # Select the part of the dataset
    dataset = dataset.select(range(part))
    
    # Prepare the dataset
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    example = dataset[10]
    speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
   ```

3. **Prepare the Data or Text Preprocessing**
   Preprocess the input data:

   ```python
    import re
    def text_preprocessing(text):
         replacements = [
         ("0", "zero"),
         ("1", "one"),
         ("2", "two"),
         ("3", "three"),
         ("4", "four"),
         ("5", "five"),
         ("6", "six"),
         ("7", "seven"),
         ("8", "eight"),
         ("9", "nine"),
         ("_", " ")]
          # Convert to lowercase
         text = text.lower()
      
         # Remove punctuation (except apostrophes)
         text = re.sub(r'[^\w\s\']', '', text)
      
         # Remove extra whitespace
         text = ' '.join(text.split())
         for src, dst in replacements:
           text = text.replace(src, dst)
         return text
   ```

3. **Process Input and Generate Audio**
   Process the input text and generate the audio:

   ```python
   text ="CUDA is a parallel computing platform and programming model."
   text = text_preprocessing(text)
   inputs = processor(text=text, return_tensors="pt")
   speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
   ```

4. **Play or Save the Audio**
   Use your preferred method to play or save the generated audio.
   ```python
    from IPython.display import Audio
    Audio(speech.numpy(), rate = 16000)
   ```

### Task 2: Hindi TTS Model

1. **Load the Model**
   Use the following code to load the Hindi TTS model:

   ```python
    from transformers import AutoProcessor, AutoModelForTextToSpectrogram, SpeechT5HifiGan
    
    # Load English model
    processor = AutoProcessor.from_pretrained("Tejasva-Maurya/Hindi_SpeechT5_finetuned")
    model = AutoModelForTextToSpectrogram.from_pretrained("Tejasva-Maurya/Hindi_SpeechT5_finetuned")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
   ```

2. **Load the Hindi Dataset and create Embeddings**
   Load the Hindi dataset and Crate Embeddings for processing:

   ```python
    import os
    import torch
    from speechbrain.pretrained import EncoderClassifier
    from datasets import load_dataset, Audio
    
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="validated", trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", spk_model_name),
    )
    def create_speaker_embedding(waveform):
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings
    def prepare_dataset(example):
        audio = example["audio"]
        example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
        return example
    
    
    # Calculate the number of rows for a part of the dataset
    part = len(dataset) //800
    
    # Select the part of the dataset
    dataset = dataset.select(range(part))
    
    # Prepare the dataset
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    example = dataset[10]
    speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
   ```
3. **Prepare the Data or Text Preprocessing**
   Preprocess the input data:
   ```python
    import re
    def text_preprocessing(text):
         replacements = [
           # Vowels and vowel matras
           ("अ", "a"),
           ("आ", "aa"),
           ("इ", "i"),
           ("ई", "ee"),
           ("उ", "u"),
           ("ऊ", "oo"),
           ("ऋ", "ri"),
           ("ए", "e"),
           ("ऐ", "ai"),
           ("ऑ", "o"),  # More accurate than 'au' for ऑ
           ("ओ", "o"),
           ("औ", "au"),
           # Consonants
           ("क", "k"),
           ("ख", "kh"),
           ("ग", "g"),
           ("घ", "gh"),
           ("ङ", "ng"),  # nasal sound
           ("च", "ch"),
           ("छ", "chh"),
           ("ज", "j"),
           ("झ", "jh"),
           ("ञ", "ny"),  # 'ny' closer to the actual sound
           ("ट", "t"),
           ("ठ", "th"),
           ("ड", "d"),
           ("ढ", "dh"),
           ("ण", "n"),  # Slight improvement for easier pronunciation
           ("त", "t"),
           ("थ", "th"),
           ("द", "d"),
           ("ध", "dh"),
           ("न", "n"),
           ("प", "p"),
           ("फ", "ph"),
           ("ब", "b"),
           ("भ", "bh"),
           ("म", "m"),
           ("य", "y"),
           ("र", "r"),
           ("ल", "l"),
           ("व", "v"),  # 'v' is closer to the Hindi 'व'
           ("श", "sh"),
           ("ष", "sh"),  # Same sound in modern pronunciation
           ("स", "s"),
           ("ह", "h"),
           # Consonant clusters and special consonants
           ("क्ष", "ksh"),
           ("त्र", "tr"),
           ("ज्ञ", "gya"),
           ("श्र", "shra"),
           # Special characters
           ("़", ""),    # Ignore nukta; can vary with regional pronunciation
           ("्", ""),    # Halant - schwa dropping (handled contextually)
           ("ऽ", ""),    # Avagraha - no direct pronunciation, often ignored
           ("ं", "n"),   # Anusvara - nasalization
           ("ः", "h"),   # Visarga - adds an 'h' sound
           ("ँ", "n"),   # Chandrabindu - nasalization
           # Vowel matras (diacritic marks)
           ("ा", "a"),
           ("ि", "i"),
           ("ी", "ee"),
           ("ु", "u"),
           ("ू", "oo"),
           ("े", "e"),
           ("ै", "ai"),
           ("ो", "o"),
           ("ौ", "au"),
           ("ृ", "ri"),  # Vowel-matra equivalent of ऋ
           # Nasalization and other marks
           ("ॅ", "e"),   # Short 'e' sound (very rare)
           ("ॉ", "o"),   # Short 'o' sound (very rare)
           # Loanwords and aspirated consonants
           ("क़", "q"),
           ("ख़", "kh"),
           ("ग़", "gh"),
           ("ज़", "z"),
           ("ड़", "r"),
           ("ढ़", "rh"),
           ("फ़", "f"),
           # Punctuation
           ("।", "."),   # Hindi sentence-ending marker -> period
       ]
    
         # Remove extra whitespace
         text = ' '.join(text.split())
         for src, dst in replacements:
           text = text.replace(src, dst)
         return text
   ```

5. **Process Input and Generate Audio**
   Process the input text and generate the audio:

   ```python
    input_text ="आज मौसम बहुत अच्छा है।"
    text = text_preprocessing(input_text)
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
   ```

7. **Play or Save the Audio**
   Use your preferred method to play or save the generated audio.
   ```python
    from IPython.display import Audio
    Audio(speech.numpy(), rate = 16000)
   ```

## Usage Instructions

To run the models, please refer to the following Jupyter notebook files included in this repository:
- **TTS technical dataset link** (https://drive.google.com/drive/folders/1-2fI5aLI78KCZGsEIAJRQ5YnRVvVSqyS?usp=sharing)
- **Hindi dataset link** (https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/viewer/hi/validated)
- **[Task 1: English TTS Model](https://github.com/Tejasva-Maurya/Assignment-2-Fine-tuning-Text-to-Speech-TTS-Models/blob/main/Task_1_running_steps.ipynb)**: This notebook provides step-by-step instructions and code to run the English TTS model.
- **[Task 2: Hindi TTS Model](https://github.com/Tejasva-Maurya/Assignment-2-Fine-tuning-Text-to-Speech-TTS-Models/blob/main/Task_2_running_steps.ipynb)**: This notebook offers guidance on running the Hindi TTS model, including dataset preparation and audio generation.

## Conclusion
This repository provides a comprehensive implementation of TTS models for both English and Hindi. Experiment with different input texts to explore the capabilities of each model! Thank you for the opportunity to work on this assignment.
