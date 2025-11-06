import os
import queue
import openai
from langdetect import detect, DetectorFactory
from google.cloud import speech
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import uuid
import requests
import difflib
import json
from datetime import datetime

# Set credentials and keys
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"ai-call-center-prototype-af97605da82a.json"

OPENAI_API_KEY = "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"  # Replace with your actual OpenAI key
ELEVENLABS_API_KEY = "sk_40256093dd7e787953415c6c876b3026dca66bf0feacb3ce"   # Replace with your actual ElevenLabs key

openai.api_key = OPENAI_API_KEY
DetectorFactory.seed = 0  # for langdetect consistency

SYSTEM_PROMPT = """
Ești agentul Andy's Pizza. Ești prietenos, dar foarte eficient. Vorbești scurt, clar și pe rând.

⚠️ REGULI STRICTE:
1. NU saluta clientul.
2. NU pui mai multe întrebări în același mesaj.
3. Întrebi un singur lucru pe rând, și AȘTEPȚI răspunsul clientului înainte de a continua.
4. NU presupune comanda sau adresa. Confirmă tot pas cu pas.
5. NU inventa informații. Fii fidel meniului.
6. Fii atent la primul lucru pe care iti spune clientul, daca iti spune ce vrea sa comande, nu mai intreba a doua oara.

📝 FLUX DE CONVERSAȚIE:
1. Dacă clientul NU a spus ce dorește, întreabă: "Ce doriți să comandați?"
2. După fiecare item, întreabă: "Mai doriți ceva?" — până clientul spune că e tot.
3. Confirmă comanda completă (Ex: "Deci o pizza Rancho și o Pepperoni.")
4. Întreabă: "Este corectă comanda?"
5. Întreabă: "Livrare, ridicare sau servire în restaurant?"
6. Dacă livrare, ceri adresa. Dacă ridicare/restaurant, ceri locația dorită.
7. Dacă se cere rezervare, ceri data, ora, locația, număr persoane.
8. Dacă se cere programare pentru o oră mai târzie: confirmi politicos că e în regulă.
9. După confirmare ceri numele și prenumele clientului.

🛑 NU continua până nu ai răspunsul pentru fiecare întrebare.

📦 MENIU:
-Pizza scampi
    Aluat, sos de roșii, cașcaval, creveți, parmezan, olive, frișcă, pătrunjel, zest de lămâie, condimente.
    140 lei.

-Pizza Tonno
    Aluat, sos de roșii, mozzarella, ton, olive, ceapă roșie, busuioc, mărar, condimente.  
    130 lei.

-Pizza Bianco
    Aluat, cașcaval, mousse de cașcaval, mozzarella, șuncă Tambov, ciuperci champignon, parmezan, busuioc, condimente.
    130 lei.

-Pizza 5 Cheeses
    Aluat, mousse de cașcaval, cașcavaluri mozzarella, brie, dorblue, parmezan.
    125 lei.

-Pizza Capricioasa
    Aluat, cașcaval, sos de roșii, șuncă, ciuperci, vânătă, brânză de oi, maioneză, usturoi.
    125 lei.

-Pizza Mario
    Aluat, cașcaval, sos de roșii, maioneză, carne de pui, cârnăcior, gogoșari, cașcaval Dorblue, spanac.
    125 lei.

-Pizza Diablo
    Aluat, cașcaval, sos de roșii, asorti de salamuri, ciuperci, gogoșari, olive, ardei iute.
    130 lei.

-Pizza Neapolitana
    Aluat, cașcaval, sos de roșii, șuncă, ciuperci, olive, măsline.
    125 lei.

-Pizza BARBEQUE
    Aluat, cașcaval, file de pui, sos de roșii, bacon, salami, sos BBQ, gogoșari.
    130 lei.

-Pizza Rancho
    Aluat, cașcaval, carne de pui, ciuperci, gogoșari, sos de roșii, maioneză.
    125 lei.

-Pizza Margherita
    Aluat, mozzarella, sos de roșii, ulei de olive, busuioc. Bucatele nu rezistă transportării.
    95 lei.

-Pizza Pepperoni
    Aluat, cașcaval, sos de roșii, salami, condimente.  
    130 lei.

Răspunde scurt, clar și politicos. NU combina întrebările. NU iniția altă întrebare până nu primești răspunsul.
"""
PIZZAS = [
    "Margherita", "Pepperoni", "Rancho", "BARBEQUE", "Neapolitana",
    "Diablo", "Mario", "Capricioasa", "5 Cheeses", "Bianco", "Tonno", "scampi"
]

MISHEARINGS = {
    "Ranciu": "Rancho",
    "Rancio": "Rancho",
    "Vrancea": "Rancho",
    "Barbeqiu": "BARBEQUE",
    "Barbechiu": "BARBEQUE",
    "Napolitana": "Neapolitana",
    "Faichize": "5 Cheeses",
    "Forțez": "5 Cheeses",
    "Forces": "5 Cheeses",
    "Force": "5 Cheeses",
    "Four Cheese": "5 Cheeses",
}


def correct_pizza_names(text):
    words = text.split()
    corrected_words = []

    for w in words:
        w_lower = w.lower()
        replaced = False
        for key, val in MISHEARINGS.items():
            if w_lower == key.lower():
                corrected_words.append(val)
                replaced = True
                break
        if replaced:
            continue

        close_matches = difflib.get_close_matches(w, PIZZAS, n=1, cutoff=0.8)
        if close_matches:
            corrected_words.append(close_matches[0])
        else:
            corrected_words.append(w)

    return " ".join(corrected_words)


def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith("ro"):
            return "romanian"
        elif lang.startswith("ru"):
            return "russian"
        elif lang.startswith("en"):
            return "english"
        else:
            return "english"
    except Exception:
        return "english"
def recognize_speech():
    client = speech.SpeechClient()

    RATE = 16000
    CHUNK = int(RATE / 10)

    class MicrophoneStream:
        def __init__(self, rate, chunk):
            self.rate = rate
            self.chunk = chunk
            self._buff = queue.Queue()
            self.closed = True

        def __enter__(self):
            self.audio_interface = pyaudio.PyAudio()
            self.audio_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._fill_buffer,
            )
            self.closed = False
            return self

        def __exit__(self, type, value, traceback):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.closed = True
            self._buff.put(None)
            self.audio_interface.terminate()

        def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
            self._buff.put(in_data)
            return None, pyaudio.paContinue

        def generator(self):
            while not self.closed:
                chunk = self._buff.get()
                if chunk is None:
                    return
                yield chunk

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ro-RO",
        alternative_language_codes=["ru-RU"],
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        max_alternatives=5,
        speech_contexts=[speech.SpeechContext(
            phrases=[
                *PIZZAS, *MISHEARINGS.keys(), *MISHEARINGS.values()
            ]
        )]
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests_gen = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)

        print("🎤 Listening...")

        try:
            partial_transcript = ""
            for response in client.streaming_recognize(streaming_config, requests_gen):
                if not response.results:
                    continue
                result = response.results[0]
                transcript = result.alternatives[0].transcript

                if result.is_final:
                    confidence = result.alternatives[0].confidence
                    print(f"🧐 Final: '{transcript}' (confidence {confidence:.2f})")

                    if confidence < 0.90 and len(result.alternatives) > 1:
                        print("🔍 Low confidence, AI will resolve the correct intent.")
                        all_alts = [alt.transcript.strip() for alt in result.alternatives]
                        for i, alt in enumerate(all_alts):
                            print(f"  Alt {i+1}: {alt}")

                        alt_prompt = "Am următoarele variante STT pentru ceea ce a spus clientul. Alege cea mai probabilă variantă corectă, sau reformulează dacă e nevoie:\n" + \
                            "\n".join([f"{i+1}. {alt}" for i, alt in enumerate(all_alts)]) + "\nRăspunsul final (în limba clientului):"

                        try:
                            response = openai.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": alt_prompt}]
                            )
                            resolved = response.choices[0].message.content.strip()
                            print(f"🔗 Resolved by AI: {resolved}")
                            return resolved, detect_language(resolved)
                        except Exception as e:
                            print(f"❌ OpenAI fallback error: {e}")

                    detected_language = detect_language(transcript)
                    return transcript.strip(), detected_language

                else:
                    if transcript != partial_transcript:
                        partial_transcript = transcript
                        print(f"🕒 Partial: '{partial_transcript}'")

        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return None, None

    return None, None
def get_ai_response(conversation, model="gpt-4o"):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=conversation,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return "Scuze, a apărut o problemă. Poți repeta?"


def get_ai_response_strict(conversation):
    strict_prompt = SYSTEM_PROMPT + "\n\nReține: Pune doar o singură întrebare per mesaj și NU combina întrebările."

    try:
        new_conversation = [{"role": "system", "content": strict_prompt}] + [m for m in conversation if m["role"] != "system"]
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=new_conversation,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI error (strict): {e}")
        return "Scuze, a apărut o problemă. Poți repeta?"


def speak_text(text, language):
    voice_id = "3z9q8Y7plHbvhDZehEII"  # Romanian/Eastern European friendly voice

    try:
        filename = f"output_{uuid.uuid4().hex}.mp3"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        json_data = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(url, headers=headers, json=json_data)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)

            audio_segment = AudioSegment.from_file(filename, format="mp3")
            play(audio_segment)
            os.remove(filename)
        else:
            print(f"❌ ElevenLabs TTS HTTP error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ ElevenLabs TTS error: {e}")
def summarize_order(conversation):
    prompt = """
Ești Andy's Pizza. Rezumă comanda clientului în format JSON strict cu următoarele câmpuri:
- pizzas: lista de pizza comandate, fiecare ca string simplu
- delivery_method: "livrare", "ridicare" sau "restaurant"
- address: adresa sau locația de ridicare (string)
- name: numele clientului (string)

Extrage toate detaliile din conversația de mai jos. Dacă un câmp nu există, pune null.

Conversatia:
""" + "\n".join([f"{m['role']}: {m['content']}" for m in conversation if m['role'] != 'system']) + """

Răspunsul JSON strict:
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw_output = response.choices[0].message.content.strip()
        print(f"\n📦 Raw GPT Order Output:\n{raw_output}\n")

        # Try to extract JSON even if wrapped in markdown
        try:
            if "```" in raw_output:
                raw_output = raw_output.split("```")[1]
            json_start = raw_output.find("{")
            json_text = raw_output[json_start:]
            return json.loads(json_text)
        except json.JSONDecodeError as je:
            print(f"⚠️ JSON decode error: {je}")
            return None

    except Exception as e:
        print(f"❌ Error summarizing order: {e}")
        return None


def main():
    print("🤖 Voice assistant ready. Say 'exit' to quit.")

    info_msg = (
        "Acest apel este preluat de un asistent AI. "
        "Vă rugăm să vorbiți rar, clar și tare. "
        "Asistentul va răspunde în limba în care salutați: română, rusă sau engleză.\n\n"
    )
    print(f"📢 {info_msg}")
    speak_text("Acest apel este preluat de un asistent AI. Vă rugăm să vorbiți rar, clar și tare. Asistentul va răspunde în limba în care salutați: română, rusă sau engleză.", "romanian")

    print("\n🤖 Waiting for greeting...")
    greeting_text, user_language = recognize_speech()
    if greeting_text is None:
        print("❌ No greeting detected. Restart the call.")
        return

    user_language = detect_language(greeting_text)
    print(f"🌍 Detected language: {user_language}")
    locked_language = user_language  # lock it

    greetings = {
        "romanian": "Bună ziua, Andy's Pizza cu ce vă putem ajuta?",
        "russian": "Здравствуйте, это Andy's Pizza. Чем можем помочь?",
        "english": "Hello, this is Andy's Pizza. How can we help you?",
    }
    greeting = greetings.get(locked_language, greetings["english"])
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"🤖 AI says: {greeting}")
    speak_text(greeting, locked_language)

    while True:
        user_input, _ = recognize_speech()
        if user_input is None:
            continue

        user_input = correct_pizza_names(user_input)
        print(f"🗣️ You said ({locked_language}): {user_input}")

        if user_input.lower() in ["exit", "quit", "stop"]:
            goodbye = {
                "romanian": "Mulțumim că ați sunat la Andy's Pizza. O zi bună!",
                "russian": "Спасибо за звонок в Andy's Pizza. Хорошего дня!",
                "english": "Thank you for calling Andy's Pizza. Have a nice day!"
            }.get(locked_language, "Thank you for calling. Goodbye!")
            print(f"🤖 AI says: {goodbye}")
            speak_text(goodbye, locked_language)
            break

        conversation_history.append({"role": "user", "content": user_input})

        reply = get_ai_response(conversation_history, model="gpt-4o")

        if reply.count('?') > 1:
            print("⚠️ Multiple questions detected, retrying with stricter prompt.")
            reply = get_ai_response_strict(conversation_history)
            if reply.count('?') > 1:
                reply = reply.split('?')[0] + '?'
                print(f"⚠️ Truncated reply to single question: {reply}")

        print(f"🤖 AI says: {reply}")
        conversation_history.append({"role": "assistant", "content": reply})
        speak_text(reply, locked_language)

        if any(phrase in reply.lower() for phrase in ["mulțumim", "vă așteptăm", "o zi bună", "thank you", "goodbye", "спасибо", "день"]):
            order_summary = summarize_order(conversation_history)
            if order_summary:
                filename = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(order_summary, f, ensure_ascii=False, indent=4)
                print(f"✅ Order saved to {filename}")
            else:
                print("⚠️ No order data extracted.")
            break


if __name__ == "__main__":
    main()
