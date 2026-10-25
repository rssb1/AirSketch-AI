# AirSketch AI

**Draw in the air. Challenge AI. Beat your best score.**

AirSketch AI is a Pictionary-style game where you recreate reference pictures by drawing with your finger in front of a webcam. Computer vision tracks your hand, and an image evaluation system scores your drawing against the reference.

## Features

- Real-time air drawing using OpenCV and MediaPipe Hands
- 18 built-in drawing challenges across easy, medium, and hard difficulties
- Gesture-based drawing controls, adjustable colors and brush sizes, and a 60-second timer
- AI-powered drawing evaluation using a vision-language model
- Offline shape-comparison fallback when AI evaluation is unavailable
- Session score tracking and mouse-drawing fallback
- Automated tests that run without a webcam or API key

## Tech Stack

Python · OpenCV · MediaPipe · NumPy · Pillow · Pygame · OpenAI API

## Installation

Requires Python 3.10–3.12.

**Windows (PowerShell)**

```powershell
git clone https://github.com/YOUR_USERNAME/AirSketch-AI.git
cd AirSketch-AI

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**macOS / Linux**

```bash
git clone https://github.com/YOUR_USERNAME/AirSketch-AI.git
cd AirSketch-AI

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

The game works without an API key using its local shape-comparison evaluator.

For AI-powered semantic evaluation, copy `.env.example` to `.env` and configure your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

Never commit your `.env` file or expose your API key publicly.

## How to Play

1. Choose a difficulty and start a challenge.
2. Recreate the reference image by pointing your index finger at the webcam.
3. Submit your drawing or let the timer run out.
4. View your score, AI feedback, and the closest matching challenge.

**Controls:** Point with your index finger to draw, use a peace sign or fist to lift the pen, and hold an open palm to clear the canvas. Keyboard shortcuts and mouse controls are also available.

## How Evaluation Works

AirSketch AI combines two evaluation methods:

- **Shape comparison:** Normalizes the drawing and reference image, then measures how closely their outlines match.
- **AI evaluation:** An optional vision-language model evaluates what the drawing depicts and its semantic similarity to the reference, then provides feedback.

When AI evaluation is unavailable, the game uses its local shape-comparison result and labels it as approximate. Scores measure resemblance to the reference, not artistic skill.

## Testing

Run the automated tests:

```bash
python -m pytest -v
```

The tests cover game logic, scoring, gesture classification, and evaluation fallback behavior without requiring a webcam or network connection.

## Limitations

- The local fallback evaluates shapes rather than understanding their meaning.
- The game supports one hand and primarily uses index-finger drawing.
- Reference challenges use simple line art, so scores depend on the complexity of the image.

## Future Improvements

- Add local semantic image recognition.
- Introduce custom drawing prompts and additional challenges.
- Add persistent high scores and multiplayer mode.


## Development Note

This project was built with substantial assistance from AI coding tools. I used AI to help implement features, structure the application, and develop tests, while reviewing and testing the resulting functionality.