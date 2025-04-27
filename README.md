Create & activate venv

powershell
Copy
Edit
python -m venv venv
.\venv\Scripts\Activate.ps1
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Fetch art assets

bash
Copy
Edit
python scripts/fetch_assets.py
Run the core game

bash
Copy
Edit
pip install -e .
ai-trainer
Run the Pygame Zero demo

bash
Copy
Edit
pgzrun demos/pgz_demo/game_zero.py
🗂️ Project Layout
bash
Copy
Edit
.
├── README.md
├── ai_platform_trainer/      # Core package
│   ├── __main__.py           # `ai-trainer` entry point
│   ├── core/                 # Config, logging, colors
│   ├── engine/               # Display, input, renderer, physics
│   ├── entities/             # Sprite classes & factory
│   ├── agents/               # RL policy & training loop
│   └── supervised/           # Supervised-learning demo
├── assets/sprites/           # Kenney CC0 art
├── demos/pgz_demo/           # Pygame Zero quick-start
│   ├── game_zero.py
│   └── images -> ../../assets/sprites
├── scripts/fetch_assets.py   # Auto-download assets
├── requirements.txt          # Runtime deps
├── dev-requirements.txt      # Linting, testing, etc.
├── setup.py / setup.cfg      # Packaging metadata
└── venv/                     # Your virtual environment
🏗️ How It Works
Game Loop

Polls input & AI actions

Updates sprites & physics

Renders at 60 FPS with brand-aligned colors

RL Agent

Vectorized environments in Pymunk

PolicyNet (MLP) trained on GPU via PPO

Real-time inference each frame drives entity behavior

Extensible

Swap in new sprites easily

Add new physics demos (pendulum, car, cloth)

Expand supervised demos or integrate vision modules

🎓 What You’ll Learn
Packaging Python apps with console scripts

Managing virtual environments & CI prerequisites

Integrating Pygame with Pymunk physics

Building and training PyTorch models on CUDA

Reinforcement-learning design patterns (GAE, PPO)

Crafting clean, maintainable, testable code

🤝 Contributing
Contributions welcome! Please see CONTRIBUTING.md for guidelines.

📄 License
This project is licensed under the MIT License—see LICENSE for details.

📬 Contact
Your Name – drew@drewclark.io
GitHub: @life423
