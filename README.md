# dutch-ai
Russian GPT model for Discord, which copy your style and generate random messages

# How to use?
1. Collect ~1000 of your messages into dataset.txt file as in example
2. Copy AI.py into Jupyter Notebook at any MLaaS-service (or VS Code, if you has enough memory) and start learning
3. Download my-dytch-gpt file into folder with bot.py
4. Make bot token at Discord Developer Portal
5. Open PowerShell, print at target directory:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install torch transformers discord.py
   $env:DISCORD_TOKEN="your_bot_token"
   $env:DISCORD_CHANNEL_ID="target_channel_id"
   $env:LEARN_USER_IDS="your_user_id"
6. Start at PowerShell python "Dutch AI\bot.py"

# Warning:
It is ruGPT, and it works at Russian language, and most of messages must be at russian

# Description
"AI.py" file is offline learning of model, and "bot.py" file is online fine-tuning and Discord API
