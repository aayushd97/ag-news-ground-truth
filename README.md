# ag-news-ground-truth

World: Items mainly about governments, politicians, armies, diplomacy or disasters, in any country. If politics is the reason behind the story, it goes here even when it affects another field.

Sports: Items mainly about a game, player, team or competition, including transfers and injuries. It should be purely about the sport itself, rather than finances or politics. 

Business: Items mainly about the money side of any company: profits, share prices, takeovers, sales, job cuts, prices. 

Sci/Tech: Items mainly about a discovery, a study, a gadget, software or a space mission, and what it does or how it works.

Tie-break: I ask which newspaper section the story would sit in. If I still can’t decide, I choose the closest fit and mark my confidence Low.





## How to run

1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Put `GROQ_API_KEY=your_key` in a file called `.env` (never commit this file).
3. `python run_ai.py main` runs both prompts on the 30 items.
4. `python run_ai.py imbalanced` runs prompt 1 on the imbalanced set.
5. `python measure.py` prints all the measurements.
6. `python app.py` launches the Gradio interface.

Model used: `openai/gpt-oss-20b` on the Groq free tier.