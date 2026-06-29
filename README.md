Link to video: https://youtu.be/SlEuwpC0eSI

# TakeMeter: Steam Review Sentiment Classifier

A fine-tuned text classifier that sorts Steam game reviews into `positive` or `negative` by the reviewer's overall stance, compared honestly against a zero-shot large language model baseline.

**Headline result:** the zero-shot Groq baseline (`llama-3.3-70b-versatile`, 70B parameters) scored 0.933 accuracy on the test set. The fine-tuned DistilBERT (66M parameters, trained on 140 examples) scored 0.767. Fine-tuning did not beat the baseline here, and the rest of this report explains why, which turns out to be the most interesting part of the project.

## Community choice and reasoning

I chose **Steam game reviews**, drawn from a curated mix of seven games via the public Steam store review pages.

Steam reviews are a large, active, text heavy discourse space. Players argue about whether a game is worth buying, and the writing ranges from one line praise to long structured critiques. Two properties made it a good fit for a classification task:

1. **Built in ground truth to sanity check against.** Every review carries the author's own Recommended or Not Recommended flag. I did not use that flag as the training label (that would be leakage), but it let me confirm my hand labels were sane.
2. **Genuine ambiguity.** Even reduced to two labels, Steam sentiment is not a keyword lookup. Reviews use sarcasm ("oh great, another paid beta"), redemption arcs ("disastrous launch, but they fixed it and now it is great"), and hype framing. Reading the overall stance instead of counting positive and negative words is the actual challenge, and it is exactly where the model fails.

## Label taxonomy

Two labels, mutually exclusive. A review is labeled by the reviewer's overall, final stance, not by counting individual pros and cons.

### `positive`
The reviewer is overall satisfied and would recommend the game. Praise dominates the overall stance, even if minor complaints appear.
- "Easily 200 hours in and still finding new things. Worth every penny."
- "Combat feels incredible and the world is gorgeous. Cannot stop playing."

### `negative`
The reviewer is overall dissatisfied and would not recommend the game. Complaints dominate the overall stance, even if minor praise appears.
- "Crashes every 20 minutes since the last update. Unplayable right now."
- "Boring grind, dead servers, and full of microtransactions. Skip it."

I originally planned a third `mixed` label for genuinely ambivalent reviews but dropped it: the boundary was too subjective to label consistently, which would have injected noise into training. Ambivalent reviews are instead resolved to their dominant final stance. See `planning.md` for the full design history.

## Data collection

- **Source:** public Steam reviews (English only) from seven games, copied from the store review pages into a paste friendly text file and converted to CSV. Game titles: Cyberpunk 2077, Fallout 4, Left 4 Dead 2, No Man's Sky, REANIMAL, Starfield, Terraria.
- **Process:** I read each review and assigned a label by hand using the definitions above. To prevent the model from learning a game name as a sentiment shortcut, the game title is kept in its own CSV column and never concatenated into the `text` the model sees. I deliberately mixed well loved and divisive titles, and used the Not Recommended filter to over collect negatives, since Steam skews heavily positive.
- **Tooling:** a small script (`scripts/build_csv.py`) converts the labeling sheet (`data/reviews_raw.txt`) into the final `data/data.csv` and reports the distribution. It does no labeling itself.

### Label distribution (200 examples)

| Label | Count |
|---|---|
| negative | 101 |
| positive | 99 |

No single label exceeds 70 percent (the split is essentially 50/50). Each game is also internally balanced so the game cannot act as a proxy for the label:

| Game | positive | negative |
|---|---|---|
| Cyberpunk 2077 | 15 | 15 |
| Fallout 4 | 15 | 15 |
| Left 4 Dead 2 | 15 | 16 |
| No Man's Sky | 14 | 15 |
| REANIMAL | 15 | 15 |
| Starfield | 15 | 15 |
| Terraria | 10 | 10 |

### Three examples that were genuinely hard to label

1. **Sentiment flip over time (Left 4 Dead 2).** "After all these years that I loved the game, unfortunately I must change my review to negative. the latest TLS update ruined the fun." Most of the words are past affection, so a word count leans positive. Decision: label by the final stance. The reviewer explicitly changes their verdict, so `negative`.
2. **Split numeric score (Left 4 Dead 2).** "the worst game i have played in my life... overall 1/10 for versus, 9/10 for playing custom maps with friends." Two opposite scores for two modes. Decision: do not average. The opening verdict and overall framing target the core multiplayer experience, so `negative`.
3. **Lukewarm conditional recommendation (Starfield).** "Can I recommend it full price?... yeah, but if you have other games in mind I would just wait for a sale. Do I regret buying it? No... Leaving the positive review." Tepid and hedged. Decision: a conditional recommendation that still says buy lands `positive`, and the reviewer states they do not regret it.

## Fine-tuning approach

- **Base model:** `distilbert-base-uncased` (66M parameters) with a 2-class sequence classification head, via HuggingFace `transformers`.
- **Training setup:** 70/15/15 stratified split (140 train, 30 validation, 30 test), tokenized at `max_length=256`, fine-tuned on a free Colab T4 GPU. Final config: 8 epochs, learning rate 2e-5, train batch size 8, warmup 10 steps, weight decay 0.01, best checkpoint selected by validation accuracy.

### Key hyperparameter decision: warmup, then a deliberate overfitting test

This is the decision I learned the most from, because it came in three stages with real before and after numbers.

1. **The warmup bug (0.50 to 0.83).** The notebook default was `warmup_steps=50`. With 140 training examples at batch size 16, an epoch is only about 9 steps, so 3 epochs is about 27 total steps. The learning rate scheduler ramps from zero over 50 steps, so training ended before the learning rate ever reached its target. The model barely moved from its random initialization and scored 0.50 (random for a 2-class task). Its confusion matrix had identical rows, meaning the output ignored the input entirely. Cutting warmup to 10 steps, halving the batch size to 8 (more steps per epoch), and raising epochs to 8 fixed it: the learning rate actually ramped and accuracy jumped to roughly 0.83.

2. **The overfitting attempt (0.83 to 0.77).** To try to close the gap to the baseline, I pushed harder: 12 epochs, learning rate 3e-5, and `max_length=512`. Accuracy dropped to 0.767. The tell was in the confidence scores: errors that had been made at 0.52 to 0.59 confidence were now made at 0.96 to 1.00. The model had memorized training patterns and become confidently wrong. More capacity to fit 140 examples hurt both accuracy and calibration.

3. **Run to run variance.** Reverting to the conservative config did not reliably reproduce 0.83; a fresh run of the identical settings produced 0.767. With only 140 training examples and a randomly initialized classification head, the test accuracy moves by one or two examples (about 0.03 to 0.07) between runs. The honest summary is that this model sits around 0.77 to 0.83, not at a single magic number. I report the locked final run below.

## Baseline description

The baseline is a zero-shot prompt to Groq's `llama-3.3-70b-versatile` with no task-specific training. The system prompt gives the two label definitions and one example each, instructs the model to judge overall stance rather than count words, and to output only the label name. Each test review was sent at `temperature=0`, and the response string was matched back to a label. All 30 of 30 responses were parseable.

Prompt used:

```
You are classifying Steam game reviews by overall sentiment.
Assign each review to exactly one of the following two categories.

positive: The reviewer is overall satisfied and would recommend the game. Praise dominates the overall stance, even if minor complaints appear.
Example: "Combat feels incredible and the world is gorgeous. Cannot stop playing."

negative: The reviewer is overall dissatisfied and would not recommend the game. Complaints dominate the overall stance, even if minor praise appears.
Example: "Crashes every 20 minutes since the last update. Unplayable right now."

Judge by the reviewer's overall, final stance, not by counting positive and negative words. Sarcasm flips surface words, so read for real intent.

Respond with ONLY the label name: positive or negative.
Do not explain your reasoning.

Valid labels:
positive
negative
```

## Evaluation report

### Overall accuracy

| Model | Accuracy | Macro F1 |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | 0.933 | 0.93 |
| Fine-tuned DistilBERT | 0.767 | 0.76 |

The fine-tuned model is 0.167 below the baseline. This is an expected outcome, not a hidden failure: a 66M parameter model trained on 140 hand labeled examples is competing against a 70B parameter model that already learned sentiment from internet scale pretraining. The comparison exists precisely to show how hard the task is for a small model with little data.

### Per-class metrics

Baseline (Groq):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.93 | 0.93 | 0.93 | 15 |
| negative | 0.93 | 0.93 | 0.93 | 15 |

Fine-tuned DistilBERT:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.83 | 0.67 | 0.74 | 15 |
| negative | 0.72 | 0.87 | 0.79 | 15 |

The asymmetry is the story: positive recall is only 0.67. The model misses a third of the positive reviews, predicting them as negative.

### Confusion matrix (fine-tuned model, test set)

| | predicted positive | predicted negative |
|---|---|---|
| **true positive** | 10 | 5 |
| **true negative** | 2 | 13 |

The dominant error is **positive predicted as negative** (5 of 7 errors). A supplementary image is committed as `confusion_matrix.png`.

### Three wrong predictions, analyzed

1. **Cyberpunk 2077, true positive, predicted negative (confidence 0.97).** "To say Cyberpunk 2077 had a rough launch is like saying the Titanic's maiden trip encountered a minor hiccup. Yes, it was that bad. A colossal failure..." This is a redemption arc: the review spends its first half describing how bad the launch was before turning to praise. The model anchors on the dense negative vocabulary up front and never recovers. This is a labeling-consistent example (it is genuinely positive), so the failure is in the model and data, not the annotation.
2. **No Man's Sky, true positive, predicted negative (confidence 0.95).** "I can't believe they kept working on it after launch. I can't believe they didn't just take the money and run. I can't believe they righted their wrongs." The anaphora "I can't believe" reads as disbelief and the individual clauses name past wrongs ("take the money and run"). The praise is structural and ironic rather than lexical, so a model keying on word level sentiment reads it as a complaint.
3. **Hype framing, true negative, predicted positive (confidence 0.89).** "This was supposed to be it. The big one. The game to top all games. It had the marketing. It had the hype... But it was not meant to be." The vocabulary is loaded with positive hype words, and the deflation ("but it was not meant to be") is subtle and short. The model is fooled by the surface excitement.

**The pattern, verified by re-reading:** the model has not learned to handle sentiment that is carried by structure rather than vocabulary. Redemption arcs and ironic praise (negative words, positive meaning) get called negative; hype framing (positive words, negative meaning) gets called positive. The errors are directionally concentrated on positive reviews because the reversal structure is more common there. This is the same `mixed` ambiguity I removed from the taxonomy, reappearing inside the two remaining classes.

### Sample classifications

Five test reviews run through the fine-tuned model, with predicted label and confidence. (The first three are correct, the last two are errors from the analysis above.)

| Review (truncated) | Predicted | Confidence | Correct? |
|---|---|---|---|
| "Nothing beats that random 2-week Terraria phase with friends. This was one of my childhood games..." | positive | 0.98 | yes (true positive) |
| "Got a next gen update, now loading times get exponentially worse every time... Great job Bethesda, guess my SSD is useless." | negative | 0.97 | yes (true negative) |
| "Creepy, surreal, and spectacular looking entry from the original creators of Little Nightmares..." | positive | 0.89 | yes (true positive) |
| "To say Cyberpunk 2077 had a rough launch is like saying the Titanic's maiden trip..." | negative | 0.97 | no (true positive) |
| "I can't believe they kept working on it after launch..." | negative | 0.95 | no (true positive) |

A reasonable correct prediction: the Terraria review ("Nothing beats that random 2-week Terraria phase with friends") is classified `positive` at 0.98 confidence. This is correct and well calibrated because the stance is stated plainly with warm, unambiguous vocabulary and no reversal or irony for the model to misread. Notably, the model also correctly caught the sarcastic negative ("Great job Bethesda, guess my SSD is useless") at 0.97, showing it handles some sarcasm when the complaint ("loading times get exponentially worse") is stated literally alongside it, unlike the purely structural reversals it misses.

## Reflection: what the model learned vs. what I intended

I intended the model to learn the reviewer's overall stance, weighing the whole review and resolving reversals and sarcasm by intent. What it actually learned is closer to a weighted sentiment-word detector: it reads the density and polarity of sentiment vocabulary and predicts accordingly. On plainly written reviews that is enough, and it gets them right with high confidence. But the moment the meaning diverges from the words (a redemption arc, ironic praise, hype that ends in disappointment) the model follows the words and gets it backwards.

This is the gap in one sentence: I labeled by intent, the model learned by vocabulary. With 140 examples there were too few reversal-structured reviews for it to learn that pattern, and DistilBERT's small capacity plus a short context window made the late-arriving "turn" easy to miss. The errors are not random noise; they are a coherent, diagnosable blind spot, which is the more valuable outcome than a high accuracy number with nothing to explain.

## Spec reflection

- **One way the spec helped:** the requirement to run a zero-shot baseline before trusting the fine-tuned model was decisive. Without it I might have reported 0.77 as a "working" classifier. Seeing the 70B baseline hit 0.93 on the same test set reframed the whole project: the question became "why does fine-tuning lose here", which is where the real learning happened. The spec's warning about checking for bugs when the model underperforms is also exactly what surfaced the warmup scheduler bug.
- **One way my implementation diverged:** I planned a three-label taxonomy (`positive`, `negative`, `mixed`) in early planning, but dropped `mixed` before annotating because the boundary was too subjective to label consistently. The spec allows 2 to 4 labels, so this stayed in bounds, but it was a real divergence from my first design. The irony, documented above, is that the ambiguity I removed at the label level came back as the model's main failure mode.

## AI usage

1. **Pipeline debugging and hyperparameter diagnosis.** I gave Claude the full notebook plus the failing results (0.50 accuracy, the all-identical confusion matrix). It identified that `warmup_steps=50` exceeded the roughly 27 total training steps so the learning rate never ramped, and proposed the fix (warmup 10, batch 8, 8 epochs). I ran it and confirmed the jump to about 0.83. I also directed it to propose an attempt to beat the baseline; its suggested aggressive config overfit (0.77 with overconfident errors), which I kept as a documented negative result rather than discarding.
2. **Data tooling and failure-pattern analysis.** I had Claude write the labeling-sheet converter (`scripts/build_csv.py`) and a balance checker, which I used throughout collection. After evaluation I gave it my list of wrong predictions and asked for systematic patterns; it proposed the "structure vs vocabulary" reversal/sarcasm pattern, which I then verified myself by re-reading all seven errors before writing the analysis above. I did all 200 labels by hand; no examples were pre-labeled by an LLM.

## Repository contents

- `planning.md` — design document written before data collection (labels, edge cases, metrics, success criteria, AI tool plan).
- `data/data.csv` — 200 hand-labeled reviews (`text`, `label`, `game`).
- `data/reviews_raw.txt` — the paste-friendly labeling sheet.
- `scripts/build_csv.py` — converts the labeling sheet to the CSV.
- `ai201_project3_takemeter_starter_clean.ipynb` — the Colab fine-tuning notebook.
- `evaluation_results.json`, `confusion_matrix.png` — committed outputs from Colab.