# TakeMeter. Steam Review Sentiment Classifier

> Planning doc. Written before data collection. Updated before each stretch feature.

## 1. Community

**Choice:** Steam game reviews. Public reviews left by players on the Steam store, pulled from the Steam store review pages. I draw from several games on purpose (a mix of well loved titles and divisive ones) so the model learns sentiment language, not the name of one specific game.

**Why this community:** Steam reviews are a massive, active, text heavy discourse space. Players argue about whether a game is worth buying, and the writing ranges from one line praise to long structured critiques. Every review carries a built in ground truth signal (the author marks the game Recommended or Not Recommended), which lets me sanity check my own hand labels. I do not use that flag as the training label. I relabel every example by hand.

**Why the discourse is varied enough to be interesting:** Even reduced to positive vs negative, Steam sentiment is not trivial. Reviews use sarcasm ("oh great, another paid beta"), backhanded praise, and conditional recommendations. The model has to read the overall stance, not just count positive and negative words. That is what keeps a two class sentiment task from being a pure keyword lookup.

## 2. Labels

Two labels, mutually exclusive.

### `positive`
The reviewer is overall satisfied and would recommend the game. Praise dominates the overall stance, even if minor complaints appear.
- "Easily 200 hours in and still finding new things. Worth every penny."
- "Combat feels incredible and the world is gorgeous. Cannot stop playing."

### `negative`
The reviewer is overall dissatisfied and would not recommend the game. Complaints dominate the overall stance, even if minor praise appears.
- "Crashes every 20 minutes since the last update. Unplayable right now."
- "Boring grind, dead servers, and full of microtransactions. Skip it."

### Decision boundary (one sentence)
If the reviewer's overall stance is that the game is worth it, label `positive`. If their overall stance is that it is not worth it, label `negative`.

## 3. Hard edge cases

With two labels, the genuinely hard case is the **ambivalent review** that praises and criticizes in roughly equal measure. There is no `mixed` bucket, so it must be resolved to the nearer pole.

**Decision rule:**
- Label by the reviewer's **final, overall stance**, not by counting pros and cons. The closing sentiment and the recommendation usually reveal it.
- "Buggy mess at launch but they fixed it and now it is fantastic" goes to `positive`. The resolution lands positive.
- "Used to love it, the new update ruined everything" goes to `negative`. The final stance is negative.
- **Sarcasm** flips surface words. "Oh wonderful, another $70 unfinished product" reads positive word by word but is clearly `negative`. Label by real intent.
- A conditional recommendation ("only worth it on sale") leans on whether the reviewer still recommends buying at all. "Great on sale, skip at full price" usually lands `positive` (they do recommend it, with a caveat); "not worth it at any price right now" lands `negative`.

### Three real difficult cases from annotation

**1. Sentiment flip over time (Left 4 Dead 2).**
"After all these years that I loved the game, unfortunately I must change my review to negative. the latest TLS update ruined the fun... the game is no longer hard, but tedious."
The review spends most of its words on past affection. The hard part is that a word count would lean positive. Decision: label by the **final stance**. The reviewer explicitly changes their verdict to negative, so it is `negative`.

**2. Split numeric score (Left 4 Dead 2).**
"the worst game i have played in my life, toxicity in versus is insufferable... overall 1/10 for versus, 9/10 for playing custom maps with friends."
The review gives two opposite scores for two game modes. Decision: do not average the numbers. The opening verdict ("worst game i have played") and the overall framing are dissatisfaction with the core multiplayer experience, so the dominant stance is `negative`.

**3. Lukewarm conditional recommendation (Starfield).**
"I can recommend this game because it's fun for what it is... Can I recommend it full price? If you can afford it... yeah, but if you have other games in mind I would just wait for a sale. Do I regret buying it? No... Leaving the positive review."
The tone is tepid and the recommendation is hedged with conditions. Decision: a conditional recommendation that still says "buy it" (even if only on sale) lands `positive`. The reviewer explicitly states they do not regret it and are leaving a positive review.

## 4. Data collection plan

- **Source:** Public Steam reviews from the store review pages, English only, drawn from several games. Public data only.
- **Target:** At least 200 examples in a single CSV with columns `text`, `label`, `game`. No pre split. The Colab notebook handles the 70/15/15 train/val/test split.
- **Per label target:** Aim for no single label above roughly 70%, ideally close to a 50/50 split. Steam skews heavily positive, so I will deliberately use the "Not Recommended" filter on each game's review page to over collect `negative` reviews.
- **Avoiding leakage and "too easy" results:** I do NOT use the author's Recommended / Not Recommended flag as the training label. It is only a starting suggestion that I correct by hand. The game name is kept in its own column and never inside the `text`, so the model cannot learn "this game = positive" instead of actual sentiment.
- **If a label is underrepresented after 200:** `negative` is the likely shortfall given Steam's positive skew. I will pull more from games with "Mixed" or "Mostly Negative" overall ratings, using the Not Recommended filter, until the split is close to even.

## 5. Evaluation metrics

- **Overall accuracy** (both models). Headline number, but not sufficient alone. With Steam's positive skew, a lazy model can score well by always guessing positive, so accuracy alone hides failure on the minority class.
- **Per class precision, recall, F1.** Needed because the classes may not be equally easy. `negative` recall is the number I watch most: it tells me whether the model actually catches dissatisfied reviews or just defaults to positive.
- **Confusion matrix.** Shows the direction of errors. I expect the interesting errors to be sarcastic or ambivalent reviews predicted as the wrong pole.
- **Why these for this task:** the risk in binary sentiment is a model that rides the majority class. Per class F1 and the confusion matrix expose that directly, where raw accuracy would not.

## 6. Definition of success

- **Minimum bar:** Fine tuned model beats the zero shot Groq baseline on overall accuracy AND on macro F1, so it improves both classes rather than just the majority one.
- **Genuinely useful threshold:** Both per class F1 at or above 0.85 and overall accuracy at or above 0.85. At that level it could power a "recommend vs not" summary signal.
- **Deployment ready bar (aspirational):** `negative` recall at or above 0.85, so genuinely dissatisfied reviews are reliably caught and not lost in the positive majority.
- **Honesty check:** binary sentiment can score very high. If accuracy is above 95%, I will inspect the confusion matrix and re read examples to confirm the model is reading stance and not just keywords, and check for any label leakage.

## 7. AI Tool Plan

This project has no app code to generate, so AI tools help at three specific points.

**Label stress testing (before annotation):**
Give Claude the label definitions plus the edge case rule and ask it to generate 5 to 10 ambivalent or sarcastic reviews that sit near the positive / negative boundary. If I can't classify the generated reviews cleanly with my own rules, the definitions need tightening. Fix before annotating 200.

**Annotation assistance (during collection):**
Decision: the Steam Recommended / Not Recommended flag pre fills a `positive` or `negative` suggestion. I review and correct every single row by hand, since the flag misses sarcasm and people who recommend a game while complaining. The flag is a labor saver, not a label. If I use an LLM to pre label a batch, I will track which rows and disclose it in the README AI usage section.

**Failure analysis (after eval):**
Paste the list of misclassified test reviews into Claude and ask it to surface systematic patterns (sarcasm, short reviews, conditional recommendations, one pole leaking into the other). Then verify each claimed pattern myself by re reading the examples before writing it up. AI proposes patterns. I confirm.

## 8. Pipeline status

- [x] M1. Community and labels chosen
- [x] M2. planning.md (this doc)
- [ ] M3. Collect and annotate 200 or more reviews
- [ ] M4. Groq zero shot baseline on test set
- [ ] M5. Fine tune DistilBERT on Colab T4
- [ ] M6. Evaluate plus README plus demo video

## 9. Possible stretch: third label

If the two class model performs near ceiling and the boundary feels too easy, re introduce a `mixed` class for genuinely ambivalent reviews as a stretch feature. This would be added before collecting the extra `mixed` examples, per the spec's rule to update planning.md before starting a stretch feature.
