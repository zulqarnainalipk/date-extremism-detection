# Codebook for categorising model errors

Read the post, the **gold label** and the **model prediction**. Ask: *why would a reasonable classifier get this wrong?* Choose the single best primary category. If two apply, use the one that most directly explains the error and put the other in `category_secondary`.

| Code | Category | Use it when ... | Typical error |
|---|---|---|---|
| 1 | **Implicit or veiled extremism** | The post is extremist (or hostile towards a group) but has no explicit slur or threat: dog whistles, coded slogans, dehumanising metaphors, insinuation, hostile framing of news. | false negative |
| 2 | **Trigger words without group-directed harm** | The post contains profanity, insults or violent words ("kill", "burn", "genocide"), but the target is an individual, an object or nothing in particular, so under the dataset rule it is not extremist. | false positive |
| 3 | **Group mentioned without hostility** | A group or identity term is used neutrally, supportively or descriptively (news, questions, support). | false positive |
| 4 | **Quotation, reported speech or counter-speech** | The post quotes, reports or condemns extremist language, or argues against it. | false positive |
| 5 | **Sarcasm, irony or joke** | The surface meaning is the opposite of the intended meaning, or the post is humorous. | either |
| 6 | **Needs external context** | The post cannot be judged without the thread, an event, an image, the fictional setting or the author's history. | either |
| 7 | **Questionable gold label** | After careful reading you disagree with the gold label (it fits the dataset rule better with the other label). | either |
| 8 | **Explicit hostile content that the model missed** | The post is clearly extremist or group-directed hostility (explicit slur, sexist or racist claim, incitement) and no special difficulty explains the miss; this is a model failure, possibly a shortcut such as treating profanity as non-extremist. | false negative |
| 9 | **Other or unclear** | None of the above fits, the post is too short or garbled, or you skipped it. Add a note. | either |

## Tips
- The dataset rule: extremist = "provoked harm to a person or a group of people, whether it be through advocacy for violence, discrimination, or other hurtful sentiments, based off of a characteristic of the group".
- Use code 7 sparingly and only after considering codes 1-6; it signals label noise, not model error.
- Judge the post as written (the texts are lower-cased and stripped of punctuation), not how it might have looked originally.
- Some posts come in minimal pairs (a hostile post and a near-identical benign rewrite). Judge each post on its own wording.
