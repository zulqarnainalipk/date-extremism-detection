Subject: Help needed: categorising 60 model errors (about 1-2 hours)

Hi <name>,

we are preparing a research paper on detecting extremist content in social media posts and want to analyse why our classifier makes mistakes. Could you help by categorising 60 posts it got wrong? You would be one of two people doing this independently.

**Content warning.** The posts contain hateful, violent and sexual language. Please take breaks, and if a post is too much, skip it (enter 9 and write "skipped" in the note column). No explanation is needed.

**What is attached**
- `annotatorN.csv`: your sheet with 60 posts (N = your number, 1 or 2)
- `codebook.md`: the 9 categories with definitions

**What to do**
1. For each post, read the text, the gold label (the dataset's label) and the model prediction (what the classifier said). The model's prediction is wrong in every row.
2. Decide the single best reason the model got it wrong and type its number (1-9) in the column `category_primary`.
3. Optional: a second number in `category_secondary` if another category also applies, and a short note in `note`.
4. Please do not edit the other columns.

**The categories, in short** (details are in the codebook)
1 = hostile towards a group but implicit or coded, no explicit slur
2 = profanity, insults or violent words, but no group is targeted
3 = a group is mentioned without hostility
4 = quotation, reported speech or counter-speech
5 = sarcasm, irony or joke
6 = cannot be judged without outside context (thread, image, event, fiction)
7 = you disagree with the gold label
8 = clearly hostile or extremist and the model simply missed it
9 = other, unclear or skipped

Use the one number that best explains the error. Use 7 only after considering 1-6. The texts are lower-case with punctuation removed, so judge them as written.

**Important: please work on your own.** Do not discuss the posts or your choices with anyone (including the other annotator) until we have collected both sheets. We measure how well two independent people agree, so it is fine, and expected, that you will sometimes pick different categories. There are no right answers to be graded on.

**Returning it.** Save the file as CSV (keep it as a .csv, UTF-8 if your program asks) and send it back to me by <date>. Please do not share the posts or the file with anyone else.

Thank you very much!
<your name>
