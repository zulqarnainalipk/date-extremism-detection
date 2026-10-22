# Data

`dataset.csv` - 3,000 posts, columns `Original_Message`, `Extremism_Label` (`EXTREMIST` / `NON_EXTREMIST`). One row has a missing label (row 2,939 of the file, in the test part).

| | |
|---|---|
| Source | *Social Media Extremism Detection Dataset* by Aditya Suresh, Anthony Lu and Vishnu Iyer (Kaggle) |
| Version | post-competition release, persistent version DOI https://doi.org/10.34740/kaggle/dsv/17368479 |
| Competition | *Social Media Extremism Detection Challenge*, https://www.kaggle.com/competitions/social-media-extremism-detection-challenge |
| Licence | Community Data License Agreement - Permissive, Version 1.0 (https://cdla.dev/permissive-1-0/) |
| SHA-256 | `ea585a13cbdf93c0398be75a2149c0744c7cbec598f3d55597473c8924e36744` |

## Official split used in this repository
Rows 1-2,250 are the training part (the competition's `train.csv`), rows 2,251-3,000 the test part (the competition's `test.csv`). The split is by row position and is applied by `src/date_pipeline.py`.

## What the data providers state
According to the dataset documentation, the posts were drawn from the 400,000-message *Hate Speech Detection curated Dataset* (Nyantudre, 2023) and about 3,000 were hand-labelled by the dataset authors with guidance from the AIEA Lab (UC Santa Cruz). A post is labelled extremist if it "provoked harm to a person or a group of people, whether it be through advocacy for violence, discrimination, or other hurtful sentiments, based off of a characteristic of the group". The providers do **not** report the number or qualifications of annotators, inter-annotator agreement, the original platform(s) or the time period of the posts, and they note that human labelling errors are possible.

## Caveats
- 468 of the 2,249 training posts that can be matched by text (20.8%) carry a different label in this release than in the competition-time `train.csv`. All results here use the labels in this file.
- The texts are lower-cased and stripped of punctuation; emojis appear as descriptive words. The data contain hateful, violent and sexual language.
- The file may be redistributed under the licence above; keep this notice and the citation when you do.
