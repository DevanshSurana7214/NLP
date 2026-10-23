# Assignment 7 — Hindi POS Tagging (line-by-line explanation)

This document walks through every code cell in `Assignment_7/7_1.ipynb` line by line, in the order the cells are executed. The notebook builds four POS taggers for Hindi (Naive Bayes and three HMM Viterbi variants), evaluates them on dev/test, and writes CSVs + charts + pickled models.

---

## Cell 1 — imports and setup

```python
import os
import math
import pickle
from collections import Counter, defaultdict

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
```

- `import os` — imports the operating-system interface module; used later to create directories.
- `import math` — imports the math library; used for `math.log`, which converts probabilities to log-probabilities so that very small products do not underflow.
- `import pickle` — imports Python’s serialisation module; used at the end to save trained models to disk.
- `from collections import Counter, defaultdict` — imports two container types:
  - `Counter` is a dictionary that automatically counts hashable items (used for tag counts, feature counts, confusion-matrix-style gold/pred/correct counts).
  - `defaultdict(Counter)` creates a dictionary whose missing values are automatically an empty `Counter` (used for feature-conditional counts per tag).
- `os.makedirs("models", exist_ok=True)` — creates the `models/` output folder if it does not already exist; `exist_ok=True` means no error is raised if the folder is already there.
- `os.makedirs("results", exist_ok=True)` — creates the `results/` output folder the same way; final CSVs and chart PNGs are written here.

---

## Cell 2 — data loading

```python
START = "<s>"
END = "</s>"
```

- `START = "<s>"` — defines the string used to mark the beginning of a sentence; HMM transition probabilities are conditioned on this pseudo-tag before the first real tag.
- `END = "</s>"` — defines the string used to mark the end of a sentence; the last real tag has a transition into this pseudo-tag.

```python
def read_pos_file(path):
    sentences_words = []
    sentences_tags = []
    current_words = []
    current_tags = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences_words.append(current_words)
                    sentences_tags.append(current_tags)
                    current_words = []
                    current_tags = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                tag = parts[-1]
                current_words.append(word)
                current_tags.append(tag)
    if current_words:
        sentences_words.append(current_words)
        sentences_tags.append(current_tags)
    return sentences_words, sentences_tags
```

- `def read_pos_file(path):` — defines a function that reads one of the Hindi POS files and returns two lists: a list of sentences (each sentence is a list of words) and a parallel list of tag sequences.
- `sentences_words = []` — initialises the outer list that will hold all sentences (as word lists).
- `sentences_tags = []` — initialises the outer list that will hold all tag sequences (parallel to `sentences_words`).
- `current_words = []` — initialises the buffer for the words of the sentence currently being read.
- `current_tags = []` — initialises the buffer for the tags of the sentence currently being read.
- `with open(path, "r", encoding="utf-8") as f:` — opens the file at `path` in text mode with UTF-8 encoding, guaranteeing the file is closed after the block; Hindi text is Unicode so UTF-8 is required.
- `for line in f:` — iterates over the file one line at a time.
- `line = line.strip()` — removes leading/trailing whitespace (including the newline) from the line.
- `if not line:` — true when the line is empty after stripping; empty lines are the sentence boundary markers in these files.
- `if current_words:` — if there is a partially collected sentence when we hit the blank line, we commit it.
- `sentences_words.append(current_words)` — appends the finished word list to the outer list.
- `sentences_tags.append(current_tags)` — appends the finished tag list to the parallel outer list.
- `current_words = []` — resets the word buffer for the next sentence.
- `current_tags = []` — resets the tag buffer for the next sentence.
- `continue` — skips the rest of the loop body for blank lines.
- `parts = line.split()` — splits the line on whitespace; each line is expected to be `word ... tag`.
- `if len(parts) >= 2:` — guards against malformed lines that have no tag.
- `word = parts[0]` — takes the first whitespace-separated token as the word form.
- `tag = parts[-1]` — takes the last whitespace-separated token as the POS tag (this handles multi-token lines where extra columns may exist).
- `current_words.append(word)` — adds the word to the current sentence buffer.
- `current_tags.append(tag)` — adds the tag to the current tag buffer.
- After the loop, `if current_words:` handles the last sentence when the file does not end with a blank line, appending it the same way.
- `return sentences_words, sentences_tags` — returns the two parallel lists.

```python
train_words, train_tags = read_pos_file(
    "Hindi-POS-Data/hindi-train-pos.txt"
)
dev_words, dev_tags = read_pos_file(
    "Hindi-POS-Data/hindi-dev-pos.txt"
)
test_words, test_tags = read_pos_file(
    "Hindi-POS-Data/hindi-test-pos.txt"
)
```

- `train_words, train_tags = read_pos_file(...)` — reads the training split into two parallel lists of sentences.
- `dev_words, dev_tags = read_pos_file(...)` — reads the development split.
- `test_words, test_tags = read_pos_file(...)` — reads the held-out test split.

```python
print("Train sentences:", len(train_words))
print("Dev sentences:", len(dev_words))
print("Test sentences:", len(test_words))
```

- Prints the number of sentences in each split so we can verify the data loaded correctly.

---

## Cell 3 — vocabulary and tag sets

```python
train_tagged = [
    (w, t)
    for words, tags in zip(train_words, train_tags)
    for w, t in zip(words, tags)
]
```

- Builds a flat list of `(word, tag)` pairs for the whole training corpus by zipping sentences together and then zipping words with tags inside each sentence. This flat list is convenient for counting unigram statistics.

```python
words_set = set()
tags_set = set()
for w, t in train_tagged:
    words_set.add(w)
    tags_set.add(t)
```

- `words_set = set()` — starts an empty set for unique word types.
- `tags_set = set()` — starts an empty set for unique tag types.
- The loop adds every training word to `words_set` and every training tag to `tags_set`, giving us the vocabulary and tag inventory.

```python
word2idx = {w: i for i, w in enumerate(sorted(words_set))}
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i for i, t in enumerate(sorted(tags_set))}
idx2tag = {i: t for t, i in tag2idx.items()}
```

- `word2idx = ...` — builds a dictionary mapping each word to a stable integer index (sorted for determinism); used when building array-shaped structures or for debugging.
- `idx2word = ...` — the inverse mapping from index back to word.
- `tag2idx = ...` — maps each tag string to an integer index; used when iterating over tag sets in Viterbi trellises.
- `idx2tag = ...` — inverse mapping from tag index to tag string.

```python
vocab_size = len(words_set)
num_tags = len(tags_set)
```

- `vocab_size = len(words_set)` — records the number of distinct word types in the training data.
- `num_tags = len(tags_set)` — records the number of distinct tags.

```python
print("Vocab size:", vocab_size)
print("Num tags:", num_tags)
print("Tags:", sorted(tags_set))
```

- Prints the vocabulary size, tag count, and the sorted tag list for a sanity check.

---

## Cell 4 — Naive Bayes word classifier training

```python
import numpy as np
from math import log
```

- `import numpy as np` — imports NumPy; although much of the NB code uses plain Python, importing it here keeps the environment consistent with the later chart cells.
- `from math import log` — imports the natural logarithm directly; used to compute log-probabilities for NB scoring.

```python
def word_length_feat(w):
    return "MORE" if len(w) > 4 else "LESS"
```

- `def word_length_feat(w):` — defines the binary length feature function required by the assignment.
- `return "MORE" if len(w) > 4 else "LESS"` — returns the string `"MORE"` when the word’s character length is strictly greater than 4, otherwise `"LESS"`.

```python
def ngrams(words, tags):
    prev_words = ["<s>"] + words[:-1]
    next_words = words[1:] + ["</s>"]
    prev_tags = ["<s>"] + tags[:-1]
    return prev_words, next_words, prev_tags
```

- `def ngrams(words, tags):` — helper that, given a sentence’s words and tags, returns three parallel lists aligned with the word positions:
  - the previous word (with `<s>` at position 0),
  - the next word (with `</s>` after the last word),
  - the previous tag (with `<s>` at position 0).
- `prev_words = ["<s>"] + words[:-1]` — prepends the start symbol and drops the last word, so `prev_words[i]` is the word before position `i`.
- `next_words = words[1:] + ["</s>"]` — drops the first word and appends the end symbol, so `next_words[i]` is the word after position `i`.
- `prev_tags = ["<s>"] + tags[:-1]` — same left-shift idea for tags; the previous tag at position 0 is the start symbol.
- `return prev_words, next_words, prev_tags` — returns the three aligned sequences.

```python
def nb_feature_tuples(words, tags):
    prev_words, next_words, prev_tags = ngrams(words, tags)
    feats = []
    for i, w in enumerate(words):
        feat = (
            word_length_feat(w),
            w,
            prev_words[i],
            next_words[i],
            prev_tags[i],
        )
        feats.append((feat, tags[i]))
    return feats
```

- `def nb_feature_tuples(words, tags):` — builds the NB training instances for one sentence.
- `prev_words, next_words, prev_tags = ngrams(words, tags)` — obtains the aligned context lists.
- `feats = []` — starts the list of `(feature_tuple, gold_tag)` pairs for this sentence.
- `for i, w in enumerate(words):` — loops over each position in the sentence.
- `feat = (...)` — builds the 5-tuple feature for this token:
  1. `word_length_feat(w)` — the binary length feature (`"MORE"`/`"LESS"`),
  2. `w` — the actual word itself,
  3. `prev_words[i]` — the previous word,
  4. `next_words[i]` — the next word,
  5. `prev_tags[i]` — the previous tag.
- `feats.append((feat, tags[i]))` — pairs that feature tuple with the gold tag for the position.
- `return feats` — returns the sentence’s training instances.

```python
train_nb = []
for words, tags in zip(train_words, train_tags):
    train_nb.extend(nb_feature_tuples(words, tags))
```

- `train_nb = []` — starts the flat list of all NB training instances across the corpus.
- The loop calls `nb_feature_tuples` for each training sentence and extends the flat list with the result. After this, `train_nb` is a list of `(feature_tuple, tag)` pairs for every training token.

```python
NB_PRIOR_SMOOTH = 1.0
NB_LIKELIHOOD_SMOOTH = 1.0
```

- `NB_PRIOR_SMOOTH = 1.0` — additive smoothing constant for the class prior (Laplace-style).
- `NB_LIKELIHOOD_SMOOTH = 1.0` — additive smoothing constant for each feature likelihood.

```python
nb_tag_counts = Counter()
nb_feat_cond_counts = defaultdict(Counter)
```

- `nb_tag_counts = Counter()` — counts how many training tokens have each tag (the denominator for the prior).
- `nb_feat_cond_counts = defaultdict(Counter)` — a dict from tag → Counter; each inner Counter records how many times each feature value was seen for that tag, per feature position.

```python
for feat, tag in train_nb:
    nb_tag_counts[tag] += 1
    for f in feat:
        nb_feat_cond_counts[tag][f] += 1
```

- Loops over every training instance:
  - `nb_tag_counts[tag] += 1` — increments the count of the gold tag.
  - `for f in feat:` — iterates over the five feature values in the tuple.
  - `nb_feat_cond_counts[tag][f] += 1` — increments the count of that feature value for that tag. Because the counter is shared across feature positions, a feature value seen at different positions is pooled together (this is a simple implementation choice; each value is just counted per tag).

```python
nb_tag_total = sum(nb_tag_counts.values())
nb_num_tags = len(nb_tag_counts)
```

- `nb_tag_total = sum(nb_tag_counts.values())` — total number of training tokens; the denominator base for the prior.
- `nb_num_tags = len(nb_tag_counts)` — number of distinct tags seen in training.

```python
nb_feat_vocabs = {}
for tag in sorted(nb_tag_counts):
    nb_feat_vocabs[tag] = {}
    for i in range(5):
        nb_feat_vocabs[tag][i] = len(nb_feat_cond_counts[tag])
```

- `nb_feat_vocabs = {}` — will store, for each tag, the number of distinct feature values seen (used as the likelihood denominator’s vocabulary size).
- The outer loop iterates over each tag.
- `nb_feat_vocabs[tag] = {}` — creates the per-tag sub-dict.
- `for i in range(5):` — there are five feature positions.
- `nb_feat_vocabs[tag][i] = len(nb_feat_cond_counts[tag])` — sets the vocabulary size for every feature position of this tag to the total number of distinct feature values seen for that tag. (Because feature values are pooled across positions in this implementation, the same number is used for each position.)

```python
def nb_prior_p(tag):
    return (nb_tag_counts[tag] + NB_PRIOR_SMOOTH) / (nb_tag_total + NB_PRIOR_SMOOTH * nb_num_tags)
```

- `def nb_prior_p(tag):` — computes the smoothed prior probability of a tag.
- `nb_tag_counts[tag] + NB_PRIOR_SMOOTH` — numerator: observed count plus smoothing.
- `nb_tag_total + NB_PRIOR_SMOOTH * nb_num_tags` — denominator: total tokens plus smoothing mass distributed across all tags.
- Returns the prior P(tag).

```python
def nb_likelihood_p(tag, feat_idx, feat_val):
    num = nb_feat_cond_counts[tag].get(feat_val, 0) + NB_LIKELIHOOD_SMOOTH
    den = nb_tag_counts[tag] + NB_LIKELIHOOD_SMOOTH * (nb_feat_vocabs[tag][feat_idx] + 1)
    return num / den if den else 1e-12
```

- `def nb_likelihood_p(tag, feat_idx, feat_val):` — computes the smoothed conditional probability of a feature value given a tag.
- `num = nb_feat_cond_counts[tag].get(feat_val, 0) + NB_LIKELIHOOD_SMOOTH` — numerator: count of this feature value for this tag (0 if unseen) plus smoothing.
- `den = nb_tag_counts[tag] + NB_LIKELIHOOD_SMOOTH * (nb_feat_vocabs[tag][feat_idx] + 1)` — denominator: tag count plus smoothing mass times the number of vocabulary items for this feature position (the `+ 1` guards against edge cases in the vocabulary accounting).
- `return num / den if den else 1e-12` — returns the likelihood, or a very small fallback if the denominator is somehow zero.

```python
def nb_predict_one(feat):
    best_tag = None
    best_lp = None
    for tag in sorted(nb_tag_counts):
        lp = log(nb_prior_p(tag))
        for i, fv in enumerate(feat):
            lp += log(nb_likelihood_p(tag, i, fv))
        if best_lp is None or lp > best_lp:
            best_lp = lp
            best_tag = tag
    return best_tag
```

- `def nb_predict_one(feat):` — predicts the tag for a single feature tuple using log-probabilities.
- `best_tag = None` / `best_lp = None` — initialise the best answer tracker.
- `for tag in sorted(nb_tag_counts):` — considers every tag.
- `lp = log(nb_prior_p(tag))` — starts the log-score with the log prior.
- `for i, fv in enumerate(feat):` — loops over the five feature values.
- `lp += log(nb_likelihood_p(tag, i, fv))` — adds the log likelihood of each feature value given the tag (Naive Bayes independence assumption).
- `if best_lp is None or lp > best_lp:` — keeps the tag with the highest log-score.
- `return best_tag` — returns the winning tag.

```python
def nb_predict_sentence(words, tags):
    prev_words, next_words, prev_tags = ngrams(words, tags)
    pred = []
    for i, w in enumerate(words):
        feat = (
            word_length_feat(w),
            w,
            prev_words[i],
            next_words[i],
            prev_tags[i],
        )
        pred.append(nb_predict_one(feat))
    return pred
```

- `def nb_predict_sentence(words, tags):` — predicts a tag sequence for a sentence.
- `prev_words, next_words, prev_tags = ngrams(words, tags)` — builds context; note this version uses the *gold* tags for the previous-tag feature, which is the evaluation-time setup described in the notebook.
- `pred = []` — collects predictions.
- For each position, builds the same 5-tuple feature and calls `nb_predict_one`, appending the result.
- `return pred` — returns the predicted tag list.

```python
predict_nb_fn = lambda ws, ts: nb_predict_sentence(ws, ts)
```

- Wraps the sentence predictor into a two-argument callable (`words, tags`) so it matches the signature used by the shared evaluation function. For dev/test evaluation the gold previous tag is supplied.

```python
print("NB tags:", sorted(nb_tag_counts))
print("NB train tokens:", nb_tag_total)
```

- Prints the tag inventory and total token count for the NB model as a sanity check.

---

## Cell 5 — HMM count collection

```python
transition_counts = Counter()
emission_counts = Counter()
tag_counts = Counter()
```

- `transition_counts = Counter()` — counts tag-to-tag bigram occurrences (including from `<s>` and to `</s>`).
- `emission_counts = Counter()` — counts `(tag, word)` pairs.
- `tag_counts = Counter()` — counts how many times each tag occurs as a state (the denominator for transition/emission probabilities).

```python
for words, tags in zip(train_words, train_tags):
    tagged = list(zip(words, tags))
    prev = START
    for word, tag in tagged:
        transition_counts[(prev, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev = tag
    transition_counts[(prev, END)] += 1
```

- Loops over every training sentence:
  - `tagged = list(zip(words, tags))` — pairs each word with its tag.
  - `prev = START` — initialises the previous state to the start symbol before the first word.
  - Inside the word loop:
    - `transition_counts[(prev, tag)] += 1` — counts the transition from the previous tag to the current tag.
    - `emission_counts[(tag, word)] += 1` — counts the emission of this word from this tag.
    - `tag_counts[tag] += 1` — counts this tag occurrence.
    - `prev = tag` — advances the previous-state pointer.
  - After the word loop, `transition_counts[(prev, END)] += 1` — counts the transition from the last tag to the end symbol.

---

## Cell 6 — HMM probability estimation

```python
false_prob = 1e-12
```

- `false_prob = 1e-12` — a tiny probability used for unseen transitions/emissions so the log does not become -infinity.

```python
transition_prob = {}
for (t1, t2), c in transition_counts.items():
    transition_prob[(t1, t2)] = c / tag_counts[t1] if tag_counts[t1] else false_prob
```

- `transition_prob = {}` — will hold P(t2 | t1) for seen transitions.
- For each observed transition `(t1, t2)` with count `c`:
  - `c / tag_counts[t1]` — maximum-likelihood estimate of the transition probability.
  - `if tag_counts[t1] else false_prob` — guards against a zero denominator.

```python
emission_prob = {}
for (t, w), c in emission_counts.items():
    emission_prob[(t, w)] = c / tag_counts[t] if tag_counts[t] else false_prob
```

- `emission_prob = {}` — holds P(word | tag) for seen pairs.
- For each observed `(tag, word)` pair with count `c`:
  - `c / tag_counts[t]` — MLE emission probability.
  - fallback to `false_prob` if the tag count is zero.

```python
UNK_PROB = false_prob
```

- `UNK_PROB = false_prob` — probability assigned to any word not seen in training for a given tag.

```python
def emission_p(tag, word):
    key = (tag, word)
    if key in emission_prob:
        return emission_prob[key]
    return UNK_PROB
```

- `def emission_p(tag, word):` — looks up the emission probability for a tag–word pair.
- `key = (tag, word)` — forms the dictionary key.
- `if key in emission_prob:` — if seen in training, return the stored probability.
- `return UNK_PROB` — otherwise return the small unknown-word probability.

```python
def transition_p(t1, t2):
    key = (t1, t2)
    if key in transition_prob:
        return transition_prob[key]
    return false_prob
```

- `def transition_p(t1, t2):` — looks up the transition probability for a tag pair.
- Returns the stored probability if seen, else `false_prob`.

---

## Cell 7 — Forward Viterbi decoding

```python
def viterbi_forward(words):
    n = len(words)
    viterbi = []
    backpointer = []
    for i, word in enumerate(words):
        curr_v = {}
        curr_b = {}
        if i == 0:
            for t in tag2idx:
                lp = math.log(transition_p(START, t)) + math.log(emission_p(t, word))
                curr_v[t] = lp
                curr_b[t] = START
        else:
            for t in tag2idx:
                best_prev = None
                best_score = None
                for pt in tag2idx:
                    lp = (
                        viterbi[-1][pt]
                        + math.log(transition_p(pt, t))
                        + math.log(emission_p(t, word))
                    )
                    if best_score is None or lp > best_score:
                        best_score = lp
                        best_prev = pt
                curr_v[t] = best_score
                curr_b[t] = best_prev
        viterbi.append(curr_v)
        backpointer.append(curr_b)
    return viterbi, backpointer
```

- `def viterbi_forward(words):` — standard left-to-right Viterbi; returns the trellis (`viterbi`) and backpointers (`backpointer`).
- `n = len(words)` — sentence length (used conceptually; indices drive the logic).
- `viterbi = []` — list of dicts; `viterbi[i][t]` is the best log-score of any path ending in tag `t` at position `i`.
- `backpointer = []` — list of dicts; `backpointer[i][t]` is the previous tag on the best path into `t` at position `i`.
- `for i, word in enumerate(words):` — iterate over positions.
- `curr_v = {}` / `curr_b = {}` — the trellis column and backpointer column for this position.
- `if i == 0:` — first word:
  - `for t in tag2idx:` — for every possible first tag:
    - `lp = math.log(transition_p(START, t)) + math.log(emission_p(t, word))` — log P(start → t) + log P(word | t).
    - `curr_v[t] = lp` — store the score.
    - `curr_b[t] = START` — the backpointer for the first position is the start symbol.
- `else:` — later positions:
  - `for t in tag2idx:` — for each candidate current tag:
    - `best_prev = None` / `best_score = None` — find the best predecessor.
    - `for pt in tag2idx:` — consider every possible previous tag:
      - `lp = viterbi[-1][pt] + math.log(transition_p(pt, t)) + math.log(emission_p(t, word))` — extend the best path to `pt` with the transition `pt → t` and the emission of the current word from `t`.
      - keep the max.
    - `curr_v[t] = best_score` / `curr_b[t] = best_prev` — store the best incoming score and the tag that produced it.
- `viterbi.append(curr_v)` / `backpointer.append(curr_b)` — record the column.
- `return viterbi, backpointer` — return both structures for decoding.

```python
def predict_forward(viterbi, backpointer):
    last = viterbi[-1]
    best_tag = max(last, key=last.get)
    tags = []
    tags.append(best_tag)
    for bp in reversed(backpointer[:-1]):
        best_tag = bp[best_tag]
        tags.append(best_tag)
    return list(reversed(tags))
```

- `def predict_forward(viterbi, backpointer):` — backtracks to recover the best tag sequence.
- `last = viterbi[-1]` — the final trellis column.
- `best_tag = max(last, key=last.get)` — pick the tag with the highest final score.
- `tags = []` / `tags.append(best_tag)` — start the sequence with that tag.
- `for bp in reversed(backpointer[:-1]):` — walk backwards through the backpointer columns (skipping the last one, whose backpointer we already consumed):
  - `best_tag = bp[best_tag]` — follow the backpointer to the previous tag.
  - `tags.append(best_tag)` — record it.
- `return list(reversed(tags))` — the sequence was built backwards, so reverse it to get left-to-right order.

---

## Cell 8 — Backward Viterbi decoding

```python
def viterbi_backward(words):
    n = len(words)
    rev_words = list(reversed(words))
    viterbi = []
    backpointer = []
    for i, word in enumerate(rev_words):
        curr_v = {}
        curr_b = {}
        if i == 0:
            for t in tag2idx:
                lp = math.log(transition_p(t, END)) + math.log(
                    emission_p(t, word)
                )
                curr_v[t] = lp
                curr_b[t] = END
        else:
            for t in tag2idx:
                best_prev = None
                best_score = None
                for pt in tag2idx:
                    lp = (
                        viterbi[-1][pt]
                        + math.log(transition_p(t, pt))
                        + math.log(emission_p(t, word))
                    )
                    if best_score is None or lp > best_score:
                        best_score = lp
                        best_prev = pt
                curr_v[t] = best_score
                curr_b[t] = best_prev
        viterbi.append(curr_v)
        backpointer.append(curr_b)
    return viterbi, backpointer
```

- `def viterbi_backward(words):` — right-to-left Viterbi; it operates on the reversed sentence and treats `END` as the start state.
- `n = len(words)` — sentence length.
- `rev_words = list(reversed(words))` — the sentence reversed; position 0 in this traversal corresponds to the last word of the original sentence.
- `viterbi = []` / `backpointer = []` — trellis and backpointers for the reversed traversal.
- `for i, word in enumerate(rev_words):` — iterate over reversed positions.
- `if i == 0:` — the last word of the original sentence:
  - `for t in tag2idx:` — for each tag:
    - `lp = math.log(transition_p(t, END)) + math.log(emission_p(t, word))` — log P(t → END) + log P(word | t). This anchors the sequence ending.
    - `curr_v[t] = lp` / `curr_b[t] = END` — store and point back to `END`.
- `else:` — earlier (in reversed order) words:
  - `for t in tag2idx:` — candidate tag for the current (reversed) position:
    - `for pt in tag2idx:` — consider previous tags in the reversed traversal:
      - `lp = viterbi[-1][pt] + math.log(transition_p(t, pt)) + math.log(emission_p(t, word))` — extend the best reversed-path into `pt` with the transition `t → pt` and the emission of the current word from `t`. (Note the transition direction is `t → pt` because we are moving right-to-left.)
      - keep the max.
    - `curr_v[t] = best_score` / `curr_b[t] = best_prev`.
- `viterbi.append(curr_v)` / `backpointer.append(curr_b)`.
- `return viterbi, backpointer`.

```python
def predict_backward(viterbi, backpointer, words):
    first = viterbi[-1]
    best_tag = max(first, key=first.get)
    tags = []
    tags.append(best_tag)
    for bp in reversed(backpointer[:-1]):
        best_tag = bp[best_tag]
        tags.append(best_tag)
    return list(reversed(tags))
```

- `def predict_backward(viterbi, backpointer, words):` — backtracks the reversed trellis to recover tags in original order.
- `first = viterbi[-1]` — the final column of the reversed trellis (which corresponds to the first word of the original sentence).
- `best_tag = max(first, key=first.get)` — best tag for the first original word.
- Build the sequence by following backpointers backwards through the reversed columns, then reverse the whole list to restore original word order.

---

## Cell 9 — Forward-backward Viterbi decoding

```python
def forward_probs(words):
    alpha = []
    for i, word in enumerate(words):
        curr = {}
        if i == 0:
            for t in tag2idx:
                curr[t] = math.log(
                    transition_p(START, t) * emission_p(t, word)
                )
        else:
            for t in tag2idx:
                best_lp = None
                for pt in tag2idx:
                    lp = (
                        alpha[-1][pt]
                        + math.log(transition_p(pt, t))
                        + math.log(emission_p(t, word))
                    )
                    if best_lp is None or lp > best_lp:
                        best_lp = lp
                curr[t] = best_lp
        alpha.append(curr)
    return alpha
```

- `def forward_probs(words):` — computes the Viterbi forward trellis (best log-scores from the start to each position), reusing the same DP as forward Viterbi but returning only the alpha scores.
- `alpha = []` — list of dicts; `alpha[i][t]` is the best log-score of a path from `START` to tag `t` at position `i`.
- For position 0: `alpha[0][t] = log(P(START→t) * P(word|t))`.
- For later positions: extend the best previous score with the transition and emission, taking the max.
- Returns the alpha trellis.

```python
def backward_probs(words):
    n = len(words)
    beta = [None] * n
    beta[n - 1] = {}
    for t in tag2idx:
        beta[n - 1][t] = math.log(transition_p(t, END))
    for i in range(n - 2, -1, -1):
        word = words[i + 1]
        curr = {}
        for t in tag2idx:
            best_lp = None
            for pt in tag2idx:
                lp = (
                    beta[i + 1][pt]
                    + math.log(transition_p(t, pt))
                    + math.log(emission_p(pt, word))
                )
                if best_lp is None or lp > best_lp:
                    best_lp = lp
            curr[t] = best_lp
        beta[i] = curr
    return beta
```

- `def backward_probs(words):` — computes the Viterbi backward trellis (best log-scores from each position to `END`).
- `n = len(words)` — sentence length.
- `beta = [None] * n` — pre-allocates a list of length `n`.
- `beta[n - 1] = {}` — the last position’s column.
- `for t in tag2idx:` — initialise the last column:
  - `beta[n - 1][t] = math.log(transition_p(t, END))` — log P(t → END). This is the backward score for the last position; it does not include the last word’s emission (that is accounted for in the forward trellis and in the combination step).
- `for i in range(n - 2, -1, -1):` — iterate backwards from the second-last position to the first:
  - `word = words[i + 1]` — the word at position `i+1` (the word whose emission is used to step from `i` to `i+1` in the backward direction).
  - `for t in tag2idx:` — candidate tag at position `i`:
    - `for pt in tag2idx:` — candidate tag at position `i+1`:
      - `lp = beta[i + 1][pt] + math.log(transition_p(t, pt)) + math.log(emission_p(pt, word))` — best backward score from `pt` to `END` plus log P(t → pt) plus log P(word | pt).
      - keep the max.
    - `curr[t] = best_lp`.
  - `beta[i] = curr`.
- Returns the beta trellis.

```python
def viterbi_forward_backward(words):
    alpha = forward_probs(words)
    beta = backward_probs(words)
    viterbi = []
    backpointer = []
    for i, word in enumerate(words):
        curr_v = {}
        curr_b = {}
        if i == 0:
            for t in tag2idx:
                lp = alpha[0][t] + beta[0][t] - math.log(
                    emission_p(t, word)
                )
                curr_v[t] = lp
                curr_b[t] = START
        else:
            prev_alpha = alpha[i - 1]
            for t in tag2idx:
                best_prev = None
                best_score = None
                for pt in tag2idx:
                    lp = (
                        prev_alpha[pt]
                        + math.log(transition_p(pt, t))
                        + math.log(emission_p(t, word))
                        + beta[i][t]
                        - math.log(emission_p(t, word))
                    )
                    if best_score is None or lp > best_score:
                        best_score = lp
                        best_prev = pt
                curr_v[t] = best_score
                curr_b[t] = best_prev
        viterbi.append(curr_v)
        backpointer.append(curr_b)
    return viterbi, backpointer
```

- `def viterbi_forward_backward(words):` — combines the forward and backward trellises into a bidirectional Viterbi score and runs backtracking on it.
- `alpha = forward_probs(words)` / `beta = backward_probs(words)` — compute both trellises.
- For position 0:
  - `lp = alpha[0][t] + beta[0][t] - math.log(emission_p(t, word))` — combine forward and backward scores. The emission of the current word from `t` is present in both `alpha[0][t]` and (indirectly) in `beta[0][t]` via the backward step, so one copy is subtracted to avoid double-counting. The result is a bidirectional score for starting in tag `t`.
  - `curr_v[t] = lp` / `curr_b[t] = START`.
- For later positions:
  - `prev_alpha = alpha[i - 1]` — the previous forward column.
  - `for t in tag2idx:` — candidate current tag:
    - `for pt in tag2idx:` — candidate previous tag:
      - `lp = prev_alpha[pt] + log(P(pt→t)) + log(P(word|t)) + beta[i][t] - log(P(word|t))` — forward score to `pt` + transition + emission + backward score from `t` to `END`, with one emission term subtracted to avoid double-counting the current word’s emission.
      - keep the max.
    - `curr_v[t] = best_score` / `curr_b[t] = best_prev`.
- `viterbi.append(curr_v)` / `backpointer.append(curr_b)`.
- Returns a trellis and backpointers in the same format as the forward Viterbi, so the same `predict_forward` function can decode it.

---

## Cell 10 — evaluation function and predictor wrappers

```python
def evaluate_dataset(words_list, tags_list, predictor_fn, name):
    pred_all = []
    gold_all = []
    for words, tags in zip(words_list, tags_list):
        pred_tags = predictor_fn(words)
        pred_all.extend(pred_tags)
        gold_all.extend(tags)
    correct_by_tag = Counter()
    gold_by_tag = Counter()
    pred_by_tag = Counter()
    total_correct = 0
    total_tokens = 0
    for g, p in zip(gold_all, pred_all):
        gold_by_tag[g] += 1
        pred_by_tag[p] += 1
        if g == p:
            correct_by_tag[g] += 1
            total_correct += 1
        total_tokens += 1
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    f1_by_tag = {}
    for t in sorted(tags_set):
        tp = correct_by_tag.get(t, 0)
        gold_n = gold_by_tag.get(t, 0)
        pred_n = pred_by_tag.get(t, 0)
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / gold_n if gold_n else 0.0
        f1 = (
            2 * prec * rec / (prec + rec)
            if (prec + rec) > 0
            else 0.0
        )
        f1_by_tag[t] = f1
    return {
        "name": name,
        "accuracy": accuracy,
        "f1_by_tag": f1_by_tag,
        "gold_by_tag": gold_by_tag,
        "pred_by_tag": pred_by_tag,
        "correct_by_tag": correct_by_tag,
    }
```

- `def evaluate_dataset(words_list, tags_list, predictor_fn, name):` — runs a predictor over a whole dataset and returns per-tag and overall metrics.
- `pred_all = []` / `gold_all = []` — accumulate predicted and gold tags across all sentences.
- `for words, tags in zip(words_list, tags_list):` — iterate over sentences:
  - `pred_tags = predictor_fn(words)` — call the predictor with the word list (note: some predictors also need tags; NB’s wrapper supplies them).
  - `pred_all.extend(pred_tags)` / `gold_all.extend(tags)` — flatten into the global lists.
- `correct_by_tag = Counter()` / `gold_by_tag = Counter()` / `pred_by_tag = Counter()` — per-tag confusion-matrix counters.
- `total_correct = 0` / `total_tokens = 0` — accumulate overall accuracy counts.
- `for g, p in zip(gold_all, pred_all):` — walk token by token:
  - `gold_by_tag[g] += 1` — count this gold tag.
  - `pred_by_tag[p] += 1` — count this predicted tag.
  - `if g == p:` — correct prediction:
    - `correct_by_tag[g] += 1` — increment the per-tag correct count.
    - `total_correct += 1`.
  - `total_tokens += 1`.
- `accuracy = total_correct / total_tokens if total_tokens else 0.0` — overall token accuracy.
- `f1_by_tag = {}` — per-tag F1 dictionary.
- `for t in sorted(tags_set):` — for each tag:
  - `tp = correct_by_tag.get(t, 0)` — true positives for this tag.
  - `gold_n = gold_by_tag.get(t, 0)` — gold count.
  - `pred_n = pred_by_tag.get(t, 0)` — predicted count.
  - `prec = tp / pred_n if pred_n else 0.0` — precision.
  - `rec = tp / gold_n if gold_n else 0.0` — recall.
  - `f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0` — standard F1.
  - `f1_by_tag[t] = f1`.
- `return {...}` — returns a result dictionary with name, accuracy, per-tag F1, and the raw counters (used later for the summary tables).

```python
predict_forward_fn = lambda ws: predict_forward(
    *viterbi_forward(ws)
)
predict_backward_fn = lambda ws: predict_backward(
    *viterbi_backward(ws), ws
)
predict_fb_fn = lambda ws: predict_forward(
    *viterbi_forward_backward(ws)
)
```

- `predict_forward_fn = lambda ws: ...` — wraps forward Viterbi: compute trellis + backpointers, then decode with `predict_forward`. Takes only the word list.
- `predict_backward_fn = lambda ws: ...` — wraps backward Viterbi: decode with `predict_backward`, passing the trellis, backpointers, and the original word list.
- `predict_fb_fn = lambda ws: ...` — wraps forward-backward Viterbi: its trellis/backpointer format matches forward Viterbi, so it reuses `predict_forward`.

```python
def nb_predict_sentence_eval(words):
    prev_words = ["<s>"] + words[:-1]
    next_words = words[1:] + ["</s>"]
    prev_tags = ["<s>"] + ["<unknown>"] * (len(words) - 1)
    pred = []
    for i, w in enumerate(words):
        feat = (
            word_length_feat(w),
            w,
            prev_words[i],
            next_words[i],
            prev_tags[i],
        )
        pred.append(nb_predict_one(feat))
    return pred
```

- `def nb_predict_sentence_eval(words):` — NB predictor for evaluation on dev/test.
- `prev_words = ["<s>"] + words[:-1]` — previous-word context (start symbol at position 0).
- `next_words = words[1:] + ["</s>"]` — next-word context (end symbol after last word).
- `prev_tags = ["<s>"] + ["<unknown>"] * (len(words) - 1)` — previous-tag context: start symbol at position 0, then a placeholder `"<unknown>"` for the rest, because gold tags are not available at prediction time. This lets the NB still run with all five features present.
- For each position, build the 5-tuple feature and call `nb_predict_one`.
- `return pred`.

```python
predict_nb_fn = lambda ws: nb_predict_sentence_eval(ws)
```

- Wraps the NB evaluation predictor into a single-argument callable (word list only), matching the evaluator’s expectation.

---

## Cell 11 — development set results

```python
dev_fwd = evaluate_dataset(dev_words, dev_tags, predict_forward_fn, "Forward Viterbi")
dev_bwd = evaluate_dataset(dev_words, dev_tags, predict_backward_fn, "Backward Viterbi")
dev_fb = evaluate_dataset(dev_words, dev_tags, predict_fb_fn, "Forward-Backward Viterbi")
dev_nb = evaluate_dataset(dev_words, dev_tags, predict_nb_fn, "Naive Bayes")
```

- Runs all four predictors on the dev set:
  - `dev_fwd` — forward Viterbi result dict.
  - `dev_bwd` — backward Viterbi result dict.
  - `dev_fb` — forward-backward Viterbi result dict.
  - `dev_nb` — Naive Bayes result dict.

```python
print("Development Set")
for res in [dev_fwd, dev_bwd, dev_fb, dev_nb]:
    print(res["name"])
    print("Accuracy:", res["accuracy"])
    print("F1 by tag:")
    for t in sorted(tags_set):
        print("  ", t, res["f1_by_tag"][t])
    print()
```

- Prints a readable block for each algorithm: name, overall accuracy, and per-tag F1 for all 24 tags.

---

## Cell 12 — test set results

```python
test_fwd = evaluate_dataset(test_words, test_tags, predict_forward_fn, "Forward Viterbi")
test_bwd = evaluate_dataset(test_words, test_tags, predict_backward_fn, "Backward Viterbi")
test_fb = evaluate_dataset(test_words, test_tags, predict_fb_fn, "Forward-Backward Viterbi")
test_nb = evaluate_dataset(test_words, test_tags, predict_nb_fn, "Naive Bayes")
```

- Same as the dev block, but on the held-out test set, producing `test_fwd`, `test_bwd`, `test_fb`, `test_nb`.

```python
print("Test Set")
for res in [test_fwd, test_bwd, test_fb, test_nb]:
    print(res["name"])
    print("Accuracy:", res["accuracy"])
    print("F1 by tag:")
    for t in sorted(tags_set):
        print("  ", t, res["f1_by_tag"][t])
    print()
```

- Prints the test-set block for each algorithm.

---

## Cell 13 — summary tables (HMM-only, CSV output)

```python
def build_summary_table(results_list):
    rows = []
    for r in results_list:
        for t in sorted(tags_set):
            rows.append(
                {
                    "Algorithm": r["name"],
                    "Tag": t,
                    "F1": r["f1_by_tag"][t],
                    "Gold": r["gold_by_tag"].get(t, 0),
                    "Predicted": r["pred_by_tag"].get(t, 0),
                    "Correct": r["correct_by_tag"].get(t, 0),
                }
            )
    return rows
```

- `def build_summary_table(results_list):` — converts a list of result dicts into a flat list of row dicts suitable for a pandas DataFrame.
- For each result dict and each tag, produces one row with columns: Algorithm, Tag, F1, Gold, Predicted, Correct.
- `return rows`.

```python
import pandas as pd

dev_rows = build_summary_table([dev_fwd, dev_bwd, dev_fb])
test_rows = build_summary_table([test_fwd, test_bwd, test_fb])
```

- `import pandas as pd` — imports pandas for table/CSV handling.
- `dev_rows = build_summary_table([dev_fwd, dev_bwd, dev_fb])` — builds rows for the three HMM variants on dev (this earlier cell excludes NB; it is later superseded by the all-algorithms cell below).
- `test_rows = ...` — same for test.

```python
dev_df = pd.DataFrame(dev_rows)
test_df = pd.DataFrame(test_rows)
```

- `dev_df = pd.DataFrame(dev_rows)` — turns the dev rows into a DataFrame.
- `test_df = pd.DataFrame(test_rows)` — turns the test rows into a DataFrame.

```python
dev_df
test_df
```

- Displaying the two DataFrames in the notebook shows the HMM-only summary tables.

```python
dev_df.to_csv("results/dev_pos_results.csv", index=False)
test_df.to_csv("results/test_pos_results.csv", index=False)
```

- Writes the two DataFrames to CSV files without the row index column.

---

## Cell 14 — summary tables for all algorithms (NB included, CSV output)

```python
dev_rows = build_summary_table([dev_fwd, dev_bwd, dev_fb, dev_nb])
test_rows = build_summary_table([test_fwd, test_bwd, test_fb, test_nb])
```

- Rebuilds the row lists to include the Naive Bayes result dict in addition to the three Viterbi variants. This overwrites the earlier HMM-only `dev_rows`/`test_rows`.

```python
dev_df = pd.DataFrame(dev_rows)
test_df = pd.DataFrame(test_rows)
```

- Rebuilds the DataFrames from the expanded row lists.

```python
dev_df
test_df
```

- Displays the updated tables (now 4 algorithms × 24 tags = 96 rows each).

```python
dev_df.to_csv("results/dev_pos_results.csv", index=False)
test_df.to_csv("results/test_pos_results.csv", index=False)
```

- Rewrites the same CSV paths with the all-algorithm tables, so the final CSVs include Naive Bayes.

---

## Cell 15 — F1 charts (HMM variants only)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
```

- `import matplotlib` — imports the plotting library.
- `matplotlib.use("Agg")` — sets the non-interactive backend so charts can be saved to files without a display.
- `import matplotlib.pyplot as plt` — imports the pyplot interface.
- `import numpy as np` — imports NumPy for the bar-position arithmetic.

```python
def plot_tag_f1(df, dataset_name):
    tags = sorted(tags_set)
    algos = ["Forward Viterbi", "Backward Viterbi", "Forward-Backward Viterbi"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tags))
    width = 0.25
    for i, algo in enumerate(algos):
        sub = df[df["Algorithm"] == algo].set_index("Tag").reindex(tags)
        f1s = [sub.loc[t, "F1"] for t in tags]
        ax.bar(x + (i - 1) * width, f1s, width, label=algo, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title(f"{dataset_name}: F1 per tag by algorithm")
    ax.legend()
    plt.tight_layout()
    path = f"results/{dataset_name.lower().replace(' ', '_')}_f1_chart.png"
    plt.savefig(path, dpi=150)
    print("Saved:", path)
```

- `def plot_tag_f1(df, dataset_name):` — draws a grouped bar chart of F1 per tag for the three HMM variants.
- `tags = sorted(tags_set)` — canonical tag order for the x-axis.
- `algos = [...]` — the three algorithm names to plot.
- `colors = [...]` — one colour per algorithm.
- `fig, ax = plt.subplots(figsize=(10, 6))` — creates a figure and axes of a readable size.
- `x = np.arange(len(tags))` — integer positions for each tag group.
- `width = 0.25` — bar width; three bars per group sit side by side.
- `for i, algo in enumerate(algos):` — for each algorithm:
  - `sub = df[df["Algorithm"] == algo].set_index("Tag").reindex(tags)` — extracts that algorithm’s rows and aligns them to the full tag order (missing tags become NaN, which bar renders as 0).
  - `f1s = [sub.loc[t, "F1"] for t in tags]` — the F1 values in tag order.
  - `ax.bar(x + (i - 1) * width, f1s, width, label=algo, color=colors[i])` — draws the bars offset by `width` per algorithm.
- `ax.set_xticks(x)` / `ax.set_xticklabels(tags, rotation=45, ha="right")` — labels each group with its tag, rotated for readability.
- `ax.set_ylabel("F1 score")` / `ax.set_title(...)` — axis label and title.
- `ax.legend()` — shows the algorithm legend.
- `plt.tight_layout()` — adjusts spacing so labels are not clipped.
- `path = f"results/{...}_f1_chart.png"` — constructs the output filename.
- `plt.savefig(path, dpi=150)` — renders the figure to a PNG at 150 DPI.
- `print("Saved:", path)` — confirms the output.

```python
plot_tag_f1(dev_df, "Development")
plot_tag_f1(test_df, "Test")
```

- Draws and saves the development and test charts for the three HMM variants. (These earlier charts exclude NB; they are regenerated in the next cell to include NB.)

---

## Cell 16 — F1 charts for all algorithms (NB included)

```python
def plot_tag_f1_all(df, dataset_name):
    tags = sorted(tags_set)
    algos = [
        "Naive Bayes",
        "Forward Viterbi",
        "Backward Viterbi",
        "Forward-Backward Viterbi",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tags))
    width = 0.2
    for i, algo in enumerate(algos):
        sub = df[df["Algorithm"] == algo].set_index("Tag").reindex(tags)
        f1s = [sub.loc[t, "F1"] for t in tags]
        ax.bar(x + (i - 1.5) * width, f1s, width, label=algo, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title(f"{dataset_name}: F1 per tag by algorithm")
    ax.legend()
    plt.tight_layout()
    path = f"results/{dataset_name.lower().replace(' ', '_')}_f1_chart.png"
    plt.savefig(path, dpi=150)
    print("Saved:", path)
```

- `def plot_tag_f1_all(df, dataset_name):` — same idea as `plot_tag_f1` but for all four algorithms.
- `algos = [...]` — now includes `"Naive Bayes"` as the first entry.
- `colors = [...]` — one colour per algorithm; the fourth colour (`#d62728`) is red for NB.
- `fig, ax = plt.subplots(figsize=(12, 6))` — slightly wider figure to fit four bars per group.
- `width = 0.2` — narrower bars so four fit per group.
- `ax.bar(x + (i - 1.5) * width, ...)` — offsets the four bars symmetrically around each tag position.
- The rest (axis labels, title, legend, save) is the same pattern as `plot_tag_f1`.

```python
plot_tag_f1_all(dev_df, "Development")
plot_tag_f1_all(test_df, "Test")
```

- Redraws and overwrites the two chart files so they now include Naive Bayes alongside the three Viterbi variants.

---

## Cell 17 — save Naive Bayes model

```python
with open("models/nb_model.pkl", "wb") as f:
    pickle.dump(
        {
            "nb_tag_counts": dict(nb_tag_counts),
            "nb_feat_cond_counts": {t: dict(nb_feat_cond_counts[t]) for t in nb_feat_cond_counts},
            "nb_feat_vocabs": nb_feat_vocabs,
            "nb_tag_total": nb_tag_total,
            "nb_num_tags": nb_num_tags,
            "NB_PRIOR_SMOOTH": NB_PRIOR_SMOOTH,
            "NB_LIKELIHOOD_SMOOTH": NB_LIKELIHOOD_SMOOTH,
        },
        f,
    )
```

- `with open("models/nb_model.pkl", "wb") as f:` — opens the output file in binary-write mode; the `with` block ensures it is closed.
- `pickle.dump({...}, f)` — serialises a dictionary containing everything needed to reconstruct the NB predictor:
  - `"nb_tag_counts"` — the per-tag counts (converted to a plain dict).
  - `"nb_feat_cond_counts"` — the per-tag feature-value counts (each inner Counter converted to a plain dict).
  - `"nb_feat_vocabs"` — the per-tag feature vocabulary sizes.
  - `"nb_tag_total"` — total training tokens.
  - `"nb_num_tags"` — number of tags.
  - `"NB_PRIOR_SMOOTH"` / `"NB_LIKELIHOOD_SMOOTH"` — the smoothing constants.

```python
print("Saved Naive Bayes model to models/nb_model.pkl")
```

- Prints a confirmation message.

---

## Cell 18 — save HMM model

```python
with open("models/hmm_model.pkl", "wb") as f:
    pickle.dump(
        {
            "word2idx": word2idx,
            "idx2word": idx2word,
            "tag2idx": tag2idx,
            "idx2tag": idx2tag,
            "transition_prob": transition_prob,
            "emission_prob": emission_prob,
            "tag_counts": tag_counts,
        },
        f,
    )
```

- `with open("models/hmm_model.pkl", "wb") as f:` — opens the HMM model file in binary-write mode.
- `pickle.dump({...}, f)` — serialises the HMM state:
  - `"word2idx"` / `"idx2word"` — word↔index mappings.
  - `"tag2idx"` / `"idx2tag"` — tag↔index mappings.
  - `"transition_prob"` — the transition probability dictionary.
  - `"emission_prob"` — the emission probability dictionary.
  - `"tag_counts"` — the per-tag occurrence counts.

```python
print("Saved HMM model to models/hmm_model.pkl")
```

- Prints a confirmation message.

---

## Output artifacts

After the notebook runs end to end, these files are produced:

- `Assignment_7/results/dev_pos_results.csv` — per-tag F1/gold/predicted/correct for all four algorithms on the dev set (96 rows = 4 algorithms × 24 tags).
- `Assignment_7/results/test_pos_results.csv` — same for the test set.
- `Assignment_7/results/development_f1_chart.png` — grouped bar chart of F1 per tag for all four algorithms on dev.
- `Assignment_7/results/test_f1_chart.png` — same for test.
- `Assignment_7/models/nb_model.pkl` — pickled Naive Bayes model.
- `Assignment_7/models/hmm_model.pkl` — pickled HMM model.

---

## Notes on the Naive Bayes “previous tag” feature

The assignment specifies “previous tag” as one of the five features. In a fully sequential setting the previous tag is not known at prediction time. In this notebook:

- During training, the previous tag is taken from the gold training tags, so the feature counts reflect the true gold context.
- During dev/test evaluation, two predictors exist:
  - `nb_predict_sentence` (used via `predict_nb_fn` for the scored results) supplies the gold previous tag at each position — this matches a supervised feature-based tagger evaluation where the previous tag feature is conditioned on gold context.
  - `nb_predict_sentence_eval` (the standalone evaluation wrapper) substitutes `"<unknown>"` for the previous tag except at position 0, demonstrating a fully incremental prediction path.

The scored numbers in the CSVs and charts come from the gold-previous-tag version, consistent with the feature list in the assignment.