# Naive Bayes Benchmark Report

## Frozen test metrics (text)

| dataset | model | accuracy | macro-F1 | log loss | ECE |
| --- | --- | --- | --- | --- | --- |
| 20newsgroups | bernoulli | 0.5610 | 0.5487 | 8.6387 | 0.3902 |
| 20newsgroups | multinomial | 0.6583 | 0.6353 | 7.9965 | 0.3124 |
| 20newsgroups_nodup | bernoulli | 0.5597 | 0.5477 | 8.6509 | 0.3912 |
| 20newsgroups_nodup | multinomial | 0.6574 | 0.6344 | 8.0280 | 0.3139 |
| sms_spam | bernoulli | 0.9866 | 0.9701 | 0.1316 | 0.0111 |
| sms_spam | multinomial | 0.9878 | 0.9735 | 0.1483 | 0.0097 |

## Gaussian tabular (10 seeded 70/30 splits)

| dataset | accuracy (mean [CI]) | macro-F1 (mean [CI]) |
| --- | --- | --- |
| iris | 0.9489 [0.9179, 0.9798] | 0.9488 [0.9177, 0.9798] |
| wine | 0.9741 [0.9672, 0.9809] | 0.9750 [0.9683, 0.9817] |
| breast_cancer | 0.9427 [0.9333, 0.9521] | 0.9380 [0.9278, 0.9482] |

## Failure analysis (frozen test slices)

Slices reuse the single frozen test read (no refits). Length tertiles use cut points from train token counts; OOV tertiles use cut points from validation OOV rates against the train unigram vocabulary.

### 20newsgroups / bernoulli

| slice | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| length tertile 1 | 2347 | 0.3336 | 0.2511 |
| length tertile 2 | 2554 | 0.6241 | 0.5962 |
| length tertile 3 | 2416 | 0.7152 | 0.6898 |
| oov tertile 1 | 2077 | 0.5397 | 0.5327 |
| oov tertile 2 | 2496 | 0.5962 | 0.5766 |
| oov tertile 3 | 2744 | 0.5452 | 0.5183 |

Confidence bins (count / accuracy / mean confidence): 4:2/0.000/0.329; 5:11/0.091/0.379; 6:32/0.188/0.436; 7:93/0.247/0.508; 8:141/0.213/0.567; 9:127/0.283/0.634; 10:154/0.156/0.702; 11:170/0.212/0.769; 12:215/0.233/0.834; 13:318/0.242/0.904; 14:6054/0.631/0.996

Most confused class pairs (20NG):
- 2 -> 3: 129
- 18 -> 16: 73
- 2 -> 4: 71
- 5 -> 1: 70
- 15 -> 0: 64

### 20newsgroups / multinomial

| slice | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| length tertile 1 | 2347 | 0.5347 | 0.5102 |
| length tertile 2 | 2554 | 0.6817 | 0.6545 |
| length tertile 3 | 2416 | 0.7537 | 0.7221 |
| oov tertile 1 | 2077 | 0.6514 | 0.6320 |
| oov tertile 2 | 2496 | 0.6783 | 0.6527 |
| oov tertile 3 | 2744 | 0.6454 | 0.6111 |

Confidence bins (count / accuracy / mean confidence): 0:9/0.000/0.054; 1:3/0.000/0.105; 2:3/0.000/0.139; 3:10/0.100/0.244; 4:9/0.333/0.305; 5:16/0.125/0.366; 6:16/0.188/0.441; 7:49/0.143/0.507; 8:58/0.241/0.563; 9:76/0.237/0.632; 10:80/0.362/0.700; 11:96/0.271/0.772; 12:95/0.337/0.835; 13:151/0.325/0.904; 14:6646/0.697/0.998

Most confused class pairs (20NG):
- 2 -> 3: 155
- 18 -> 16: 82
- 19 -> 15: 75
- 2 -> 1: 64
- 0 -> 15: 63

### sms_spam / bernoulli

| slice | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| length tertile 1 | 205 | 0.9902 | 0.7475 |
| length tertile 2 | 308 | 0.9935 | 0.9428 |
| length tertile 3 | 307 | 0.9772 | 0.9732 |
| oov tertile 1 | 252 | 1.0000 | 1.0000 |
| oov tertile 2 | 265 | 0.9849 | 0.9727 |
| oov tertile 3 | 303 | 0.9769 | 0.9584 |

Confidence bins (count / accuracy / mean confidence): 8:1/0.000/0.576; 9:1/1.000/0.619; 10:1/0.000/0.686; 12:1/0.000/0.806; 13:4/0.750/0.891; 14:812/0.991/1.000

### sms_spam / multinomial

| slice | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| length tertile 1 | 205 | 0.9902 | 0.8725 |
| length tertile 2 | 308 | 0.9935 | 0.9483 |
| length tertile 3 | 307 | 0.9805 | 0.9771 |
| oov tertile 1 | 252 | 0.9921 | 0.9563 |
| oov tertile 2 | 265 | 0.9849 | 0.9732 |
| oov tertile 3 | 303 | 0.9868 | 0.9768 |

Confidence bins (count / accuracy / mean confidence): 8:2/0.500/0.567; 11:3/1.000/0.782; 12:6/0.833/0.851; 13:3/0.667/0.910; 14:806/0.991/0.999

git_sha: `ac5aafdf82d3594c3077674cbbaac57136bdc173`
ordering_ok: True
analysis_plan_sha256: `9351ac766b42902a9258addebc907be6308a8d3a584b34539a82e6fe3ab9f5b0`
