# Edit flow progress

## Training over code correction samples
- Several days to train, clear decrease in training loss, less clear decrease in validation loss
- BLEU score only tested on small validation set (5) for speed, so not very reliable. Also sequences are highly similar,
so mostly serves as a sanity check.
 
## Sampling
- Performs worse with more samples
- Given the very low changes between sequences, more samples seems to add noise, 
with more chances for it to diverge from the correct sequence.
- Training set with low (100) samples: Often seems to solve the task. If not, often gets the right area of the code.
  (unsurprising, just memorizing)
- Val set, (100 samples): Far fewer solutions, often gets right area of code. Corrections are often in style of 
fixing errors (swapping 0 and 1, changing boolean conditions).

## Takeaways
- Clearly learning (high correctness on training samples, more 'sane' outputs on validation, 
rather than randomness from before (i.e. structure often in edits, even if not correct))
- Might want better metrics (correct area, how much of added code is present, 
how much of removed code is gone, how much extra code is added)
- Sampling hard problem, might need to figure out how to do reverse sample correction from paper
- Training longer could help
- Could add prediction of progress (e.g. predict how many more edits needed, or how far along we are)? 
  - Some related work on this

## Next steps
- Add better metrics
- Experiment a bit more with sampling
- Document code 
- Determine whether to continue or move on to thesis
- Datasets for code + error to fix