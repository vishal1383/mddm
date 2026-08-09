# All-States Confidence Training V2

## Hypothesis

An anchor is useful only if the pre-anchor forward predicts both the catalyst
and its unlocked set strongly enough for inference to commit them together.
V1 ranked these targets but supervised only 13.85% of cached states and did not
directly enforce the inference confidence boundary.

## Training

V2 streams every cached pre-anchor state in each selected record. For every
state, the target set is the gold catalyst plus its complete cached U_after set.
The loss is:

~~~text
anchor CE + unlocked CE + confidence-margin loss
  + 0.25 * position-selection loss + 5 * frozen-base KL
~~~

The confidence loss requires gold-vs-rest log odds of at least
`log(0.70 / 0.30)`. Catalyst and unlocked CE are averaged separately, so a
large unlock burst cannot erase catalyst supervision.

## Calibration

The smoke run trains on the first 500 GSM8K training records and evaluates on
held-out training records beginning at index 7000. It compares trained
thresholds 0.70, 0.80, 0.90, and 0.95 against base LLaDA at 0.95. A candidate
must reach at least 4 tokens per forward while remaining within two accuracy
points of base on the held-out slice.

This is only the calibration gate. A final improvement claim additionally
requires paired full-test accuracy to be non-inferior and measured wall-clock
latency to improve.

~~~bash
bash Token2Token/run_all_states_v2_smoke.sh
~~~
