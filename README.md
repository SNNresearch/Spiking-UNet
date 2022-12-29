## ANN train
If you train ANN network on the CBSD, please run
`python3 ann_train.py -n CBSD -T `

If you test ANN results on the CBSD, please run 
`python3 ann_train.py -n CBSD `

The format:

`python3 ann_train.py -n dataset_name [-s1 seed1] [-s2 seed2] [-s3 seed3] [-b batch_size] [-e epochs] [-op optimizer] [-lr leanring_rate] [-T]`

The parameters of ann_train.py are as follow:

-n:  BSD / CBSD

-s1: int

-s2: int

-s3: int

-b:  int

-e:  int

-op: adam / sgd / rms / adadelta

-lr: float

-T:  '-T' represents train mode

## evaluate converted SNN
`python3 ann_snn_denoising.py -n dataset_name -t timesteps [-m method] [-s scale_method] [-d]`

The parameters of ann_snn_denoising.py are as follow:

-n: Set12 / BSD / CBSD

-t: int

-m: layer_wise / connection_wise

-s: robust / max   (robust: 99.9 percentile of activations; max: max of activations)

-d: '-d' represents SNN with "reduce by subtraction mechanism"

-neuron: multi / IF

## SNN train
`python3 snn_train.py -n dataset_name -t timesteps [-m method] [-s1 seed1] [-s2 seed2] [-s3 seed3] [-b batch_size] [-e epochs] [-lr leanring_rate] [-s scale_method] [-op optimizer] [-T] [-d]`

The parameters of snn_train.py are as follow:

-n: BSD / CBSD

-t: int

-s1: int

-s2: int

-s3: int

-b:  int

-e:  int

-lr: float

-op: adam / sgd / rms / adadelta

-T: '-T' represents train mode

-s: robust / max   (robust: 99.9 percentile of activations; max: max of activations)

-m: layer_wise / connection_wise

-d: '-d' represents SNN with "reduce by subtraction mechanism"

-neuron: multi / IF
