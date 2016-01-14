[
  [
    "conv2D", 
    {
      "init": "uniform", 
      "input_shape": [
        3, 
        48, 
        48
      ], 
      "nb_col": 5, 
      "nb_filter": 20, 
      "nb_row": 5
    }
  ], 
  [
    "maxpool2D", 
    {
      "pool_size": [
        2, 
        2
      ]
    }
  ], 
  [
    "conv2D", 
    {
      "init": "uniform", 
      "nb_col": 5, 
      "nb_filter": 20, 
      "nb_row": 5
    }
  ], 
  [
    "maxpool2D", 
    {
      "pool_size": [
        2, 
        2
      ]
    }
  ], 
  [
    "flatten", 
    null
  ], 
  [
    "dense", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "output_dim": 500
    }
  ], 
  [
    "dense", 
    {
      "activation": "softmax", 
      "init": "uniform", 
      "output_dim": 100
    }
  ]
]