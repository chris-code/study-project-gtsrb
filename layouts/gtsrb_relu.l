[
  [
    "conv2D", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "input_shape": [
        3, 
        48, 
        48
      ], 
      "nb_col": 7, 
      "nb_filter": 100, 
      "nb_row": 7
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
      "activation": "relu", 
      "init": "uniform", 
      "nb_col": 4, 
      "nb_filter": 150, 
      "nb_row": 4
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
      "activation": "relu", 
      "init": "uniform", 
      "nb_col": 4, 
      "nb_filter": 250, 
      "nb_row": 4
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
    {}
  ], 
  [
    "dense", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "output_dim": 300
    }
  ], 
  [
    "dense", 
    {
      "activation": "softmax", 
      "init": "uniform", 
      "output_dim": 43
    }
  ]
]