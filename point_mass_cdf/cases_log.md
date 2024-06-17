This file is used to log some good cases and related hyperparameters.

### case 1
controller:

  Tmax: 20

  step_time: 0.1

  weight_input: [1.2, 1.0]

  smooth_input: [1.0, 1.0]

  weight_slack: 100

  clf_lambda: 1.0

  cbf_gamma: 0.9

robot:
  u_max: [2.7, 2.7]

  u_min: [-2.7, -2.7]

  initial_state: [2.5, -0.3, 0.0]

  target_state: [-2.3, 0.6, 0.0]

  radius: 0.1

  destination_margin: 0.1

  sensor_range: 6.0

  margin: 0.1

  obs_states: [2.5, -2.35]
  
  obs_radii: [0.3]

  obs_vel: [0.0, 0.5]
