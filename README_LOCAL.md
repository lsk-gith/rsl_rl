# 机器
```aiignore
172.24.0.166
```
# 位置
```aiignore
/data1/lsk
```
# 测试命令
```aiignore

PYTHONPATH=/data1/lsk/rsl_rl
 /data1/anaconda3/envs/py310/bin/pytest -v -s tests/algorithms/test_ppo.py
 
 输出
 ======================================================================= test session starts ========================================================================
platform linux -- Python 3.10.18, pytest-9.0.2, pluggy-1.6.0 -- /data1/anaconda3/envs/py310/bin/python3.10
cachedir: .pytest_cache
rootdir: /data1/lsk/rsl_rl
configfile: pyproject.toml
collected 9 items                                                                                                                                                  

tests/algorithms/test_ppo.py::TestGAEComputation::test_gae_returns_hand_computed PASSED                                                                      [ 11%]
tests/algorithms/test_ppo.py::TestGAEComputation::test_gae_terminal_state_cuts_bootstrap PASSED                                                              [ 22%]
tests/algorithms/test_ppo.py::TestGAEComputation::test_advantage_normalization_global PASSED                                                                 [ 33%]
tests/algorithms/test_ppo.py::TestTimeoutBootstrapping::test_timeout_adds_bootstrap_to_reward PASSED                                                         [ 44%]
tests/algorithms/test_ppo.py::TestPPOLosses::test_surrogate_loss_clipping PASSED                                                                             [ 55%]
tests/algorithms/test_ppo.py::TestPPOLosses::test_value_loss_clipping PASSED                                                                                 [ 66%]
tests/algorithms/test_ppo.py::TestAdaptiveLearningRate::test_lr_decreases_when_kl_too_high PASSED                                                            [ 77%]
tests/algorithms/test_ppo.py::TestAdaptiveLearningRate::test_lr_increases_when_kl_too_low PASSED                                                             [ 88%]
tests/algorithms/test_ppo.py::TestAdaptiveLearningRate::test_lr_unchanged_in_stable_range PASSED                                                             [100%]

======================================================================== 9 passed in 0.12s =========================================================================
 
 
 /data1/anaconda3/envs/py310/bin/pytest -v -s tests/runners/test_on_policy_runner.py 

 tests/runners/test_on_policy_runner.py::TestCNNRunner::test_cnn_save_load_restores_parameters --------------------------------------------------------------------------------
Resolved observation sets: 
         actor :  ['policy', 'image']
         critic :  ['policy', 'image']
--------------------------------------------------------------------------------
Actor Model: CNNModel(
  (obs_normalizer): Identity()
  (distribution): GaussianDistribution()
  (mlp): MLP(
    (0): Linear(in_features=204, out_features=32, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=32, out_features=4, bias=True)
  )
  (cnns): ModuleDict(
    (image): CNN(
      (0): Conv2d(1, 4, kernel_size=(3, 3), stride=(2, 2))
      (1): ELU(alpha=1.0)
      (2): Flatten(start_dim=1, end_dim=-1)
    )
  )
)
Critic Model: CNNModel(
  (obs_normalizer): Identity()
  (mlp): MLP(
    (0): Linear(in_features=204, out_features=32, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=32, out_features=1, bias=True)
  )
  (cnns): ModuleDict(
    (image): CNN(
      (0): Conv2d(1, 4, kernel_size=(3, 3), stride=(2, 2))
      (1): ELU(alpha=1.0)
      (2): Flatten(start_dim=1, end_dim=-1)
    )
  )
)
PASSED
 
 
 
 
```