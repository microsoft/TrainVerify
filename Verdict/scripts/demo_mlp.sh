#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

python main.py --sm gen_model/mgeners/mlp_default_dp1_pp1_tp1_nm1_gbs128_ly2_h0_hi128_sq0.pkl --pm gen_model/mgeners/mlp_default_dp2_pp2_tp2_nm2_gbs128_ly2_h0_hi128_sq0.pkl --seed 0 --time  --max_ser_proc 30 --max_vrf_proc 30 --loglevel INFO --no_cache_nodes --no_cache_stages |& tee -a data/logs/mlp_default_dp2_pp2_tp2_nm2_gbs128_ly2_h0_hi128_sq0.txt

